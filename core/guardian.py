"""
Guardian AI — Main Entry Point
================================

Ties together all modules:
  🎙 Ear:    speech-detection.py (Whisper mic transcription)
  🧠 Brain:  offline-classifier.py (Longformer threat classifier)
  📊 Scorer: threat_scorer.py (EMA threat accumulation)
  💾 Memory: memory.py (short-term buffer + long-term store)

Runs the full pipeline with a live debug dashboard showing all
internal state in real-time: utterances, threat scores, memory,
and state transitions.

Usage:
  python guardian.py
"""

import sys
import os
import time
import threading
import importlib

# Import Guardian AI modules (all in core/ package now)
from core.speech_detection import start_transcription
from core.offline_classifier import SafetyClassifier
from core.threat_scorer import ThreatScorer, ThreatState, STATE_COLORS, RESET
from core.memory import MemoryManager


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
# Set to True to use the real Longformer classifier
# Set to False to use a keyword-based mock (for testing without model)
USE_REAL_CLASSIFIER = True

# Threat scorer parameters
ALPHA = 0.5               # EMA smoothing (higher = more reactive)
DECAY_RATE = 0.95          # Per-second decay during silence
PANIC_THRESHOLD = 0.95     # Instant emergency on this score
HYSTERESIS_GAP = 0.10      # De-escalation gap


# ══════════════════════════════════════════════════════════════════
#  MOCK CLASSIFIER (for testing before fine-tuning)
# ══════════════════════════════════════════════════════════════════
class MockClassifier:
    """
    A simple keyword-based classifier for testing the pipeline
    BEFORE the Longformer model is fine-tuned.

    Uses keyword matching to approximate threat detection.
    Replace this with SafetyClassifier once fine-tuning is done.
    """

    # Keywords indicating potential danger (women's safety focus)
    EMERGENCY_KEYWORDS = [
        # Direct distress
        "help", "help me", "save me", "please help",
        "let me go", "leave me alone", "stop it",
        "don't touch", "stop touching", "get away",
        "get off me", "back off",
        # Following / stalking
        "following me", "following", "stalking",
        "watching me", "someone is behind",
        # Threats / violence
        "kill", "hurt", "hit me", "attack",
        "threatening", "weapon", "knife", "gun",
        # Fear / distress
        "scared", "afraid", "terrified", "fear",
        "scream", "screaming", "crying",
        # Assault / harassment
        "touching me", "molest", "assault",
        "harass", "rape", "grab",
        # Captivity
        "trapped", "locked", "can't leave", "won't let me",
        "kidnap", "abduct",
    ]

    # Keywords indicating safety (reduce score)
    SAFE_KEYWORDS = [
        "movie", "game", "show", "song", "book",
        "joke", "funny", "laugh", "playing",
        "grocery", "weather", "lunch", "dinner",
        "coffee", "friend", "happy", "good",
    ]

    def classify(self, text: str) -> float:
        """Return a mock threat probability based on keywords."""
        text_lower = text.lower()
        score = 0.05  # base score

        # Count emergency keywords
        for kw in self.EMERGENCY_KEYWORDS:
            if kw in text_lower:
                score += 0.25

        # Reduce for safe keywords
        for kw in self.SAFE_KEYWORDS:
            if kw in text_lower:
                score -= 0.15

        return max(0.0, min(1.0, score))

    def classify_with_details(self, text: str) -> dict:
        prob = self.classify(text)
        return {
            "emergency_prob": prob,
            "safe_prob": 1.0 - prob,
            "predicted_label": "EMERGENCY" if prob > 0.5 else "SAFE",
            "token_count": len(text.split()),
        }


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD DISPLAY
# ══════════════════════════════════════════════════════════════════
def print_dashboard(
    utterance_text: str,
    utterance_num: int,
    classification: dict,
    scorer: ThreatScorer,
    memory: MemoryManager,
):
    """
    Print a comprehensive debug dashboard showing all system state.

    This is the main output the user sees during testing.
    """
    # Clear some space
    print(f"\n{'═' * 70}")
    print(f"  🎙  UTTERANCE #{utterance_num}")
    print(f"{'─' * 70}")

    # ── What was said ──────────────────────────────────────────
    print(f"  Speech:  \033[97m{utterance_text}\033[0m")

    # ── Classification result ──────────────────────────────────
    prob = classification["emergency_prob"]
    label = classification["predicted_label"]
    label_color = "\033[91m" if label == "EMERGENCY" else "\033[92m"
    print(f"\n  ┌─ CLASSIFIER ──────────────────────────────────────┐")
    print(f"  │  Prediction:  {label_color}{label:10s}\033[0m"
          f"                           │")
    print(f"  │  Emergency:   {prob:.1%}         "
          f"Safe: {classification['safe_prob']:.1%}              │")
    print(f"  │  Tokens used: {classification['token_count']}"
          f"                                     │")
    print(f"  └──────────────────────────────────────────────────┘")

    # ── Threat scorer state ────────────────────────────────────
    print(scorer.get_debug_info())

    # ── Memory state ───────────────────────────────────────────
    print(memory.get_debug_info())

    print(f"{'═' * 70}\n")


def print_state_change(old_state: ThreatState, new_state: ThreatState, level: float):
    """Print a prominent alert when threat state changes."""
    color = STATE_COLORS.get(new_state, "")
    if new_state.value in ("ALERT", "EMERGENCY"):
        print(f"\n  {'⚡' * 20}")
        print(f"  {color}  !! STATE CHANGE: {old_state.value} → {new_state.value} "
              f"(level={level:.3f}) !!{RESET}")
        print(f"  {'⚡' * 20}\n")
    else:
        print(f"\n  ⚡ State: {old_state.value} → {color}{new_state.value}{RESET} "
              f"(level={level:.3f})\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN GUARDIAN LOOP
# ══════════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║         🛡️  GUARDIAN AI — SAFETY MONITOR  🛡️         ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    # ── Initialize Classifier ──────────────────────────────────
    if USE_REAL_CLASSIFIER:
        print("  Loading Longformer classifier …")
        classifier = SafetyClassifier()
        classifier.load()
    else:
        print("  ⚠️  Using MOCK classifier (keyword-based)")
        print("     Set USE_REAL_CLASSIFIER=True after fine-tuning.\n")
        classifier = MockClassifier()

    # ── Initialize Threat Scorer ───────────────────────────────
    scorer = ThreatScorer(
        alpha=ALPHA,
        decay_rate=DECAY_RATE,
        panic_threshold=PANIC_THRESHOLD,
        hysteresis_gap=HYSTERESIS_GAP,
        on_state_change=print_state_change,
    )

    # ── Initialize Memory ─────────────────────────────────────
    memory = MemoryManager(
        short_term_max=20,
        long_term_threat_threshold=0.4,
        long_term_max=50,
    )

    # ── Decay thread ───────────────────────────────────────────
    # Runs in background to apply time decay during silence
    decay_running = True

    def decay_loop():
        while decay_running:
            time.sleep(1.0)
            scorer.apply_decay()
            # Cleanup old safe memories periodically
            memory.cleanup(scorer.threat_level)

    decay_thread = threading.Thread(target=decay_loop, daemon=True)
    decay_thread.start()

    # ── Callback: called by speech-detection on each utterance ─
    def on_utterance(text: str, utterance_num: int):
        """
        Called by the ear module when a new utterance is transcribed.
        This is where the brain pipeline runs:
          text → memory → classifier → scorer → dashboard
        """
        # Step 1: Get context from memory (for classifier input)
        context = memory.get_context_for_classifier()
        classifier_input = f"{context} {text}" if context else text

        # Step 2: Classify the context + new utterance
        classification = classifier.classify_with_details(classifier_input)
        threat_prob = classification["emergency_prob"]

        # Step 3: Update memory with the new utterance + its score
        memory.add_utterance(text, threat_prob)

        # Step 4: Feed score to threat accumulator
        state = scorer.update(threat_prob)

        # Step 5: Print the full debug dashboard
        print_dashboard(text, utterance_num, classification, scorer, memory)

        # Step 6: Handle emergency actions (placeholder for now)
        if state == ThreatState.EMERGENCY:
            print("  🚨🚨🚨  EMERGENCY ACTIONS WOULD TRIGGER HERE  🚨🚨🚨")
            print("  (Emergency response module not yet implemented)\n")

    # ── Start the ear ──────────────────────────────────────────
    print("  Starting speech detection …\n")
    try:
        start_transcription(on_utterance=on_utterance)
    except KeyboardInterrupt:
        pass
    finally:
        decay_running = False

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  🛡️  GUARDIAN AI — SESSION SUMMARY")
    print(f"{'─' * 70}")
    print(f"  Final threat state: {scorer.current_state.value}")
    print(f"  Final threat level: {scorer.threat_level:.3f}")
    print(f"  Utterances in short-term: {len(memory.short_term)}")
    print(f"  Items in long-term memory: {len(memory.long_term)}")
    print()
    if memory.long_term:
        print(f"  ⚠️  Flagged utterances:")
        for utt in memory.long_term:
            print(f"     [{utt.formatted_time()}] ({utt.threat_score:.2f}) {utt.text}")
    else:
        print(f"  ✅ No threatening utterances recorded.")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
