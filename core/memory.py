"""
Guardian AI — Long-Term Memory System
=======================================

Manages two memory tiers for the classifier:

1. SHORT-TERM BUFFER:
   Rolling window of the last N utterances (raw conversation text).
   This is the "recent context" the classifier sees.

2. LONG-TERM MEMORY:
   Stores utterances flagged as threatening (score above a threshold)
   with timestamps. These persist even after the short-term buffer
   rolls over, giving the classifier historical context.

The memory formats both tiers into a single string that fits within
Longformer's 4096-token limit for classifier input.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════
@dataclass
class Utterance:
    """A single transcribed utterance with metadata."""
    text: str                           # the transcribed text
    timestamp: float                    # when it was spoken (time.time())
    threat_score: float = 0.0           # classifier's threat probability

    def age_seconds(self) -> float:
        """How many seconds ago this was spoken."""
        return time.time() - self.timestamp

    def formatted_time(self) -> str:
        """Human-readable timestamp (HH:MM:SS)."""
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


# ══════════════════════════════════════════════════════════════════
#  MEMORY MANAGER
# ══════════════════════════════════════════════════════════════════
class MemoryManager:
    """
    Manages short-term and long-term conversation memory.

    Parameters
    ----------
    short_term_max : int
        Maximum utterances to keep in the short-term buffer.
        Oldest are discarded when this limit is reached.

    long_term_threat_threshold : float
        Minimum threat score for an utterance to be stored in
        long-term memory. Default 0.4 (moderately concerning).

    long_term_max : int
        Maximum items in long-term memory. Oldest discarded first.

    long_term_expiry_seconds : float
        How long safe long-term memories persist before cleanup.
        If threat stayed low for this duration, old memories are pruned.

    max_context_tokens : int
        Approximate max tokens in the formatted context string.
        Keeps us within Longformer's 4096 limit (with some margin).
    """

    def __init__(
        self,
        short_term_max: int = 20,
        long_term_threat_threshold: float = 0.4,
        long_term_max: int = 50,
        long_term_expiry_seconds: float = 600.0,  # 10 minutes
        max_context_tokens: int = 450,             # MuRIL limit: 512 minus special tokens
    ):
        self.short_term_max = short_term_max
        self.long_term_threat_threshold = long_term_threat_threshold
        self.long_term_max = long_term_max
        self.long_term_expiry_seconds = long_term_expiry_seconds
        self.max_context_tokens = max_context_tokens

        # ── Storage ────────────────────────────────────────────
        self.short_term: list[Utterance] = []
        self.long_term: list[Utterance] = []

    def add_utterance(self, text: str, threat_score: float = 0.0) -> Utterance:
        """
        Add a new transcribed utterance to memory.

        - Always added to short-term buffer (rolls over when full)
        - Added to long-term memory ONLY if threat_score >= threshold

        Parameters
        ----------
        text : str
            The transcribed text.
        threat_score : float
            Threat probability from the classifier (0.0-1.0).

        Returns
        -------
        Utterance
            The created utterance object.
        """
        utterance = Utterance(
            text=text.strip(),
            timestamp=time.time(),
            threat_score=threat_score,
        )

        # ── Add to short-term buffer ──────────────────────────
        self.short_term.append(utterance)
        if len(self.short_term) > self.short_term_max:
            self.short_term.pop(0)  # remove oldest

        # ── Add to long-term if threatening ────────────────────
        if threat_score >= self.long_term_threat_threshold:
            self.long_term.append(utterance)
            if len(self.long_term) > self.long_term_max:
                self.long_term.pop(0)

        return utterance

    def get_context_for_classifier(self) -> str:
        """
        Format memory into a single string for the classifier.

        The output format is:
          [MEMORY] threatening text 1 | threatening text 2 [RECENT] recent utterance 1. recent utterance 2. ...

        This lets the classifier see both:
          - Historical red flags (long-term memory)
          - Current conversation context (short-term buffer)

        The total length is capped to stay within the token limit.
        """
        parts = []

        # ── Long-term memory section ──────────────────────────
        if self.long_term:
            lt_texts = []
            for utt in self.long_term:
                lt_texts.append(f"[{utt.formatted_time()} threat={utt.threat_score:.2f}] {utt.text}")
            lt_section = " | ".join(lt_texts)
            parts.append(f"[MEMORY] {lt_section}")

        # ── Short-term buffer section ─────────────────────────
        if self.short_term:
            st_texts = [utt.text for utt in self.short_term]
            st_section = ". ".join(st_texts)
            parts.append(f"[RECENT] {st_section}")

        context = " ".join(parts)

        # ── Truncate if too long ──────────────────────────────
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = self.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[-max_chars:]  # keep the most recent part

        return context

    def get_short_term_text(self) -> str:
        """Get just the short-term buffer as plain text."""
        return " | ".join(utt.text for utt in self.short_term)

    def cleanup(self, current_threat_level: float):
        """
        Prune old long-term memories when situation is safe.

        Only cleans up if the current threat level is low, meaning
        the situation has calmed down. This prevents deleting
        important memories during an ongoing incident.

        Parameters
        ----------
        current_threat_level : float
            Current accumulated threat level from the ThreatScorer.
        """
        if current_threat_level > 0.3:
            return  # don't clean up during elevated threat

        now = time.time()
        self.long_term = [
            utt for utt in self.long_term
            if (now - utt.timestamp) < self.long_term_expiry_seconds
        ]

    def get_debug_info(self) -> str:
        """Return a formatted debug string showing all memory state."""
        lines = [
            f"  ┌─ MEMORY ────────────────────────────────────────────┐",
        ]

        # Short-term buffer
        lines.append(f"  │  SHORT-TERM BUFFER ({len(self.short_term)}/{self.short_term_max} utterances):")
        if self.short_term:
            # Show last 5 utterances max in debug view
            display = self.short_term[-5:]
            for utt in display:
                age = utt.age_seconds()
                text_preview = utt.text[:45] + "…" if len(utt.text) > 45 else utt.text
                score_color = "\033[91m" if utt.threat_score > 0.5 else "\033[92m"
                lines.append(
                    f"  │    {utt.formatted_time()} "
                    f"{score_color}[{utt.threat_score:.2f}]{RESET} "
                    f"{text_preview}"
                )
            if len(self.short_term) > 5:
                lines.append(f"  │    ... and {len(self.short_term) - 5} older utterances")
        else:
            lines.append(f"  │    (empty)")

        # Long-term memory
        lines.append(f"  │  ")
        lines.append(f"  │  LONG-TERM MEMORY ({len(self.long_term)}/{self.long_term_max} items):")
        if self.long_term:
            for utt in self.long_term[-5:]:
                text_preview = utt.text[:45] + "…" if len(utt.text) > 45 else utt.text
                lines.append(
                    f"  │    ⚠ {utt.formatted_time()} "
                    f"\033[91m[{utt.threat_score:.2f}]{RESET} "
                    f"{text_preview}"
                )
            if len(self.long_term) > 5:
                lines.append(f"  │    ... and {len(self.long_term) - 5} older entries")
        else:
            lines.append(f"  │    (empty — no threats recorded)")

        lines.append(f"  └───────────────────────────────────────────────────┘")
        return "\n".join(lines)


RESET = "\033[0m"


# ══════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """Demo: simulate utterances being added to memory."""
    mem = MemoryManager()

    test_data = [
        ("hi how are you doing today", 0.05),
        ("I'm good just walking home", 0.03),
        ("nice weather isn't it", 0.02),
        ("hey where are you going", 0.25),
        ("none of your business please leave me alone", 0.55),
        ("come on don't be like that", 0.48),
        ("I said stop following me", 0.72),
        ("someone help me please", 0.91),
    ]

    print("\n  Simulating conversation with threat scores:\n")
    for text, score in test_data:
        mem.add_utterance(text, score)
        print(f"  New utterance: \"{text}\" (score={score:.2f})")
        print(mem.get_debug_info())
        print()

    print("\n  Context string for classifier:")
    print(f"  {mem.get_context_for_classifier()}")
