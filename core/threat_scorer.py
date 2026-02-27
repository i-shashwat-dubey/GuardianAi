"""
Guardian AI — Threat Scoring Engine
====================================

Takes raw threat probabilities from the classifier and manages
a graduated threat level using:
  - Exponential Moving Average (EMA) for smooth accumulation
  - Time-based decay when situation calms
  - Instant override for extreme threats
  - Hysteresis to prevent state flickering

Threat States:
  SAFE     (0.0 - 0.3)  → Do nothing
  MONITOR  (0.3 - 0.6)  → Heighten sensitivity, start logging
  ALERT    (0.6 - 0.8)  → Prepare emergency actions
  EMERGENCY(0.8 - 1.0)  → Full trigger: alarm, SOS, record
"""

import time
from enum import Enum
from typing import Callable, Optional


# ══════════════════════════════════════════════════════════════════
#  THREAT STATES
# ══════════════════════════════════════════════════════════════════
class ThreatState(Enum):
    """The four graduated threat levels."""
    SAFE      = "SAFE"
    MONITOR   = "MONITOR"
    ALERT     = "ALERT"
    EMERGENCY = "EMERGENCY"


# Color codes for terminal display
STATE_COLORS = {
    ThreatState.SAFE:      "\033[92m",   # green
    ThreatState.MONITOR:   "\033[93m",   # yellow
    ThreatState.ALERT:     "\033[91m",   # red
    ThreatState.EMERGENCY: "\033[95;1m", # bright magenta bold
}
RESET = "\033[0m"


# ══════════════════════════════════════════════════════════════════
#  THREAT SCORER
# ══════════════════════════════════════════════════════════════════
class ThreatScorer:
    """
    Maintains a running threat level that accumulates, decays,
    and triggers state transitions.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (0-1). Higher = more reactive to new scores.
        0.5 means new score and history are weighted equally.

    decay_rate : float
        How fast threat drops per second during silence. 0.95 means
        threat retains 95% of its value each second of no input.

    panic_threshold : float
        If a single raw score exceeds this, jump straight to EMERGENCY.

    hysteresis_gap : float
        How far below a threshold the score must drop to de-escalate.
        Prevents rapid flickering between states.

    on_state_change : callable, optional
        Callback invoked when threat state changes.
        Signature: on_state_change(old_state, new_state, threat_level)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        decay_rate: float = 0.95,
        panic_threshold: float = 0.95,
        hysteresis_gap: float = 0.10,
        on_state_change: Optional[Callable] = None,
    ):
        # ── Configuration ──────────────────────────────────────
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.panic_threshold = panic_threshold
        self.hysteresis_gap = hysteresis_gap
        self.on_state_change = on_state_change

        # ── State ──────────────────────────────────────────────
        self.threat_level: float = 0.0
        self.current_state: ThreatState = ThreatState.SAFE
        self.last_update_time: float = time.time()
        self.last_raw_score: float = 0.0

        # ── History (for debug display) ────────────────────────
        self.score_history: list[tuple[float, float]] = []  # (timestamp, score)

        # ── Thresholds ─────────────────────────────────────────
        #    Enter thresholds (going UP)
        self.thresholds_up = {
            ThreatState.MONITOR:   0.30,
            ThreatState.ALERT:     0.60,
            ThreatState.EMERGENCY: 0.80,
        }
        #    Exit thresholds (going DOWN) — lower by hysteresis_gap
        self.thresholds_down = {
            ThreatState.MONITOR:   0.30 - hysteresis_gap,  # 0.20
            ThreatState.ALERT:     0.60 - hysteresis_gap,  # 0.50
            ThreatState.EMERGENCY: 0.80 - hysteresis_gap,  # 0.70
        }

    def update(self, raw_score: float) -> ThreatState:
        """
        Feed a new raw threat probability from the classifier.

        Parameters
        ----------
        raw_score : float
            Threat probability from classifier (0.0 = safe, 1.0 = emergency)

        Returns
        -------
        ThreatState
            The current threat state after this update.
        """
        now = time.time()
        self.last_raw_score = raw_score

        # ── Step 1: Apply time decay since last update ─────────
        # If time has passed since last input, the threat level
        # should have been decaying continuously
        elapsed = now - self.last_update_time
        if elapsed > 0:
            # decay^elapsed gives the remaining fraction
            # e.g., 0.95^5 ≈ 0.77 → 77% of threat remains after 5 seconds
            self.threat_level *= self.decay_rate ** elapsed

        # ── Step 2: Check for instant panic override ───────────
        # A very high single score skips the gradual accumulation
        if raw_score >= self.panic_threshold:
            self.threat_level = max(self.threat_level, 0.90)

        # ── Step 3: Apply EMA (Exponential Moving Average) ─────
        # new_level = α × raw_score + (1-α) × previous_level
        # This smooths out individual scores — prevents one bad
        # score from causing a false alarm, but multiple bad
        # scores will push the level up
        self.threat_level = (
            self.alpha * raw_score +
            (1 - self.alpha) * self.threat_level
        )

        # ── Step 4: Clamp to [0, 1] range ─────────────────────
        self.threat_level = max(0.0, min(1.0, self.threat_level))

        # ── Step 5: Determine new state with hysteresis ────────
        old_state = self.current_state
        self.current_state = self._determine_state()

        # ── Step 6: Notify on state change ─────────────────────
        if self.current_state != old_state and self.on_state_change:
            self.on_state_change(old_state, self.current_state, self.threat_level)

        # ── Update tracking ────────────────────────────────────
        self.last_update_time = now
        self.score_history.append((now, self.threat_level))
        # Keep only last 100 entries
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-100:]

        return self.current_state

    def apply_decay(self):
        """
        Apply time-based decay without a new score.
        Call this periodically during silence to let threat drop.
        """
        now = time.time()
        elapsed = now - self.last_update_time
        if elapsed > 0:
            self.threat_level *= self.decay_rate ** elapsed
            self.threat_level = max(0.0, self.threat_level)
            old_state = self.current_state
            self.current_state = self._determine_state()
            if self.current_state != old_state and self.on_state_change:
                self.on_state_change(old_state, self.current_state, self.threat_level)
            self.last_update_time = now

    def _determine_state(self) -> ThreatState:
        """
        Determine threat state using hysteresis.

        When ESCALATING (going up): use the higher thresholds
        When DE-ESCALATING (going down): use the lower thresholds

        This prevents rapid flickering at boundary values.
        Example: If ALERT threshold is 0.60 and hysteresis gap is 0.10,
        you enter ALERT at 0.60 but only exit back to MONITOR at 0.50.
        """
        level = self.threat_level
        state = self.current_state

        # Check for escalation (use UP thresholds)
        if level >= self.thresholds_up[ThreatState.EMERGENCY]:
            return ThreatState.EMERGENCY
        elif level >= self.thresholds_up[ThreatState.ALERT]:
            if state.value in ("SAFE", "MONITOR"):
                return ThreatState.ALERT
            return state  # stay at current if already higher
        elif level >= self.thresholds_up[ThreatState.MONITOR]:
            if state == ThreatState.SAFE:
                return ThreatState.MONITOR
            # If currently ALERT/EMERGENCY, check de-escalation
            if state == ThreatState.ALERT and level < self.thresholds_down[ThreatState.ALERT]:
                return ThreatState.MONITOR
            if state == ThreatState.EMERGENCY and level < self.thresholds_down[ThreatState.EMERGENCY]:
                return ThreatState.ALERT
            return state

        # Below MONITOR threshold
        if state == ThreatState.MONITOR and level < self.thresholds_down[ThreatState.MONITOR]:
            return ThreatState.SAFE
        if state == ThreatState.ALERT and level < self.thresholds_down[ThreatState.ALERT]:
            return ThreatState.MONITOR
        if state == ThreatState.EMERGENCY and level < self.thresholds_down[ThreatState.EMERGENCY]:
            return ThreatState.ALERT

        if level < self.thresholds_down[ThreatState.MONITOR]:
            return ThreatState.SAFE

        return state

    def get_meter_bar(self, width: int = 30) -> str:
        """
        Render a visual threat meter bar for terminal display.

        Returns something like:
          ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░ 0.37 [MONITOR]
        """
        filled = int(self.threat_level * width)
        empty = width - filled
        color = STATE_COLORS.get(self.current_state, "")
        bar = f"{color}{'▓' * filled}{'░' * empty}{RESET}"
        return f"{bar} {self.threat_level:.2f} [{self.current_state.value}]"

    def get_debug_info(self) -> str:
        """Return a formatted debug string showing all scorer state."""
        color = STATE_COLORS.get(self.current_state, "")
        lines = [
            f"  ┌─ THREAT SCORER ─────────────────────────────────┐",
            f"  │  Meter:     {self.get_meter_bar(25):50s}│",
            f"  │  Raw score: {self.last_raw_score:<8.3f}  "
            f"Threat level: {self.threat_level:<8.3f}      │",
            f"  │  State:     {color}{self.current_state.value:10s}{RESET}"
            f"                               │",
            f"  └──────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """Demo: simulate a sequence of threat scores to see the meter."""

    def on_change(old, new, level):
        print(f"\n  ⚡ STATE CHANGE: {old.value} → {new.value} (level={level:.3f})\n")

    scorer = ThreatScorer(on_state_change=on_change)

    # Simulate an escalating situation
    test_scores = [
        (0.1, "just chatting normally"),
        (0.2, "still normal"),
        (0.15, "quiet conversation"),
        (0.4, "slightly uncomfortable words"),
        (0.5, "getting more concerning"),
        (0.7, "threatening language detected"),
        (0.8, "clear distress signals"),
        (0.9, "screaming for help"),
        (0.95, "extreme distress — PANIC"),
        (0.3, "situation calming"),
        (0.1, "back to normal"),
        (0.05, "quiet"),
    ]

    print("\n  Simulating threat score sequence:\n")
    for score, description in test_scores:
        state = scorer.update(score)
        print(f"  Input: {score:.2f} ({description})")
        print(scorer.get_debug_info())
        print()
        time.sleep(0.3)  # small delay to show decay effect
