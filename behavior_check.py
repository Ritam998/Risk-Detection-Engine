

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_layer import NLPResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MessageRecord:
    """Snapshot of one message stored in the sliding window."""
    raw_score: float
    keywords: list[str]
    sentiment: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BehaviorWindow:
   
    user_id: str
    records: deque[MessageRecord] = field(default_factory=deque)


@dataclass
class BehaviorReport:
   
    multiplier: float
    flags: list[str]


# ---------------------------------------------------------------------------
# Thresholds (centralised so they're easy to tune)
# ---------------------------------------------------------------------------

@dataclass
class BehaviorThresholds:
    escalation_slope: float = 3.0        # points/message to flag escalation
    escalation_window: int = 4           # how many consecutive messages to check
    keyword_repeat_count: int = 3        # same keyword N times → flag
    velocity_seconds: float = 30.0       # N messages within this window = velocity flag
    velocity_count: int = 5              # number of messages for velocity flag
    sentiment_swing_threshold: float = 1.2  # abs delta between consecutive sentiments
    sentiment_swing_count: int = 3       # how many swings to flag


# ---------------------------------------------------------------------------
# Behavior Checker
# ---------------------------------------------------------------------------

class BehaviorChecker:
    """
    Analyses a BehaviorWindow to produce a BehaviorReport.

    Each detected pattern contributes a multiplier increment.
    Multipliers are additive (not multiplicative) to avoid runaway inflation.

    Pattern → Multiplier Contribution
    -----------------------------------
    Score escalation          +0.30
    Keyword repetition        +0.25
    Rapid-fire velocity       +0.20
    Sentiment oscillation     +0.15
    """

    # Multiplier increments per detected pattern
    _ESCALATION_BOOST = 0.30
    _KEYWORD_BOOST = 0.25
    _VELOCITY_BOOST = 0.20
    _SENTIMENT_BOOST = 0.15

    def __init__(
        self,
        window_size: int = 10,
        thresholds: BehaviorThresholds | None = None,
    ) -> None:
        self._window_size = window_size
        self._thresholds = thresholds or BehaviorThresholds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        window: BehaviorWindow,
        nlp_result: "NLPResult",
    ) -> BehaviorReport:
        """
        Append the new message to the window, then run all pattern checks.

        Parameters
        ----------
        window     : mutable BehaviorWindow for this user (mutated in place)
        nlp_result : current message's NLP data

        Returns
        -------
        BehaviorReport with multiplier and flags list
        """
        # 1. Add current message to window
        record = MessageRecord(
            raw_score=nlp_result.raw_score,
            keywords=list(nlp_result.keywords),
            sentiment=nlp_result.sentiment,
        )
        window.records.append(record)
        # Enforce window size
        while len(window.records) > self._window_size:
            window.records.popleft()

        # 2. Run pattern checks
        multiplier = 1.0
        flags: list[str] = []

        check_result = self._check_escalation(window)
        if check_result:
            multiplier += self._ESCALATION_BOOST
            flags.append(check_result)

        check_result = self._check_keyword_repetition(window)
        if check_result:
            multiplier += self._KEYWORD_BOOST
            flags.append(check_result)

        check_result = self._check_velocity(window)
        if check_result:
            multiplier += self._VELOCITY_BOOST
            flags.append(check_result)

        check_result = self._check_sentiment_oscillation(window)
        if check_result:
            multiplier += self._SENTIMENT_BOOST
            flags.append(check_result)

        return BehaviorReport(multiplier=round(multiplier, 4), flags=flags)

    # ------------------------------------------------------------------
    # Pattern detection methods
    # ------------------------------------------------------------------

    def _check_escalation(self, window: BehaviorWindow) -> str | None:
        """
        Detect a rising score trend over the last N messages.
        Uses simple linear regression slope to avoid false positives
        from a single spike.
        """
        t = self._thresholds
        records = list(window.records)[-t.escalation_window:]
        if len(records) < t.escalation_window:
            return None

        scores = [r.raw_score for r in records]
        slope = _linear_slope(scores)

        if slope >= t.escalation_slope:
            return (
                f"ESCALATION: score rising {slope:.1f} pts/msg "
                f"over last {t.escalation_window} messages"
            )
        return None

    def _check_keyword_repetition(self, window: BehaviorWindow) -> str | None:
        """
        Count how many times each keyword appears across all window records.
        Flag if any keyword appears ≥ keyword_repeat_count times.
        """
        t = self._thresholds
        keyword_counts: dict[str, int] = {}
        for record in window.records:
            for kw in record.keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        repeated = [
            kw for kw, cnt in keyword_counts.items()
            if cnt >= t.keyword_repeat_count
        ]
        if repeated:
            top = repeated[:3]  # cap to avoid verbose logs
            return (
                f"KEYWORD_REPEAT: [{', '.join(top)}] each appeared "
                f"≥{t.keyword_repeat_count}x in last {len(window.records)} messages"
            )
        return None

    def _check_velocity(self, window: BehaviorWindow) -> str | None:
        """
        Detect message bursts: N messages within velocity_seconds.
        """
        t = self._thresholds
        records = list(window.records)
        if len(records) < t.velocity_count:
            return None

        recent = records[-t.velocity_count:]
        time_span = recent[-1].timestamp - recent[0].timestamp

        if 0 < time_span <= t.velocity_seconds:
            rate = t.velocity_count / time_span
            return (
                f"VELOCITY: {t.velocity_count} messages in {time_span:.1f}s "
                f"({rate:.1f} msg/s)"
            )
        return None

    def _check_sentiment_oscillation(self, window: BehaviorWindow) -> str | None:
        """
        Detect erratic sentiment swings (large alternating deltas).
        Counts consecutive pairs with |delta| > threshold.
        """
        t = self._thresholds
        records = list(window.records)
        if len(records) < 2:
            return None

        sentiments = [r.sentiment for r in records]
        swings = 0
        for i in range(1, len(sentiments)):
            delta = abs(sentiments[i] - sentiments[i - 1])
            if delta >= t.sentiment_swing_threshold:
                swings += 1

        if swings >= t.sentiment_swing_count:
            return (
                f"SENTIMENT_OSCILLATION: {swings} large sentiment swings "
                f"detected (threshold ±{t.sentiment_swing_threshold})"
            )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_slope(values: list[float]) -> float:
    """
    Compute the slope of the best-fit line through a list of values.
    x = [0, 1, 2, ..., n-1], y = values.
    Returns slope in units/step.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0
