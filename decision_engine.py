

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW      = "LOW"       # 0  – 34
    MEDIUM   = "MEDIUM"    # 35 – 54
    HIGH     = "HIGH"      # 55 – 74
    CRITICAL = "CRITICAL"  # 75 – 100

    @staticmethod
    def from_score(score: float) -> "RiskLevel":
        if score < 35:
            return RiskLevel.LOW
        if score < 55:
            return RiskLevel.MEDIUM
        if score < 75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Alert threshold configuration
# ---------------------------------------------------------------------------

@dataclass
class DecisionConfig:
    """
    Controls when the Decision Engine fires an alert.

    alert_threshold   : final_score must exceed this to trigger alert
    min_raw_score     : raw_score must also exceed this (prevents alert on
                        low-confidence model output pushing the composite high)
    min_confidence    : confidence below this floors the final score regardless
    deviation_factor  : additional weight for high deviation (amplifies outliers)
    """
    alert_threshold: float  = 60.0
    min_raw_score: float    = 40.0
    min_confidence: float   = 0.30
    deviation_factor: float = 1.0      # set >1.0 to make deviation more decisive


# ---------------------------------------------------------------------------
# I/O contracts
# ---------------------------------------------------------------------------

@dataclass
class DecisionInput:
    """All signals needed for a final decision. Assembled by SessionLayer."""
    user_id: str
    raw_score: float        # 0–100 from NLP
    ema_score: float        # current EMA after incorporating raw_score
    confidence: float       # 0–1 model confidence from NLP
    behavior_multiplier: float  # 1.0+ from BehaviorChecker
    sentiment: float        # -1 (negative) to +1 (positive)


@dataclass
class DecisionResult:
    """Full decision output. Consumed by SessionLayer to populate SessionResult."""
    alert: bool
    risk_level: str
    final_score: float
    deviation: float
    reason: str


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Stateless final arbiter. Given a DecisionInput, produces a DecisionResult.

    Stateless design means it can be called concurrently without locks,
    and is trivially unit-testable.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self._cfg = config or DecisionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, inp: DecisionInput) -> DecisionResult:
        """
        Compute composite score, classify risk level, and fire alert if warranted.

        Parameters
        ----------
        inp : DecisionInput with all signals from upstream layers

        Returns
        -------
        DecisionResult with alert flag, risk level, score, and audit reason
        """
        cfg = self._cfg

        # ── 1. Deviation ───────────────────────────────────────────────
        # How far above the user's personal EMA baseline is the current score?
        deviation = max(0.0, inp.raw_score - inp.ema_score)

        # ── 2. Sentiment adjustment ────────────────────────────────────
        # Negative sentiment → high score contribution (0–100 range)
        # sentiment=-1 → sentiment_adj=100; sentiment=+1 → sentiment_adj=0
        sentiment_adj = (1.0 - inp.sentiment) * 50.0

        # ── 3. Weighted composite ──────────────────────────────────────
        weighted = (
            inp.raw_score   * 0.35
            + inp.ema_score * 0.30
            + deviation     * 0.25 * cfg.deviation_factor
            + sentiment_adj * 0.10
        )

        # ── 4. Confidence weight ───────────────────────────────────────
        # Low-confidence model outputs reduce the final score.
        confidence_weight = 0.7 + 0.3 * inp.confidence

        # ── 5. Behavior multiplier ──────────────────────────────────────
        final_score = weighted * confidence_weight * inp.behavior_multiplier
        final_score = round(_clamp(final_score, 0.0, 100.0), 4)

        # ── 6. Risk level classification ───────────────────────────────
        risk_level = RiskLevel.from_score(final_score)

        # ── 7. Alert decision ──────────────────────────────────────────
        alert, reason = self._evaluate_alert(inp, final_score, deviation, risk_level)

        return DecisionResult(
            alert=alert,
            risk_level=risk_level.value,
            final_score=final_score,
            deviation=round(deviation, 4),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Alert logic (separated for clarity and testability)
    # ------------------------------------------------------------------

    def _evaluate_alert(
        self,
        inp: DecisionInput,
        final_score: float,
        deviation: float,
        risk_level: RiskLevel,
    ) -> tuple[bool, str]:
        """
        Apply multi-signal gate to decide whether to fire an alert.

        Returns
        -------
        (alert: bool, reason: str)
        """
        cfg = self._cfg

        # Gate 1: Confidence too low to act
        if inp.confidence < cfg.min_confidence:
            return False, (
                f"NO_ALERT: confidence {inp.confidence:.2f} below minimum "
                f"{cfg.min_confidence:.2f}; score suppressed"
            )

        # Gate 2: Raw score below floor (model uncertainty)
        if inp.raw_score < cfg.min_raw_score:
            return False, (
                f"NO_ALERT: raw_score {inp.raw_score:.1f} below floor "
                f"{cfg.min_raw_score:.1f}"
            )

        # Gate 3: Composite threshold
        if final_score < cfg.alert_threshold:
            return False, (
                f"NO_ALERT: final_score {final_score:.1f} < threshold "
                f"{cfg.alert_threshold:.1f}; risk={risk_level.value}"
            )

        # Alert fired
        trigger_reasons = []
        if final_score >= cfg.alert_threshold:
            trigger_reasons.append(f"score={final_score:.1f}")
        if deviation > 20:
            trigger_reasons.append(f"deviation={deviation:.1f}")
        if inp.behavior_multiplier > 1.0:
            trigger_reasons.append(f"behavior×{inp.behavior_multiplier:.2f}")

        return True, (
            f"ALERT [{risk_level.value}]: "
            + " | ".join(trigger_reasons)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
