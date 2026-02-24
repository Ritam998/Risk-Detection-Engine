

from __future__ import annotations

import time
import pytest

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.ema_layer import EMALayer, EMAConfig
from services.behavior_check import (
    BehaviorChecker,
    BehaviorThresholds,
    BehaviorWindow,
)
from services.decision_engine import (
    DecisionEngine,
    DecisionConfig,
    DecisionInput,
    RiskLevel,
)
from services.session_layer import (
    SessionLayer,
    NLPResult,
    UserBaseline,
    SessionResult,
)



def _nlp(
    user_id: str = "u1",
    raw_score: float = 50.0,
    confidence: float = 0.9,
    keywords: list[str] | None = None,
    sentiment: float = 0.0,
    message_text: str = "test message",
) -> NLPResult:
    return NLPResult(
        raw_score=raw_score,
        confidence=confidence,
        keywords=keywords or [],
        sentiment=sentiment,
        message_text=message_text,
        user_id=user_id,
    )


def _baseline(
    user_id: str = "u1",
    ema_value: float = 50.0,
    message_count: int = 20,
    known_keywords: list[str] | None = None,
) -> UserBaseline:
    return UserBaseline(
        user_id=user_id,
        ema_value=ema_value,
        message_count=message_count,
        known_keywords=known_keywords or [],
        last_updated=time.time(),
    )


class TestEMALayer:

    def test_new_user_gets_initial_ema(self):
        """Brand-new user with 0 messages should start from config.initial_ema."""
        cfg = EMAConfig(initial_ema=50.0, alpha_cold_start=0.3)
        layer = EMALayer(config=cfg)
        result = layer.update("u_new", new_score=80.0, stored_ema=0.0, message_count=0)
        # EMA = 0.3 * 80 + 0.7 * 50 = 24 + 35 = 59
        assert abs(result - 59.0) < 0.01

    def test_existing_user_seeds_from_db(self):
        """Existing user (message_count > 0) should seed EMA from stored value."""
        cfg = EMAConfig(alpha_stable=0.1, cold_start_msgs=5)
        layer = EMALayer(config=cfg)
        result = layer.update("u1", new_score=60.0, stored_ema=40.0, message_count=10)
        # Using stable alpha because count >= cold_start
        # EMA = 0.1 * 60 + 0.9 * 40 = 6 + 36 = 42
        assert abs(result - 42.0) < 0.01

    def test_cold_start_uses_fast_alpha(self):
        cfg = EMAConfig(alpha_cold_start=0.5, alpha_stable=0.1, cold_start_msgs=5, initial_ema=50.0)
        layer = EMALayer(config=cfg)
        result = layer.update("u1", new_score=100.0, stored_ema=0.0, message_count=0)
        # cold start: EMA = 0.5 * 100 + 0.5 * 50 = 75
        assert abs(result - 75.0) < 0.01

    def test_stable_phase_uses_slow_alpha(self):
        cfg = EMAConfig(alpha_cold_start=0.5, alpha_stable=0.1, cold_start_msgs=3, initial_ema=50.0)
        layer = EMALayer(config=cfg)
        result = layer.update("u1", new_score=100.0, stored_ema=50.0, message_count=5)
        # stable: EMA = 0.1 * 100 + 0.9 * 50 = 10 + 45 = 55
        assert abs(result - 55.0) < 0.01

    def test_ema_clamps_to_range(self):
        cfg = EMAConfig(clamp_min=0.0, clamp_max=100.0, initial_ema=90.0, alpha_cold_start=0.5)
        layer = EMALayer(config=cfg)
        result = layer.update("u1", new_score=150.0, stored_ema=0.0, message_count=0)
        assert result <= 100.0

    def test_second_message_uses_in_memory_state(self):
        """Second call should use in-memory EMA, not the DB stored_ema."""
        layer = EMALayer(EMAConfig(alpha_stable=0.1, cold_start_msgs=1, initial_ema=50.0))
        first = layer.update("u1", new_score=60.0, stored_ema=50.0, message_count=5)
        # Pass different stored_ema — should be ignored
        second = layer.update("u1", new_score=70.0, stored_ema=999.0, message_count=5)
        expected_second = round(0.1 * 70 + 0.9 * first, 4)
        assert abs(second - expected_second) < 0.01

    def test_deviation_positive(self):
        layer = EMALayer(EMAConfig(alpha_stable=0.1, cold_start_msgs=1, initial_ema=30.0))
        layer.update("u1", new_score=30.0, stored_ema=30.0, message_count=10)
        dev = layer.compute_deviation("u1", raw_score=80.0)
        assert dev > 0

    def test_deviation_floored_at_zero(self):
        layer = EMALayer(EMAConfig(alpha_stable=0.1, cold_start_msgs=1, initial_ema=80.0))
        layer.update("u1", new_score=80.0, stored_ema=80.0, message_count=10)
        dev = layer.compute_deviation("u1", raw_score=10.0)
        assert dev == 0.0

    def test_clear_user(self):
        layer = EMALayer()
        layer.update("u1", 50.0, 50.0, 10)
        assert layer.get_current_ema("u1") is not None
        layer.clear_user("u1")
        assert layer.get_current_ema("u1") is None



class TestBehaviorChecker:

    def _window(self, uid="u1") -> BehaviorWindow:
        return BehaviorWindow(user_id=uid)

    def test_no_flags_on_first_message(self):
        checker = BehaviorChecker(window_size=10)
        window = self._window()
        report = checker.evaluate(window, _nlp(raw_score=30.0))
        assert report.multiplier == 1.0
        assert report.flags == []

    def test_escalation_detected(self):
        thresholds = BehaviorThresholds(escalation_slope=2.0, escalation_window=4)
        checker = BehaviorChecker(window_size=10, thresholds=thresholds)
        window = self._window()
        # Send rising scores: 30, 40, 50, 60 — slope = 10/msg
        for score in [30.0, 40.0, 50.0, 60.0]:
            report = checker.evaluate(window, _nlp(raw_score=score))
        assert any("ESCALATION" in f for f in report.flags)
        assert report.multiplier > 1.0

    def test_no_escalation_on_flat_scores(self):
        checker = BehaviorChecker(window_size=10)
        window = self._window()
        for score in [40.0, 41.0, 39.0, 40.5]:
            report = checker.evaluate(window, _nlp(raw_score=score))
        assert not any("ESCALATION" in f for f in report.flags)

    def test_keyword_repetition_detected(self):
        thresholds = BehaviorThresholds(keyword_repeat_count=3)
        checker = BehaviorChecker(window_size=10, thresholds=thresholds)
        window = self._window()
        for _ in range(3):
            report = checker.evaluate(window, _nlp(keywords=["threat", "weapon"]))
        assert any("KEYWORD_REPEAT" in f for f in report.flags)

    def test_velocity_detected(self, monkeypatch):
        """Simulate rapid-fire messages by mocking time."""
        thresholds = BehaviorThresholds(velocity_count=5, velocity_seconds=30.0)
        checker = BehaviorChecker(window_size=10, thresholds=thresholds)
        window = self._window()

        fake_time = [1000.0]

        import services.behavior_check as bc_mod
        monkeypatch.setattr(bc_mod.time, "time", lambda: fake_time[0])

        for i in range(5):
            fake_time[0] = 1000.0 + i * 5  # 5s apart = 25s total < 30s threshold
            report = checker.evaluate(window, _nlp())

        assert any("VELOCITY" in f for f in report.flags)

    def test_sentiment_oscillation_detected(self):
        thresholds = BehaviorThresholds(
            sentiment_swing_threshold=1.0, sentiment_swing_count=3
        )
        checker = BehaviorChecker(window_size=10, thresholds=thresholds)
        window = self._window()
        sentiments = [-1.0, 1.0, -1.0, 1.0, -1.0]  # extreme swings
        for s in sentiments:
            report = checker.evaluate(window, _nlp(sentiment=s))
        assert any("SENTIMENT_OSCILLATION" in f for f in report.flags)

    def test_window_size_enforced(self):
        checker = BehaviorChecker(window_size=3)
        window = self._window()
        for score in [10, 20, 30, 40, 50]:
            checker.evaluate(window, _nlp(raw_score=float(score)))
        assert len(window.records) <= 3

    def test_multiple_flags_accumulate_multiplier(self):
        """Multiple patterns should increase multiplier additively."""
        thresholds = BehaviorThresholds(
            escalation_slope=2.0,
            escalation_window=4,
            keyword_repeat_count=2,
        )
        checker = BehaviorChecker(window_size=10, thresholds=thresholds)
        window = self._window()
        for score in [30.0, 45.0, 60.0, 75.0]:
            report = checker.evaluate(
                window, _nlp(raw_score=score, keywords=["danger"])
            )
        assert report.multiplier > 1.3  # escalation + keyword


# ============================================================================
# DecisionEngine Tests
# ============================================================================

class TestDecisionEngine:

    def _inp(
        self,
        raw_score=50.0,
        ema_score=50.0,
        confidence=0.9,
        behavior_multiplier=1.0,
        sentiment=0.0,
    ) -> DecisionInput:
        return DecisionInput(
            user_id="u1",
            raw_score=raw_score,
            ema_score=ema_score,
            confidence=confidence,
            behavior_multiplier=behavior_multiplier,
            sentiment=sentiment,
        )

    def test_low_confidence_suppresses_alert(self):
        cfg = DecisionConfig(min_confidence=0.5)
        engine = DecisionEngine(config=cfg)
        result = engine.decide(self._inp(raw_score=90.0, confidence=0.2))
        assert result.alert is False
        assert "confidence" in result.reason.lower()

    def test_low_raw_score_suppresses_alert(self):
        cfg = DecisionConfig(min_raw_score=40.0, alert_threshold=30.0)
        engine = DecisionEngine(config=cfg)
        result = engine.decide(self._inp(raw_score=20.0, confidence=0.95))
        assert result.alert is False

    def test_high_score_triggers_alert(self):
        cfg = DecisionConfig(alert_threshold=55.0, min_raw_score=40.0, min_confidence=0.3)
        engine = DecisionEngine(config=cfg)
        result = engine.decide(
            self._inp(raw_score=90.0, ema_score=85.0, confidence=0.95, sentiment=-0.8)
        )
        assert result.alert is True

    def test_risk_level_low(self):
        assert RiskLevel.from_score(10.0) == RiskLevel.LOW

    def test_risk_level_medium(self):
        assert RiskLevel.from_score(45.0) == RiskLevel.MEDIUM

    def test_risk_level_high(self):
        assert RiskLevel.from_score(65.0) == RiskLevel.HIGH

    def test_risk_level_critical(self):
        assert RiskLevel.from_score(80.0) == RiskLevel.CRITICAL

    def test_behavior_multiplier_inflates_score(self):
        cfg = DecisionConfig(alert_threshold=60.0, min_raw_score=0.0, min_confidence=0.0)
        engine = DecisionEngine(config=cfg)
        no_behavior = engine.decide(self._inp(raw_score=50.0, ema_score=50.0, behavior_multiplier=1.0))
        with_behavior = engine.decide(self._inp(raw_score=50.0, ema_score=50.0, behavior_multiplier=1.5))
        assert with_behavior.final_score > no_behavior.final_score

    def test_negative_sentiment_raises_score(self):
        cfg = DecisionConfig(min_raw_score=0.0, min_confidence=0.0)
        engine = DecisionEngine(config=cfg)
        neutral = engine.decide(self._inp(sentiment=0.0))
        negative = engine.decide(self._inp(sentiment=-1.0))
        assert negative.final_score > neutral.final_score

    def test_final_score_clamped_to_100(self):
        cfg = DecisionConfig(min_raw_score=0.0, min_confidence=0.0)
        engine = DecisionEngine(config=cfg)
        result = engine.decide(
            self._inp(raw_score=100.0, ema_score=100.0, behavior_multiplier=2.0, sentiment=-1.0)
        )
        assert result.final_score <= 100.0


# ============================================================================
# SessionLayer Integration Tests
# ============================================================================

class TestSessionLayer:

    def test_returns_session_result(self):
        engine = SessionLayer()
        result = engine.process(_nlp(), _baseline())
        assert isinstance(result, SessionResult)
        assert result.user_id == "u1"
        assert 0.0 <= result.final_score <= 100.0
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result.session_id.startswith("rsn-")

    def test_alert_fires_on_high_risk_message(self):
        engine = SessionLayer()
        # User with very low baseline suddenly sends a high-risk message
        baseline = _baseline(ema_value=5.0, message_count=50)
        nlp = _nlp(raw_score=95.0, confidence=0.95, sentiment=-1.0)
        result = engine.process(nlp, baseline)
        assert result.alert is True

    def test_no_alert_on_low_risk(self):
        cfg_decision = DecisionConfig(alert_threshold=70.0)
        engine = SessionLayer()
        baseline = _baseline(ema_value=50.0, message_count=30)
        nlp = _nlp(raw_score=20.0, confidence=0.9, sentiment=0.5)
        result = engine.process(nlp, baseline)
        assert result.alert is False

    def test_processing_time_recorded(self):
        engine = SessionLayer()
        result = engine.process(_nlp(), _baseline())
        assert result.processing_time_ms > 0

    def test_session_id_unique_per_call(self):
        engine = SessionLayer()
        r1 = engine.process(_nlp(), _baseline())
        r2 = engine.process(_nlp(), _baseline())
        assert r1.session_id != r2.session_id

    def test_reset_user_session(self):
        engine = SessionLayer()
        engine.process(_nlp(), _baseline())
        # Should not raise
        engine.reset_user_session("u1")

    def test_multiple_messages_same_user(self):
        """EMA should adapt across messages for the same user in one session."""
        engine = SessionLayer()
        baseline = _baseline(ema_value=10.0, message_count=5)
        scores = []
        for score in [20.0, 30.0, 40.0, 50.0, 60.0]:
            result = engine.process(_nlp(raw_score=score), baseline)
            scores.append(result.ema_score)
        # EMA should be rising but smoothed
        assert scores[-1] > scores[0]
        assert scores[-1] < 60.0  # smoothing means it doesn't fully reach 60

    def test_behavior_flags_propagated(self):
        """Escalating messages should produce behavior flags in result."""
        engine = SessionLayer()
        baseline = _baseline(ema_value=5.0, message_count=20)
        results = []
        for score in [30.0, 45.0, 60.0, 75.0]:
            result = engine.process(_nlp(raw_score=score), baseline)
            results.append(result)
        # The last result should have escalation flag
        last = results[-1]
        assert any("ESCALATION" in f for f in last.behavior_flags)

    def test_different_users_isolated(self):
        """Two users processed by same engine should not share state."""
        engine = SessionLayer()
        b1 = _baseline("user_A", ema_value=10.0, message_count=5)
        b2 = _baseline("user_B", ema_value=80.0, message_count=5)
        r1 = engine.process(_nlp("user_A", raw_score=50.0), b1)
        r2 = engine.process(_nlp("user_B", raw_score=50.0), b2)
        # user_A has low baseline, user_B has high — EMAs should differ
        assert r1.ema_score != r2.ema_score
