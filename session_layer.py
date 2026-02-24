

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .ema_layer import EMALayer, EMAConfig
from .behavior_check import BehaviorChecker, BehaviorWindow
from .decision_engine import DecisionEngine, DecisionInput, DecisionResult


@dataclass
class NLPResult:
    
    raw_score: float          
    confidence: float         
    keywords: list[str]       
    sentiment: float         
    message_text: str        
    user_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserBaseline:
  
    user_id: str
    ema_value: float          
    message_count: int        
    known_keywords: list[str] 
    last_updated: float       


@dataclass
class SessionResult:
  
    session_id: str
    user_id: str
    alert: bool
    risk_level: str           
    final_score: float        
    ema_score: float        
    deviation: float          
    behavior_flags: list[str] 
    decision_reason: str     
    processing_time_ms: float
    timestamp: float = field(default_factory=time.time)



class SessionLayer:
    

    def __init__(
        self,
        ema_config: Optional[EMAConfig] = None,
        behavior_window_size: int = 10,
    ) -> None:
        self._ema_layer = EMALayer(config=ema_config or EMAConfig())
        self._behavior_checker = BehaviorChecker(window_size=behavior_window_size)
        self._decision_engine = DecisionEngine()

        # In-memory session store:  user_id → BehaviorWindow
        # The EMA state is managed inside EMALayer keyed by user_id.
        self._behavior_windows: dict[str, BehaviorWindow] = {}

   

    def process(
        self,
        nlp_result: NLPResult,
        user_baseline: UserBaseline,
    ) -> SessionResult:
        
        start_ts = time.perf_counter()
        session_id = _generate_session_id()
        uid = nlp_result.user_id

        # ── Step 1: EMA update ─────────────────────────────────────────
        ema_score = self._ema_layer.update(
            user_id=uid,
            new_score=nlp_result.raw_score,
            stored_ema=user_baseline.ema_value,
            message_count=user_baseline.message_count,
        )

        # ── Step 2: Behavior check ─────────────────────────────────────
        window = self._behavior_windows.setdefault(uid, BehaviorWindow(user_id=uid))
        behavior_report = self._behavior_checker.evaluate(
            window=window,
            nlp_result=nlp_result,
        )

        # ── Step 3: Decision engine ────────────────────────────────────
        decision_input = DecisionInput(
            user_id=uid,
            raw_score=nlp_result.raw_score,
            ema_score=ema_score,
            confidence=nlp_result.confidence,
            behavior_multiplier=behavior_report.multiplier,
            sentiment=nlp_result.sentiment,
        )
        decision: DecisionResult = self._decision_engine.decide(decision_input)

        elapsed_ms = (time.perf_counter() - start_ts) * 1000

        return SessionResult(
            session_id=session_id,
            user_id=uid,
            alert=decision.alert,
            risk_level=decision.risk_level,
            final_score=decision.final_score,
            ema_score=ema_score,
            deviation=decision.deviation,
            behavior_flags=behavior_report.flags,
            decision_reason=decision.reason,
            processing_time_ms=round(elapsed_ms, 3),
        )

    def reset_user_session(self, user_id: str) -> None:
        
        self._behavior_windows.pop(user_id, None)
        self._ema_layer.clear_user(user_id)



def _generate_session_id() -> str:
    return f"rsn-{uuid.uuid4().hex[:12]}"
