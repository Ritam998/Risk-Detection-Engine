

from __future__ import annotations

from dataclasses import dataclass



@dataclass
class EMAConfig:
   
    alpha_cold_start: float = 0.30   # react quickly in cold-start
    alpha_stable: float = 0.10       # smooth baseline after warm-up
    cold_start_msgs: int = 10        # messages before stable phase
    initial_ema: float = 50.0        # neutral baseline for new users
    clamp_min: float = 0.0
    clamp_max: float = 100.0


class EMALayer:
    """
    Maintains per-user EMA state in memory between messages within a session.

    The Database Layer is responsible for persisting `ema_value` and
    `message_count` between sessions. On the first call for a user, the
    stored values from the DB are provided; subsequent intra-session calls
    use in-memory state.

    Formula
    -------
        EMA_t = α × score_t + (1 − α) × EMA_{t-1}

    where α is dynamic based on the user's message history.
    """

    def __init__(self, config: EMAConfig | None = None) -> None:
        self._config = config or EMAConfig()

        # In-memory state per user:  user_id → (ema_value, message_count)
        self._state: dict[str, tuple[float, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        user_id: str,
        new_score: float,
        stored_ema: float,
        message_count: int,
    ) -> float:
    
        cfg = self._config

        if user_id not in self._state:
            # First message in this session — seed from database values
            current_ema = stored_ema if message_count > 0 else cfg.initial_ema
            current_count = message_count
        else:
            current_ema, current_count = self._state[user_id]

        # Choose alpha based on warm-up phase
        alpha = (
            cfg.alpha_cold_start
            if current_count < cfg.cold_start_msgs
            else cfg.alpha_stable
        )

        updated_ema = alpha * new_score + (1.0 - alpha) * current_ema
        updated_ema = _clamp(updated_ema, cfg.clamp_min, cfg.clamp_max)
        updated_count = current_count + 1

        self._state[user_id] = (updated_ema, updated_count)
        return round(updated_ema, 4)

    def get_current_ema(self, user_id: str) -> float | None:
        """Return the current in-memory EMA for a user, or None if unseen."""
        state = self._state.get(user_id)
        return state[0] if state else None

    def get_message_count(self, user_id: str) -> int:
        """Return in-memory message count for a user, or 0 if unseen."""
        state = self._state.get(user_id)
        return state[1] if state else 0

    def clear_user(self, user_id: str) -> None:
        """Remove a user's in-memory state (called by SessionLayer on reset)."""
        self._state.pop(user_id, None)

    # ------------------------------------------------------------------
    # Utility: deviation from baseline
    # ------------------------------------------------------------------

    def compute_deviation(self, user_id: str, raw_score: float) -> float:
        """
        Calculate how many points the raw_score deviates above the
        current EMA baseline. Negative deviations are floored to 0
        (we only care about upward deviations for alerting).

        Returns
        -------
        float : deviation score ≥ 0
        """
        current_ema = self.get_current_ema(user_id)
        if current_ema is None:
            return 0.0
        deviation = raw_score - current_ema
        return round(max(deviation, 0.0), 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
