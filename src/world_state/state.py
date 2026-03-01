"""
World state aggregation.

Maintains the kitchen state by combining:
- Zone signals from VLM perception
- Active timers from TimerEngine
- Alert cooldowns (anti-nag)

Simplified for single-threaded agentic loop — no threading locks needed.
Recipe step tracking is handled by StepEngine.
"""

import time

from src.agent.schemas import Recipe
from src.world_state.timer_engine import TimerEngine


class WorldState:
    """Aggregates all kitchen state for the agent."""

    def __init__(self, recipe: Recipe | None = None):
        self.timer_engine = TimerEngine()
        self.zone_signals: dict[str, dict] = {}  # zone_name -> latest signals
        self.last_alert_times: dict[str, float] = {}  # rule_id -> last alert timestamp
        self.recipe = recipe

    def update_zone(self, zone: str, signals: dict):
        """Update signals from a perception zone."""
        self.zone_signals[zone] = {
            "signals": signals,
            "ts": time.time(),
        }

    def can_alert(self, rule_id: str, cooldown_seconds: float) -> bool:
        """Check if enough time has passed since last alert for this rule."""
        last = self.last_alert_times.get(rule_id, 0)
        return (time.time() - last) >= cooldown_seconds

    def record_alert(self, rule_id: str):
        self.last_alert_times[rule_id] = time.time()

    def get_flat_signals(self) -> dict[str, object]:
        """
        Return a completely flat dict for safety rule evaluation.

        Keys are "zone.signal_name", e.g.:
            {"stove.pot_present": True, "counter.hands_active": False}
        """
        flat: dict[str, object] = {}
        for zone_name, zone_data in self.zone_signals.items():
            signals = zone_data.get("signals", {})
            for signal_name, value in signals.items():
                flat[f"{zone_name}.{signal_name}"] = value
        return flat

    def get_snapshot(self) -> dict:
        """Build a compact state snapshot (for optional WebSocket UI)."""
        return {
            "zones": {k: v for k, v in self.zone_signals.items()},
            "timers": self.timer_engine.to_dict(),
            "expired_timers": [
                t.to_dict() for t in self.timer_engine.get_expired()
            ],
        }
