"""
World state aggregation.

Maintains the complete kitchen state by combining:
- Zone signals from perception (per-camera detections)
- Active timers from TimerEngine
- Recipe step tracking
- Safety rule status
- Alert cooldowns (anti-nag)
"""

import time
from dataclasses import dataclass, field

import yaml

from src.world_state.timer_engine import TimerEngine


@dataclass
class RecipeStep:
    id: int
    name: str
    description: str
    status: str = "pending"  # pending, in_progress, done
    duration_seconds: int | None = None


class WorldState:
    """Aggregates all kitchen state for the agent."""

    def __init__(self, recipe_path: str | None = None):
        self.timer_engine = TimerEngine()
        self.zone_signals: dict[str, dict] = {}  # zone_name -> latest signals
        self.steps: list[RecipeStep] = []
        self.last_alert_times: dict[str, float] = {}  # rule_id -> last alert timestamp

        if recipe_path:
            self._load_recipe(recipe_path)

    def _load_recipe(self, path: str):
        with open(path) as f:
            recipe = yaml.safe_load(f)
        self.steps = [
            RecipeStep(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                duration_seconds=s.get("duration_seconds"),
            )
            for s in recipe["recipe"]["steps"]
        ]

    def update_zone(self, zone: str, signals: dict, confidence: dict):
        """Update signals from a perception zone."""
        self.zone_signals[zone] = {
            "signals": signals,
            "confidence": confidence,
            "ts": time.time(),
        }

    def mark_step_done(self, step_id: int):
        for step in self.steps:
            if step.id == step_id:
                step.status = "done"
                break

    def reorder_steps(self, new_order: list[int]):
        step_map = {s.id: s for s in self.steps}
        self.steps = [step_map[sid] for sid in new_order if sid in step_map]

    def can_alert(self, rule_id: str, cooldown_seconds: float) -> bool:
        """Check if enough time has passed since last alert for this rule."""
        last = self.last_alert_times.get(rule_id, 0)
        return (time.time() - last) >= cooldown_seconds

    def record_alert(self, rule_id: str):
        self.last_alert_times[rule_id] = time.time()

    def get_snapshot(self) -> dict:
        """Build a compact state snapshot for the agent prompt."""
        return {
            "zones": self.zone_signals,
            "timers": self.timer_engine.to_dict(),
            "steps": [
                {"id": s.id, "name": s.name, "status": s.status} for s in self.steps
            ],
            "expired_timers": [
                t.to_dict() for t in self.timer_engine.get_expired()
            ],
        }
