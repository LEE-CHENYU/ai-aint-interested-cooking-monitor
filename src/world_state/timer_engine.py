"""
Timer management engine.

Creates, tracks, and expires timers for cooking steps.
Alerts the agent when timers are about to expire or have expired.
"""

import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    name: str
    duration_seconds: int
    start_time: float = field(default_factory=time.time)
    paused: bool = False

    @property
    def remaining_seconds(self) -> float:
        if self.paused:
            return self.duration_seconds
        elapsed = time.time() - self.start_time
        return max(0, self.duration_seconds - elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining_seconds <= 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "remaining_s": round(self.remaining_seconds),
            "expired": self.expired,
        }


class TimerEngine:
    """Manages all active cooking timers."""

    def __init__(self):
        self.timers: dict[str, Timer] = {}

    def create_timer(self, name: str, seconds: int) -> Timer:
        timer = Timer(name=name, duration_seconds=seconds)
        self.timers[name] = timer
        print(f"Timer created: {name} ({seconds}s)")
        return timer

    def adjust_timer(self, name: str, delta_seconds: int):
        if name in self.timers:
            self.timers[name].duration_seconds += delta_seconds
            print(f"Timer adjusted: {name} ({delta_seconds:+d}s)")

    def cancel_timer(self, name: str):
        self.timers.pop(name, None)

    def get_expired(self) -> list[Timer]:
        return [t for t in self.timers.values() if t.expired]

    def get_active(self) -> list[Timer]:
        return [t for t in self.timers.values() if not t.expired]

    def cleanup_expired(self):
        expired = [name for name, t in self.timers.items() if t.expired]
        for name in expired:
            del self.timers[name]

    def to_dict(self) -> list[dict]:
        return [t.to_dict() for t in self.timers.values()]
