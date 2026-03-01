"""
Temporal smoothing for detection results.

Requires N out of M recent frames to agree before reporting a state change.
This prevents flickering detections from causing false alerts.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class SmoothedSignal:
    value: any
    stable: bool  # True if the value has been consistent for the window
    agreement_ratio: float  # e.g., 5/7 = 0.71


class TemporalSmoother:
    """Smooths detection signals over a sliding window of frames."""

    def __init__(self, window_size: int = 3, threshold: float = 0.66):
        self.window_size = window_size
        self.threshold = threshold
        self.history: dict[str, deque] = {}

    def update(self, signal_name: str, value: any) -> SmoothedSignal:
        """Add a new observation and return the smoothed signal."""
        if signal_name not in self.history:
            self.history[signal_name] = deque(maxlen=self.window_size)

        self.history[signal_name].append(value)
        window = self.history[signal_name]

        if len(window) < 3:
            return SmoothedSignal(value=value, stable=False, agreement_ratio=0.0)

        # Count most common value in window
        from collections import Counter

        counts = Counter(window)
        most_common_value, most_common_count = counts.most_common(1)[0]
        ratio = most_common_count / len(window)

        return SmoothedSignal(
            value=most_common_value,
            stable=ratio >= self.threshold,
            agreement_ratio=ratio,
        )

    def reset(self, signal_name: str | None = None):
        """Reset history for a signal or all signals."""
        if signal_name:
            self.history.pop(signal_name, None)
        else:
            self.history.clear()
