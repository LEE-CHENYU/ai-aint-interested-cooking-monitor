"""
Event detection module.

Classifies visual events from camera frames.
Uses either a multimodal VLM (Gemma 3n) or a lightweight classifier.

Each zone has its own set of events to detect (defined in zones.yaml).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    zone: str
    timestamp: float
    signals: dict[str, any]  # e.g. {"water_state": "boiling", "pot_present": True}
    confidence: dict[str, float]  # e.g. {"water_state": 0.84}


class EventDetector:
    """Detects kitchen events from camera frames."""

    def __init__(self, zone: str, events: list[str]):
        self.zone = zone
        self.events = events
        self.model = None  # Loaded lazily

    def load_model(self):
        """Load the detection model (VLM or classifier)."""
        # TODO: Implement model loading
        # Option A: Gemma 3n multimodal - classify into label set
        # Option B: Transfer-learned MobileNet for single event
        pass

    def detect(self, frame: np.ndarray, timestamp: float) -> DetectionResult:
        """Run detection on a single frame."""
        # TODO: Implement detection
        # 1. Preprocess frame (resize, normalize)
        # 2. Run through model
        # 3. Parse output into structured signals

        # Placeholder return
        return DetectionResult(
            zone=self.zone,
            timestamp=timestamp,
            signals={event: None for event in self.events},
            confidence={event: 0.0 for event in self.events},
        )
