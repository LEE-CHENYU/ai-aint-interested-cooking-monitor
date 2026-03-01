"""
Camera ingest module.

Captures frames from camera feeds (or pre-recorded video) at configured FPS.
Each camera maps to a zone (stove, counter) defined in configs/zones.yaml.
"""

from dataclasses import dataclass

import cv2
import yaml


@dataclass
class CameraConfig:
    zone: str
    source: int | str  # Device index or video file path
    fps: int
    angle: str  # top_view or front_view


def load_zone_configs(config_path: str = "configs/zones.yaml") -> list[CameraConfig]:
    """Load camera zone configurations."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cameras = []
    for zone_name, zone_cfg in config["zones"].items():
        cameras.append(
            CameraConfig(
                zone=zone_name,
                source=0,  # Default to first camera; override per deployment
                fps=zone_cfg["fps"],
                angle=zone_cfg["camera_angle"],
            )
        )
    return cameras


class CameraStream:
    """Manages a single camera feed."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.config.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera for zone '{self.config.zone}': {self.config.source}"
            )

    def read_frame(self):
        """Read a single frame. Returns (success, frame)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self):
        if self.cap:
            self.cap.release()
