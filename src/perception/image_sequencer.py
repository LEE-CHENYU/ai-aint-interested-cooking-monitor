"""
Image sequencer for demo mode.

Loads a directory of images (sorted by filename) and serves them
one at a time, simulating a camera feed from pre-captured frames.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageSequencer:
    """Iterates over a directory of images as if they were camera frames."""

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self._files: list[Path] = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ) if self.image_dir.exists() else []
        self._index = 0
        logger.info("ImageSequencer: %d images from %s", len(self._files), image_dir)

    def next(self) -> tuple[np.ndarray | None, str | None]:
        """Return the next (frame, filename) pair, or (None, None) if done."""
        if self._index >= len(self._files):
            return None, None
        path = self._files[self._index]
        self._index += 1
        img = Image.open(path).convert("RGB")
        frame = np.array(img)
        return frame, path.name

    @property
    def done(self) -> bool:
        return self._index >= len(self._files)

    def reset(self):
        self._index = 0

    def __len__(self) -> int:
        return len(self._files)
