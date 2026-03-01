"""
TTS engine — console output + macOS speech synthesis.

Always prints to console with priority-based prefixes.
Optionally speaks aloud using macOS `say` command (non-blocking).
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

# Console prefixes from the agentic workflow doc
_PREFIXES = {
    "low": "   ",
    "medium": ">> ",
    "high": "!! ",
    "critical": "** WARNING ** ",
}

# Speech rates (words per minute)
_SPEECH_RATES = {
    "low": 200,
    "medium": 200,
    "high": 200,
    "critical": 230,
}


class TTSEngine:
    """Console + optional macOS TTS output."""

    def __init__(self, audio: bool = True):
        self.audio = audio
        self._process: subprocess.Popen | None = None

    def speak(self, text: str, priority: str = "medium"):
        """Print to console and optionally speak aloud."""
        prefix = _PREFIXES.get(priority, ">> ")
        print(f"{prefix}{text}")

        if self.audio:
            # Kill any ongoing speech for higher-priority interrupts
            if self._process and self._process.poll() is None:
                if priority in ("high", "critical"):
                    self._process.terminate()
                else:
                    # Don't interrupt for low/medium
                    return

            rate = _SPEECH_RATES.get(priority, 200)
            try:
                self._process = subprocess.Popen(
                    ["say", "-r", str(rate), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                # `say` not available (non-macOS)
                logger.debug("TTS not available (say command not found)")
