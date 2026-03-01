"""
TTS engine — ElevenLabs cloud synthesis with pygame local playback.

Calls ElevenLabs text-to-speech API, plays audio locally via pygame,
and optionally pushes base64 MP3 over WebSocket to the phone UI.
Console logging with priority prefixes is always active.
"""

import base64
import io
import logging
import os
import threading
from pathlib import Path

from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

# Console prefixes from the agentic workflow doc
_PREFIXES = {
    "low": "   ",
    "medium": ">> ",
    "high": "!! ",
    "critical": "** WARNING ** ",
}

# Default voice — ElevenLabs "George" (warm, clear male)
_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"


def _init_pygame():
    """Lazy-init pygame.mixer (called once on first playback)."""
    import pygame

    if not pygame.mixer.get_init():
        pygame.mixer.init()


class TTSEngine:
    """ElevenLabs TTS with local playback and optional WebSocket delivery."""

    def __init__(
        self,
        audio: bool = True,
        voice_id: str | None = None,
        model_id: str = "eleven_multilingual_v2",
        ui_server=None,
    ):
        self.audio = audio
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", _DEFAULT_VOICE_ID)
        self.model_id = model_id
        self.ui_server = ui_server

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set — TTS will be silent")
            self._client = None
        else:
            self._client = ElevenLabs(api_key=api_key)

        self._lock = threading.Lock()
        self._pygame_ready = False

    def speak(self, text: str, priority: str = "medium"):
        """Synthesize and play speech. Non-blocking for low/medium priority."""
        prefix = _PREFIXES.get(priority, ">> ")
        print(f"{prefix}{text}")

        if not self.audio or not self._client:
            return

        if priority in ("high", "critical"):
            # High-priority: interrupt current audio, synthesize synchronously
            self._stop_playback()
            self._synthesize_and_play(text, priority)
        else:
            # Low/medium: synthesize in background thread
            threading.Thread(
                target=self._synthesize_and_play,
                args=(text, priority),
                daemon=True,
            ).start()

    def generate_file(self, text: str, output_path: str | Path) -> Path:
        """Generate speech and save to an MP3 file. Returns the output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._client:
            raise RuntimeError("ELEVENLABS_API_KEY not set — cannot generate audio")

        audio_iter = self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_iter)

        output_path.write_bytes(audio_bytes)
        logger.info("Saved audio: %s (%d bytes)", output_path, len(audio_bytes))
        return output_path

    def _synthesize_and_play(self, text: str, priority: str):
        """Call ElevenLabs API, play locally, and send over WebSocket."""
        try:
            audio_iter = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio_iter)
        except Exception:
            logger.exception("ElevenLabs API call failed")
            return

        # Play locally via pygame
        self._play_bytes(audio_bytes)

        # Send to phone UI over WebSocket
        if self.ui_server:
            b64 = base64.b64encode(audio_bytes).decode("ascii")
            self.ui_server.fire_audio(b64, priority)

    def _play_bytes(self, audio_bytes: bytes):
        """Play MP3 bytes through speakers using pygame.mixer."""
        try:
            import pygame

            with self._lock:
                if not self._pygame_ready:
                    _init_pygame()
                    self._pygame_ready = True

            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            sound.play()
            # Block until this sound finishes (so sequential speaks don't overlap)
            while pygame.mixer.get_busy():
                pygame.time.wait(50)
        except Exception:
            logger.exception("Local audio playback failed")

    def _stop_playback(self):
        """Stop any currently playing audio (for priority interrupts)."""
        try:
            import pygame

            if self._pygame_ready and pygame.mixer.get_init():
                pygame.mixer.stop()
        except Exception:
            pass
