# ElevenLabs Voice Module Design

## Goal

Replace macOS `say` TTS with ElevenLabs API in the existing `TTSEngine` class.
Support both real-time speech during agent execution and pre-generated audio clips
for demos. Deliver audio via local speakers (pygame) and WebSocket to phone UI.

## Architecture

### Modified files

| File | Change |
|------|--------|
| `src/agent/tts.py` | Replace `say` backend with ElevenLabs + pygame playback |
| `src/ui/server.py` | Add `send_audio` / `fire_audio` + `/audio/` HTTP route |
| `requirements.txt` | Add `elevenlabs`, `pygame`, `python-dotenv` |

### New files

| File | Purpose |
|------|---------|
| `scripts/generate_audio.py` | CLI to pre-generate audio from recipe steps |
| `.env.example` | Document `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` |

## TTSEngine API

```python
class TTSEngine:
    def __init__(self, audio=True, voice_id=None, ui_server=None)
    def speak(self, text: str, priority: str = "medium") -> None
    def generate_file(self, text: str, output_path: str) -> Path
```

- `speak()` calls ElevenLabs API, plays via pygame.mixer (non-blocking),
  and sends base64 MP3 over WebSocket if ui_server is attached.
- Priority interruption logic preserved: high/critical kills ongoing playback.
- Console print with priority prefixes kept for logging.

## WebSocket Protocol

New message type:
```json
{"type": "audio", "base64": "<mp3-bytes>", "priority": "medium"}
```

## Pre-generation Script

```bash
python scripts/generate_audio.py \
  --recipe configs/recipes/rice.yaml \
  --output data/audio/rice/
```

Generates: `step_1.mp3`, `step_2.mp3`, ..., `safety_boilover.mp3`, `recipe_done.mp3`

## Dependencies

- `elevenlabs` — official Python SDK
- `pygame` — cross-platform audio playback
- `python-dotenv` — load `.env` for API key

## Security

- API key stored in `.env` (gitignored), never hardcoded
- `.env.example` documents required vars without real values
