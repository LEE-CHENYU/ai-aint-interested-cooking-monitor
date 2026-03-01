"""
VLM-based detector.

Uses a multimodal model (Gemma 3 via AutoModelForImageTextToText) to
classify kitchen frames. Returns structured JSON with dish state, safety
assessment, and step-completion signal.

Mock mode cycles through scripted sequences for demo without a GPU.
"""

import json
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Base VLM system prompt
_VLM_SYSTEM = (
    "You are a kitchen safety monitor. Look at this photo of a stove."
)

# Step-aware extension template
_STEP_TEMPLATE = (
    " The cook is currently on: '{vlm_signal}'."
    ' Respond with ONLY a JSON object:'
    ' {{"dish": "...", "state": "...", "safe": true/false,'
    ' "reason": "...", "step_complete": true/false}}'
)

# Fallback (no active VLM step)
_NO_STEP_TEMPLATE = (
    ' Respond with ONLY a JSON object:'
    ' {{"dish": "...", "state": "...", "safe": true/false, "reason": "..."}}'
)

# ---------------------------------------------------------------------------
# Mock sequences for demo mode
# ---------------------------------------------------------------------------

MOCK_SEQUENCES: dict[str, list[dict]] = {
    "Pasta with Tomato Sauce": [
        # Phase 1: Water heating (step 1 — VLM completion)
        # Need 2/3 True frames in a window of 3 to pass temporal smoothing
        {"dish": "pasta", "state": "cold", "safe": True, "reason": "water not hot yet", "step_complete": False},
        {"dish": "pasta", "state": "simmering", "safe": True, "reason": "water heating up", "step_complete": False},
        {"dish": "pasta", "state": "boiling", "safe": True, "reason": "rolling boil reached", "step_complete": True},
        {"dish": "pasta", "state": "boiling", "safe": True, "reason": "at rolling boil", "step_complete": True},
        {"dish": "pasta", "state": "boiling", "safe": True, "reason": "still boiling", "step_complete": True},
        # Phase 2: Cooking (timers handle steps 2-5)
        {"dish": "pasta", "state": "boiling", "safe": True, "reason": "pasta cooking", "step_complete": False},
        {"dish": "pasta", "state": "boiling", "safe": True, "reason": "pasta cooking normally", "step_complete": False},
        # Phase 3: Safety event — boil over risk
        {"dish": "pasta", "state": "boil_over", "safe": False, "reason": "pot overflowing", "step_complete": False},
        {"dish": "pasta", "state": "boiling", "safe": True, "reason": "situation resolved", "step_complete": False},
        # Phase 4: Smoke event
        {"dish": "pasta", "state": "simmering", "safe": False, "reason": "smoke detected from sauce pan", "step_complete": False},
        {"dish": "pasta", "state": "simmering", "safe": True, "reason": "smoke cleared", "step_complete": False},
        # Phase 5: Finishing
        {"dish": "pasta", "state": "done", "safe": True, "reason": "cooking complete", "step_complete": False},
    ],
}


def _parse_vlm_json(raw: str) -> dict | None:
    """Multi-layered JSON extraction from VLM output."""
    # Layer 1: direct parse
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: markdown fence
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Layer 3: first {...} substring
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


class VLMDetector:
    """Detects kitchen state from frames using a VLM or mock sequences."""

    def __init__(self, model_path: str | None = None, mock: bool = True):
        self.mock = mock
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._mock_index: dict[str, int] = {}  # per-dish cycle index

        if not mock and model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load VLM using AutoModelForImageTextToText (NOT AutoModelForCausalLM)."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info("Loading VLM from %s...", model_path)

        device = "cpu"
        dtype = torch.float32
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()
        self._device = device
        logger.info("VLM loaded on %s", device)

    def detect(self, frame: np.ndarray, step_signal: str | None = None) -> dict:
        """
        Run detection on a single frame.

        Args:
            frame: RGB image as numpy array
            step_signal: current step's vlm_signal (e.g. "water is at a rolling boil")

        Returns:
            dict with keys: dish, state, safe, reason, step_complete (if step_signal given)
        """
        if self.mock:
            return self._detect_mock()
        return self._detect_real(frame, step_signal)

    def _detect_mock(self, dish: str = "Pasta with Tomato Sauce") -> dict:
        """Cycle through mock sequences for the given dish."""
        seq = MOCK_SEQUENCES.get(dish, MOCK_SEQUENCES.get("Pasta with Tomato Sauce", []))
        if not seq:
            return {"dish": "unknown", "state": "unknown", "safe": True, "reason": "no mock data"}

        idx = self._mock_index.get(dish, 0)
        result = seq[idx % len(seq)]
        self._mock_index[dish] = idx + 1
        return dict(result)

    def _detect_real(self, frame: np.ndarray, step_signal: str | None = None) -> dict:
        """Run real VLM inference on a frame."""
        import torch
        from PIL import Image

        if self._model is None or self._processor is None:
            logger.warning("VLM not loaded, returning default")
            return {"dish": "unknown", "state": "unknown", "safe": True, "reason": "model not loaded"}

        # Build step-aware prompt
        if step_signal:
            prompt_text = _VLM_SYSTEM + _STEP_TEMPLATE.format(vlm_signal=step_signal)
        else:
            prompt_text = _VLM_SYSTEM + _NO_STEP_TEMPLATE

        image = Image.fromarray(frame)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        input_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[input_text],
            images=[image],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
            )

        response = self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        logger.debug("VLM raw: %s", response[:200])

        parsed = _parse_vlm_json(response)
        if parsed:
            return parsed

        logger.warning("Could not parse VLM output: %s", response[:200])
        return {"dish": "unknown", "state": "unknown", "safe": True, "reason": "parse error"}
