"""
Model loader for kitchen policy inference.

Loads base Gemma 3 4B + LoRA adapter using transformers + peft (not Unsloth).
Formats prompts using the exact same chat template as training.
Handles device detection (CUDA / MPS / CPU).
"""

import json
import logging

logger = logging.getLogger(__name__)

# Must match infra/fine_tuning/train.py exactly
SYSTEM_PROMPT = (
    "You are a kitchen monitoring agent. Given the current kitchen state as JSON, "
    "decide what actions to take. Output a JSON object with an 'actions' array. "
    "Available tools: set_timer(name, seconds), adjust_timer(name, delta_seconds), "
    "speak(text, priority), show_card(title, bullets), mark_step_done(step_id), "
    "reorder_steps(new_order). Only act when necessary. If no action is needed, "
    "return an empty actions array."
)


def _detect_device() -> str:
    """Pick the best available device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class KitchenPolicyModel:
    """Loads base model + LoRA adapter and runs inference."""

    def __init__(
        self,
        base_model: str = "google/gemma-3-4b-it",
        adapter_path: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.device = device or _detect_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    def _resolve_base_model(self) -> str:
        """
        Resolve the correct base model name for inference.

        Unsloth trains with 'unsloth/gemma-3-4b-it' but the adapter is
        architecturally identical to 'google/gemma-3-4b-it'.  If the adapter's
        adapter_config.json references an unsloth/* model, map it to the
        canonical google/* name so we don't pull in Unsloth at inference time.
        """
        import json as _json
        from pathlib import Path

        if not self.adapter_path:
            return self.base_model_name

        config_file = Path(self.adapter_path) / "adapter_config.json"
        if config_file.exists():
            cfg = _json.loads(config_file.read_text())
            saved_base = cfg.get("base_model_name_or_path", "")
            if saved_base.startswith("unsloth/"):
                canonical = saved_base.replace("unsloth/", "google/", 1)
                logger.info(
                    "Adapter trained with %s — loading canonical %s instead",
                    saved_base, canonical,
                )
                return canonical
            if saved_base:
                return saved_base

        return self.base_model_name

    def load(self):
        """
        Load base model and optionally merge LoRA adapter.

        Compatible with Unsloth-trained adapters:
        - Reads adapter_config.json to resolve the correct base model
        - Loads tokenizer from the adapter dir (Unsloth saves a modified
          tokenizer with the gemma-3 chat template applied)
        - LoRA weights saved by Unsloth are standard PEFT format
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        base_model = self._resolve_base_model()

        # Prefer tokenizer saved alongside the adapter (Unsloth applies
        # get_chat_template(tokenizer, "gemma-3") during training and saves
        # the modified tokenizer with the adapter).  Fall back to base model.
        tokenizer_source = self.adapter_path or base_model
        logger.info("Loading tokenizer from %s", tokenizer_source)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            logger.warning(
                "Tokenizer not found at adapter path, falling back to %s", base_model
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        logger.info("Loading base model from %s onto %s", base_model, self.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=self.device if self.device == "cuda" else None,
        )

        if self.adapter_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter from %s", self.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()
            logger.info("LoRA adapter merged")

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model ready on %s", self.device)

    def generate(self, state_dict: dict) -> str:
        """
        Build the chat prompt from a kitchen state dict,
        run inference, and return the raw generated text.
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Current kitchen state:\n{json.dumps(state_dict, indent=2)}",
            },
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Decode only the newly generated tokens (strip the prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
