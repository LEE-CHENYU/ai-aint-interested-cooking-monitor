#!/usr/bin/env python3
"""
Test merged VLM inference locally on mapo tofu images.

Measures per-image latency and checks prediction quality.
Runs on MPS (Apple Silicon) or CPU.

Usage:
  python scripts/test_local_inference.py
"""

import json
import os
import re
import time
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "vlm_ft_merged" / "vlm_ft_merged"
MATRIX_DIR = ROOT / "data" / "synthetic" / "matrix" / "matrix_images" / "mapo_tofu"
VARIATIONS_DIR = ROOT / "data" / "synthetic" / "variations" / "mapo_tofu"

SYSTEM_PROMPT = (
    "You are a kitchen safety monitor. Look at this top-down photo of a stove. "
    'Respond with ONLY a JSON object, no other text: '
    '{"dish": "<what food>", "state": "<cold|simmering|boiling|boil_over|done|burnt>", '
    '"safe": true/false, "reason": "<10 words max>"}'
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_json_response(text: str) -> dict | None:
    """Extract first JSON object from model output."""
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def run_inference(model, processor, image_path: Path) -> tuple[dict | None, float]:
    """Run single image inference, return (parsed_result, latency_ms)."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": SYSTEM_PROMPT},
        ]},
    ]

    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=input_text, images=[image], return_tensors="pt",
    ).to(DEVICE)

    input_len = inputs["input_ids"].shape[1]

    t0 = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    latency = (time.monotonic() - t0) * 1000  # ms

    response_text = processor.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )
    result = parse_json_response(response_text)

    return result, latency, response_text


def main():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16 if DEVICE == "mps" else torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    load_time = time.monotonic() - t0
    print(f"Loaded in {load_time:.1f}s\n")

    # Collect all mapo tofu images
    images = []
    for f in sorted(MATRIX_DIR.iterdir()):
        if f.suffix.lower() == ".png":
            images.append(("matrix", f))
    for f in sorted(VARIATIONS_DIR.iterdir()):
        if f.suffix.lower() == ".png":
            images.append(("variation", f))

    print(f"Testing {len(images)} mapo tofu images")
    print("=" * 70)

    latencies = []
    results = []

    for source, img_path in images:
        result, latency_ms, raw = run_inference(model, processor, img_path)
        latencies.append(latency_ms)

        status = "OK" if result else "PARSE_FAIL"
        state = result.get("state", "?") if result else "?"
        safe = result.get("safe", "?") if result else "?"

        print(f"  {img_path.name:30s}  {latency_ms:6.0f}ms  "
              f"state={state:12s}  safe={str(safe):5s}  [{status}]")

        results.append({
            "file": img_path.name,
            "source": source,
            "latency_ms": round(latency_ms),
            "result": result,
            "raw": raw,
        })

    print("=" * 70)
    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]
    print(f"\nLatency:  avg={avg:.0f}ms  p50={p50:.0f}ms  p90={p90:.0f}ms")
    print(f"          min={min(latencies):.0f}ms  max={max(latencies):.0f}ms")
    print(f"Images:   {len(images)} total, "
          f"{sum(1 for r in results if r['result'])} parsed OK")

    # Save results
    out_path = ROOT / "data" / "local_inference_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": DEVICE,
            "model": str(MODEL_PATH),
            "load_time_s": round(load_time, 1),
            "latency_avg_ms": round(avg),
            "latency_p50_ms": round(p50),
            "latency_p90_ms": round(p90),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
