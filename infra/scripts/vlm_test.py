"""
Test Gemma 3 4B as a VLM on cropped kitchen images.
Feed each cell from the grid and ask: what's happening? is this safe?

Usage (on GCP instance):
    python /tmp/vlm_test.py /tmp/cropped/
"""

import argparse
import json
import os
import time
import glob
import base64
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image


SYSTEM = (
    "You are a kitchen safety monitor. Look at this top-down photo of a stove. "
    "Respond with ONLY a JSON object, no other text:\n"
    '{"dish": "<what food>", "state": "<cold|simmering|boiling|boil_over|done|burnt>", '
    '"safe": true/false, "reason": "<10 words max>"}'
)

DISHES = ["pasta", "ramen", "mapo_tofu", "curry", "soup", "rice"]
STATES = ["1_cold", "2_simmer", "3_boil", "4_BOIL_OVER", "5_done_ok", "6_FAILED"]


def run(args):
    img_dir = Path(args.image_dir)

    print("=" * 60)
    print("VLM TEST: Gemma 3 4B on kitchen grid cells")
    print("=" * 60)

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    results = []
    total = 0
    correct_state = 0
    correct_safe = 0

    for dish in DISHES:
        dish_dir = img_dir / dish
        if not dish_dir.exists():
            continue

        for state_file in sorted(dish_dir.glob("*.png")):
            state_name = state_file.stem
            total += 1

            # Ground truth
            is_failure = "BOIL_OVER" in state_name or "FAILED" in state_name
            gt_safe = not is_failure

            # Load image
            image = Image.open(state_file).convert("RGB")

            # Build message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": SYSTEM},
                    ],
                }
            ]

            input_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[input_text],
                images=[image],
                return_tensors="pt",
            ).to("cuda")

            t1 = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                )
            gen_time = time.time() - t1

            response = processor.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Parse JSON
            parsed = None
            pred_safe = None
            pred_state = None
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > 0:
                    parsed = json.loads(response[start:end])
                    pred_safe = parsed.get("safe")
                    pred_state = parsed.get("state", "")
            except (json.JSONDecodeError, ValueError):
                pass

            # Score
            safe_match = pred_safe == gt_safe if pred_safe is not None else False
            if safe_match:
                correct_safe += 1

            # State matching (loose)
            gt_state_category = state_name.split("_", 1)[-1] if "_" in state_name else state_name
            state_match = False
            if pred_state:
                ps = pred_state.lower()
                if "BOIL_OVER" in state_name and ("boil_over" in ps or "overflow" in ps or "spill" in ps):
                    state_match = True
                elif "FAILED" in state_name and ("burnt" in ps or "burn" in ps or "fail" in ps or "overcook" in ps):
                    state_match = True
                elif "cold" in state_name and ("cold" in ps or "raw" in ps or "uncooked" in ps):
                    state_match = True
                elif "simmer" in state_name and ("simmer" in ps or "cooking" in ps or "gentle" in ps):
                    state_match = True
                elif "boil" in state_name.lower() and "BOIL_OVER" not in state_name and ("boil" in ps or "rolling" in ps or "vigorous" in ps):
                    state_match = True
                elif "done" in state_name and ("done" in ps or "cooked" in ps or "ready" in ps or "finish" in ps):
                    state_match = True
            if state_match:
                correct_state += 1

            safe_icon = "OK" if safe_match else "XX"
            state_icon = "OK" if state_match else "XX"
            fail_tag = " <<<FAIL" if is_failure else ""

            print(f"\n  {dish}/{state_name} ({gen_time:.1f}s){fail_tag}")
            print(f"    Response: {response[:200]}")
            print(f"    Safe: pred={pred_safe} gt={gt_safe} [{safe_icon}] | State: pred={pred_state} [{state_icon}]")

            results.append({
                "dish": dish,
                "state": state_name,
                "gt_safe": gt_safe,
                "pred_safe": pred_safe,
                "pred_state": pred_state,
                "safe_correct": safe_match,
                "state_correct": state_match,
                "time": gen_time,
                "raw_response": response[:300],
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images:     {total}")
    print(f"Safety accuracy:  {correct_safe}/{total} ({correct_safe/total*100:.0f}%)")
    print(f"State accuracy:   {correct_state}/{total} ({correct_state/total*100:.0f}%)")
    avg_time = sum(r["time"] for r in results) / total if total > 0 else 0
    print(f"Avg gen time:     {avg_time:.2f}s")

    # Per-dish breakdown
    print(f"\n{'DISH':<12} {'SAFE':>6} {'STATE':>6}")
    print("-" * 28)
    for dish in DISHES:
        dish_results = [r for r in results if r["dish"] == dish]
        if not dish_results:
            continue
        ds = sum(1 for r in dish_results if r["safe_correct"])
        dt = sum(1 for r in dish_results if r["state_correct"])
        n = len(dish_results)
        print(f"{dish:<12} {ds}/{n:>4} {dt}/{n:>4}")

    # Confusion on safety
    tp = sum(1 for r in results if r["gt_safe"] and r["pred_safe"] is True)
    tn = sum(1 for r in results if not r["gt_safe"] and r["pred_safe"] is False)
    fp = sum(1 for r in results if not r["gt_safe"] and r["pred_safe"] is True)
    fn = sum(1 for r in results if r["gt_safe"] and r["pred_safe"] is False)
    print(f"\nSafety confusion matrix:")
    print(f"  TP (safe=safe):     {tp}")
    print(f"  TN (danger=danger): {tn}")
    print(f"  FP (danger->safe):  {fp}  {'<< DANGEROUS' if fp > 0 else ''}")
    print(f"  FN (safe->danger):  {fn}  (false alarm, acceptable)")

    print(f"\nVRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # Save results
    out_path = img_dir / "vlm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Directory with dish/state.png structure")
    parser.add_argument("--model", default="unsloth/gemma-3-4b-it", help="VLM model")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
