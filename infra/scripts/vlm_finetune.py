"""
Quick VLM fine-tune: Gemma 3 4B on kitchen grid cells using Unsloth.
Train on labeled cropped images, then assess improvement.

Usage:
    python /tmp/vlm_finetune.py /tmp/cropped/
    python /tmp/vlm_finetune.py /tmp/cropped/ --epochs 5 --lr 2e-4
"""

import argparse
import json
import time
import torch
from pathlib import Path
from PIL import Image

# ── Ground truth ─────────────────────────────────────────────────────────────

LABELS = {
    "1_cold":       {"state": "cold",      "safe": True,  "reason": "Not cooking yet, no heat"},
    "2_simmer":     {"state": "simmering",  "safe": True,  "reason": "Gentle simmer, under control"},
    "3_boil":       {"state": "boiling",    "safe": True,  "reason": "Active boil but contained in pot"},
    "4_BOIL_OVER":  {"state": "boil_over",  "safe": False, "reason": "Liquid spilling over pot rim onto stove"},
    "5_done_ok":    {"state": "done",       "safe": True,  "reason": "Cooking complete, looks good"},
    "6_FAILED":     {"state": "burnt",      "safe": False, "reason": "Food overcooked, burnt or ruined"},
}

DISH_NAMES = {
    "pasta": "pasta", "ramen": "ramen noodles", "mapo_tofu": "mapo tofu",
    "curry": "Japanese curry", "soup": "chicken soup", "rice": "white rice",
}

SYSTEM = (
    "You are a kitchen safety monitor. Look at this top-down photo of a stove. "
    "Respond with ONLY a JSON object, no other text:\n"
    '{"dish": "<what food>", "state": "<cold|simmering|boiling|boil_over|done|burnt>", '
    '"safe": true/false, "reason": "<10 words max>"}'
)

DISHES = ["pasta", "ramen", "mapo_tofu", "curry", "soup", "rice"]
STATES = ["1_cold", "2_simmer", "3_boil", "4_BOIL_OVER", "5_done_ok", "6_FAILED"]


def build_samples(img_dir):
    samples = []
    for dish in DISHES:
        for state in STATES:
            img_path = img_dir / dish / f"{state}.png"
            if not img_path.exists():
                continue
            label = LABELS[state]
            samples.append({
                "image_path": str(img_path),
                "dish": dish, "state": state,
                "is_failure": not label["safe"],
                "response": json.dumps({
                    "dish": DISH_NAMES.get(dish, dish),
                    "state": label["state"],
                    "safe": label["safe"],
                    "reason": label["reason"],
                }),
            })
    fail_count = sum(1 for s in samples if s["is_failure"])
    print(f"Dataset: {len(samples)} samples ({fail_count} failure)")
    return samples


def train(args):
    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig

    img_dir = Path(args.image_dir)
    samples = build_samples(img_dir)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading {args.model}...")
    t0 = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    print(f"Loaded in {time.time() - t0:.1f}s | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # ── Format dataset ───────────────────────────────────────────────────────
    def to_conversation(sample):
        image = Image.open(sample["image_path"]).convert("RGB")
        return {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["response"]},
            ]},
        ]}

    # Oversample failures 2x for balance
    balanced = []
    for s in samples:
        balanced.append(s)
        if s["is_failure"]:
            balanced.append(s)
    print(f"Balanced: {len(balanced)} samples")

    dataset = [to_conversation(s) for s in balanced]

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nTraining {args.epochs} epochs, lr={args.lr}...")
    t0 = time.time()

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/tmp/vlm_ft_output",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=args.lr,
            warmup_steps=5,
            logging_steps=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=1024,
        ),
    )

    trainer.train()
    print(f"\nTrained in {time.time() - t0:.1f}s")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ── Assess ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("POST FINE-TUNE ASSESSMENT")
    print("=" * 60)

    FastVisionModel.for_inference(model)

    results = []
    correct_safe = 0
    correct_state = 0
    total = 0

    for sample in samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        total += 1
        gt_safe = not sample["is_failure"]
        gt_state = LABELS[sample["state"]]["state"]

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": SYSTEM},
        ]}]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
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

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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

        safe_ok = pred_safe == gt_safe
        state_ok = pred_state == gt_state if pred_state else False
        if safe_ok:
            correct_safe += 1
        if state_ok:
            correct_state += 1

        tag = "<<<FAIL" if sample["is_failure"] else ""
        s_icon = "OK" if safe_ok else "XX"
        t_icon = "OK" if state_ok else "XX"
        print(f"  {sample['dish']}/{sample['state']} ({gen_time:.1f}s) {tag}")
        print(f"    state={pred_state} [{t_icon}] safe={pred_safe} [{s_icon}]")

        results.append({
            "dish": sample["dish"], "state": sample["state"],
            "gt_safe": gt_safe, "pred_safe": pred_safe,
            "gt_state": gt_state, "pred_state": pred_state,
            "safe_ok": safe_ok, "state_ok": state_ok,
            "time": gen_time, "raw": response[:300],
        })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Safety: {correct_safe}/{total} ({correct_safe/total*100:.0f}%)")
    print(f"State:  {correct_state}/{total} ({correct_state/total*100:.0f}%)")

    tp = sum(1 for r in results if r["gt_safe"] and r["pred_safe"] is True)
    tn = sum(1 for r in results if not r["gt_safe"] and r["pred_safe"] is False)
    fp = sum(1 for r in results if not r["gt_safe"] and r["pred_safe"] is True)
    fn = sum(1 for r in results if r["gt_safe"] and r["pred_safe"] is False)
    print(f"\n  TP={tp} TN={tn} FP={fp} {'<< DANGEROUS' if fp else '(ZERO!)'} FN={fn}")

    print(f"\n  BEFORE -> AFTER")
    print(f"  Safety: 75% -> {correct_safe/total*100:.0f}%")
    print(f"  State:  25% -> {correct_state/total*100:.0f}%")
    print(f"  FP:     9   -> {fp}")

    model.save_pretrained("/tmp/vlm_ft_lora")
    tokenizer.save_pretrained("/tmp/vlm_ft_lora")
    print(f"\nLoRA saved: /tmp/vlm_ft_lora")

    with open(img_dir / "vlm_ft_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_dir")
    p.add_argument("--model", default="unsloth/gemma-3-4b-it")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
