#!/usr/bin/env python3
"""
Automated Multi-Task VLM Fine-Tuning Pipeline.

Phase-batched orchestrator: loads each model once for all tasks, saving ~33% time.

  Phase A: Batch all baselines (1 VLM load, 4×36 inferences)
  Phase B: Batch all variations (1 SDXL load, 4×72 images)
  Phase C: Sequential train + assess + merge (per task)

Usage:
    python run_pipeline.py --output-dir /tmp/pipeline_output --time-budget 120
    python run_pipeline.py --resume  # continue from last completed phase
    python run_pipeline.py --skip-baselines --skip-generation  # training only
"""

import argparse
import json
import time
from pathlib import Path

import torch

from task_configs import TASK_CONFIGS, TASK_ORDER, BAKEOFF_TASKS
from utils import (
    clear_vram,
    vram_stats,
    parse_prompts,
    assess_task,
    build_training_dataset,
    make_epoch_callback,
    print_milestone,
    print_summary,
    print_bakeoff,
    save_pipeline_state,
    load_pipeline_state,
)


# =============================================================================
# Adaptive Time Management
# =============================================================================

BUDGET_CHECKPOINTS = {
    "after_baselines":  {"deadline_min": 20,  "action": None},
    "after_generation": {"deadline_min": 30,  "action": "reduce_variations"},
    "after_task_1":     {"deadline_min": 55,  "action": "reduce_epochs"},
    "after_task_2":     {"deadline_min": 80,  "action": "reduce_epochs"},
    "after_task_3":     {"deadline_min": 105, "action": "skip_gguf"},
}


def check_time_budget(start_time, checkpoint_name, budget_minutes, settings):
    """Check elapsed time and adjust settings if running behind."""
    elapsed_min = (time.time() - start_time) / 60
    remaining_min = budget_minutes - elapsed_min
    info = BUDGET_CHECKPOINTS.get(checkpoint_name, {})
    deadline = info.get("deadline_min")
    action = info.get("action")

    print(f"\n  [Time] {elapsed_min:.1f} min elapsed, {remaining_min:.1f} min remaining")

    if deadline and elapsed_min > deadline and action:
        if action == "reduce_variations" and settings["variations"] > 1:
            settings["variations"] = 1
            print(f"  [Time] BEHIND SCHEDULE — reducing variations to {settings['variations']}")
        elif action == "reduce_epochs" and settings["epochs"] > 2:
            settings["epochs"] = 2
            print(f"  [Time] BEHIND SCHEDULE — reducing epochs to {settings['epochs']}")
        elif action == "skip_gguf":
            settings["skip_gguf"] = True
            print("  [Time] BEHIND SCHEDULE — will skip GGUF export")

    return elapsed_min, remaining_min


# =============================================================================
# Phase A: Batch All Baselines
# =============================================================================

def run_baselines(tasks, source_dir, output_dir, state):
    """Run baseline assessment for all tasks with a single VLM load."""
    if "baselines" in state.get("completed_phases", []):
        print("\n>>> Phase A: SKIPPED (already completed) <<<")
        return state.get("baseline_results", {})

    print("\n" + "=" * 60)
    print("  PHASE A: BATCH ALL BASELINES")
    print("=" * 60)

    from unsloth import FastVisionModel

    print("\nLoading Gemma 3 4B (4-bit) for baseline assessment...")
    t0 = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/gemma-3-4b-it",
        load_in_4bit=True,
        max_seq_length=2048,
    )
    FastVisionModel.for_inference(model)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s | VRAM: {vram_stats()['current_gb']} GB")

    baseline_results = {}

    for task_name in tasks:
        config = TASK_CONFIGS[task_name]
        img_dir = Path(source_dir) / task_name / "matrix_images"
        task_output = Path(output_dir) / task_name

        # Check if baseline already done for this task
        baseline_file = task_output / "baseline_results.json"
        if baseline_file.exists():
            print(f"\n  {task_name}: loading existing baseline")
            with open(baseline_file) as f:
                baseline_results[task_name] = json.load(f)
            continue

        print(f"\n  Running baseline for {task_name} ({len(config['rows'])}×{len(config['cols'])} images)...")
        t1 = time.time()
        metrics = assess_task(model, tokenizer, config, str(img_dir))
        elapsed = time.time() - t1

        # Save results (without per-image responses for compactness)
        save_metrics = {k: v for k, v in metrics.items() if k != "results"}
        save_metrics["per_image"] = [
            {k: v for k, v in r.items() if k != "response"}
            for r in metrics["results"]
        ]
        task_output.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, "w") as f:
            json.dump(save_metrics, f, indent=2)

        baseline_results[task_name] = save_metrics
        print_milestone(task_name, "BASELINE", save_metrics, elapsed)

    # Cleanup VLM
    del model, tokenizer
    clear_vram()

    return baseline_results


# =============================================================================
# Bake-Off: Decide task order for similar tasks
# =============================================================================

def run_bakeoff(baseline_results, all_tasks):
    """Compare task1_boilover vs task6_boiling, pick primary, reorder tasks.

    The task with lower baseline accuracy becomes primary (more room to improve).
    The other becomes last in the queue (optional if time permits).
    """
    task_a, task_b = BAKEOFF_TASKS
    if task_a not in baseline_results or task_b not in baseline_results:
        print("  Bake-off: skipped (not all baselines available)")
        return all_tasks

    metrics_a = baseline_results[task_a]
    metrics_b = baseline_results[task_b]
    acc_a = metrics_a["binary_accuracy"]
    acc_b = metrics_b["binary_accuracy"]
    fp_a = metrics_a["fp"]
    fp_b = metrics_b["fp"]

    # Pick the one with lower accuracy as primary (more improvement potential)
    # Tiebreak: higher FP count (more dangerous baseline → more urgent to fix)
    if acc_a < acc_b or (acc_a == acc_b and fp_a >= fp_b):
        winner, loser = task_a, task_b
    else:
        winner, loser = task_b, task_a

    print_bakeoff(task_a, metrics_a, task_b, metrics_b, winner, loser)

    # Reorder: winner first, non-bakeoff tasks in original order, loser last
    final_order = [winner]
    for t in all_tasks:
        if t not in BAKEOFF_TASKS:
            final_order.append(t)
    final_order.append(loser)

    print(f"  Task execution order: {' -> '.join(final_order)}")
    return final_order


# =============================================================================
# Phase B: Batch All Variations
# =============================================================================

def run_variations(tasks, source_dir, output_dir, settings, state):
    """Generate image variations for all tasks with a single SDXL load."""
    if "generation" in state.get("completed_phases", []):
        print("\n>>> Phase B: SKIPPED (already completed) <<<")
        return

    print("\n" + "=" * 60)
    print("  PHASE B: BATCH ALL VARIATIONS (SDXL img2img)")
    print("=" * 60)

    from diffusers import StableDiffusionXLImg2ImgPipeline
    from PIL import Image

    NEGATIVE = (
        "cartoon, anime, illustration, drawing, painting, CGI, render, "
        "low quality, blurry, text, watermark, logo, distorted, "
        "extra objects, unrealistic colors, side angle, diagonal view"
    )

    num_variations = settings["variations"]
    strength = settings.get("strength", 0.35)

    print(f"\nLoading SDXL img2img pipeline (generating {num_variations} variations/image)...")
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    print(f"  Loaded in {time.time() - t0:.1f}s | VRAM: {vram_stats()['current_gb']} GB")

    for task_name in tasks:
        config = TASK_CONFIGS[task_name]
        img_dir = Path(source_dir) / task_name / "matrix_images"
        prompts_file = Path(source_dir) / task_name / "prompts.txt"
        var_dir = Path(output_dir) / task_name / "variations"

        # Check if variations already exist for this task
        var_labels = var_dir / "labels.json"
        if var_labels.exists():
            print(f"\n  {task_name}: variations already exist, skipping")
            continue

        if not prompts_file.exists():
            print(f"\n  {task_name}: no prompts.txt found at {prompts_file}, skipping")
            continue

        prompts = parse_prompts(str(prompts_file))
        var_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Generating variations for {task_name}...")
        t1 = time.time()
        labels = []
        count = 0
        total_expected = len(config["rows"]) * len(config["cols"]) * num_variations

        for row in config["rows"]:
            row_dir = var_dir / row
            row_dir.mkdir(parents=True, exist_ok=True)

            for col in config["cols"]:
                src_path = img_dir / row / f"{col}.png"
                key = f"{row}/{col}"
                prompt = prompts.get(key, "")

                if not src_path.exists() or not prompt:
                    continue

                src_image = Image.open(str(src_path)).convert("RGB")
                src_image = src_image.resize((768, 768), Image.LANCZOS)

                is_unsafe = col in config["unsafe_cols"]

                for v in range(num_variations):
                    count += 1
                    fname = f"{col}_v{v + 1}.png"
                    out_path = row_dir / fname

                    if out_path.exists():
                        labels.append({
                            "file": f"{row}/{fname}", "row": row, "col": col,
                            "is_unsafe": is_unsafe, "variation": v + 1,
                        })
                        continue

                    seed = 42 + v * 1000 + hash(key) % 10000
                    generator = torch.Generator("cuda").manual_seed(seed)

                    result = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE,
                        image=src_image,
                        strength=strength,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        generator=generator,
                    ).images[0]

                    result.save(str(out_path))
                    labels.append({
                        "file": f"{row}/{fname}", "row": row, "col": col,
                        "is_unsafe": is_unsafe, "variation": v + 1,
                    })

                    if count % 20 == 0:
                        tag = "UNSAFE" if is_unsafe else "safe"
                        print(f"    [{count}/{total_expected}] {row}/{fname} [{tag}]")

        with open(var_labels, "w") as f:
            json.dump(labels, f, indent=2)

        elapsed = time.time() - t1
        print_milestone(task_name, "VARIATION GEN", {
            "images_generated": count,
            "expected": total_expected,
        }, elapsed)

    # Cleanup SDXL
    del pipe
    clear_vram()


# =============================================================================
# Phase C: Train + Assess + Merge (per task)
# =============================================================================

def run_train_task(task_name, source_dir, output_dir, settings, baseline_results):
    """Train, assess, and export a single task's fine-tuned model."""
    config = TASK_CONFIGS[task_name]
    task_output = Path(output_dir) / task_name
    task_output.mkdir(parents=True, exist_ok=True)
    img_dir = Path(source_dir) / task_name / "matrix_images"
    var_dir = task_output / "variations"

    epochs = settings.get("epochs", config["epochs"])
    lr = config["lr"]
    lora_rank = config["lora_rank"]
    patience = settings["patience"]

    print(f"\n{'─' * 60}")
    print(f"  TRAINING: {task_name} ({config['description']})")
    print(f"  epochs={epochs}, lr={lr}, rank={lora_rank}, patience={patience}")
    print(f"  finetune_vision={config['finetune_vision']}")
    print(f"{'─' * 60}")

    from unsloth import FastVisionModel, is_bfloat16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # ----- Load model + LoRA -----
    print("\n  Loading Gemma 3 4B + fresh LoRA...")
    t0 = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/gemma-3-4b-it",
        load_in_4bit=True,
        max_seq_length=2048,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config["finetune_vision"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s | VRAM: {vram_stats()['current_gb']} GB")

    # ----- Build dataset -----
    print("\n  Building training dataset...")
    var_dir_str = str(var_dir) if var_dir.exists() else None
    samples = build_training_dataset(config, str(img_dir), var_dir_str)
    print(f"  Training samples: {len(samples)} (with oversampling)")

    dataset = Dataset.from_list(samples)

    # ----- Callback for per-epoch assessment -----
    callback = make_epoch_callback(
        tokenizer, config, str(img_dir), str(task_output), patience=patience
    )

    # ----- Train -----
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        callbacks=[callback],
        args=SFTConfig(
            output_dir=str(task_output / "sft_output"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            warmup_steps=5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            save_strategy="no",  # we save via callback
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )

    print("\n  Starting training...")
    t_train = time.time()
    trainer.train()
    train_time = time.time() - t_train
    print(f"\n  Training complete in {train_time / 60:.1f} min")

    # ----- Final assessment -----
    print("\n  Running final assessment on source images...")
    FastVisionModel.for_inference(model)
    final_metrics = assess_task(model, tokenizer, config, str(img_dir))

    # Save final results
    final_save = {k: v for k, v in final_metrics.items() if k != "results"}
    final_save["per_image"] = [
        {k: v for k, v in r.items() if k != "response"}
        for r in final_metrics["results"]
    ]
    final_save["best_epoch"] = callback.best_epoch
    final_save["epoch_results"] = callback.epoch_results
    with open(task_output / "final_results.json", "w") as f:
        json.dump(final_save, f, indent=2)

    # Print before→after comparison
    baseline = baseline_results.get(task_name, {})
    print_milestone(task_name, "TRAINING COMPLETE", {
        "baseline_accuracy": baseline.get("binary_accuracy", 0),
        "final_accuracy": final_metrics["binary_accuracy"],
        "delta": final_metrics["binary_accuracy"] - baseline.get("binary_accuracy", 0),
        "baseline_fp": baseline.get("fp", "?"),
        "final_fp": final_metrics["fp"],
        "best_epoch": callback.best_epoch,
        "train_time_min": round(train_time / 60, 1),
    })

    # ----- Export: merge LoRA into base weights -----
    print("\n  Merging LoRA into base model (16-bit)...")
    merged_dir = task_output / "merged_model"
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  Merged model saved to: {merged_dir}")

    # ----- Export: GGUF (optional) -----
    if not settings.get("skip_gguf", False):
        print("\n  Exporting GGUF (q4_k_m)...")
        try:
            gguf_dir = task_output / "gguf"
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method="q4_k_m",
            )
            print(f"  GGUF saved to: {gguf_dir}")
        except Exception as e:
            print(f"  GGUF export failed: {e} (non-critical, skipping)")
    else:
        print("\n  Skipping GGUF export (time budget)")

    # Cleanup
    del model, tokenizer, trainer
    clear_vram()

    return {
        "baseline_accuracy": baseline.get("binary_accuracy", 0),
        "final_accuracy": final_metrics["binary_accuracy"],
        "baseline_fp": baseline.get("fp", "?"),
        "final_fp": final_metrics["fp"],
        "best_epoch": callback.best_epoch,
        "train_time_min": round(train_time / 60, 1),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated Multi-Task VLM Fine-Tuning Pipeline"
    )
    parser.add_argument(
        "--output-dir", default="/tmp/pipeline_output",
        help="Output directory for all results (default: /tmp/pipeline_output)",
    )
    parser.add_argument(
        "--source-dir", default="/tmp/matrix",
        help="Path to matrix data (default: /tmp/matrix)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Subset of tasks to run (default: all 4)",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline assessment (reuse existing results)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip variation generation (reuse existing variations)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last completed phase",
    )
    parser.add_argument(
        "--variations", type=int, default=2,
        help="Variations per image (default: 2)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Max training epochs (default: 3)",
    )
    parser.add_argument(
        "--patience", type=int, default=1,
        help="Early stopping patience (default: 1)",
    )
    parser.add_argument(
        "--strength", type=float, default=0.35,
        help="SDXL img2img strength (default: 0.35)",
    )
    parser.add_argument(
        "--time-budget", type=int, default=120,
        help="Hard time limit in minutes (default: 120)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir

    # Determine which tasks to run
    requested_tasks = args.tasks or list(TASK_ORDER)
    for t in requested_tasks:
        if t not in TASK_CONFIGS:
            print(f"ERROR: unknown task '{t}'. Available: {list(TASK_CONFIGS.keys())}")
            return

    # Mutable settings (can be adjusted by time management)
    settings = {
        "variations": args.variations,
        "epochs": args.epochs,
        "patience": args.patience,
        "strength": args.strength,
        "skip_gguf": False,
    }

    # Load pipeline state for resume (ensure completed_phases always exists)
    state = load_pipeline_state(str(output_dir)) if args.resume else {}
    state.setdefault("completed_phases", [])
    state.setdefault("completed_tasks", [])

    start_time = time.time()
    budget = args.time_budget

    print("=" * 60)
    print("  AUTOMATED MULTI-TASK VLM FINE-TUNING PIPELINE")
    print("=" * 60)
    print(f"  Tasks: {requested_tasks}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Variations: {settings['variations']}, Epochs: {settings['epochs']}, "
          f"Patience: {settings['patience']}")
    print(f"  Time budget: {budget} min")
    if args.resume:
        print(f"  Resuming from state: {state.get('completed_phases', [])}")
    print()

    # ─── PHASE A: BATCH ALL BASELINES ───
    if not args.skip_baselines:
        baseline_results = run_baselines(requested_tasks, source_dir, str(output_dir), state)

        if "baselines" not in state["completed_phases"]:
            state["completed_phases"].append("baselines")
        state["baseline_results"] = baseline_results
        save_pipeline_state(str(output_dir), state)

        _, remaining = check_time_budget(start_time, "after_baselines", budget, settings)
    else:
        # Load existing baselines
        baseline_results = {}
        for task_name in requested_tasks:
            bf = Path(output_dir) / task_name / "baseline_results.json"
            if bf.exists():
                with open(bf) as f:
                    baseline_results[task_name] = json.load(f)
        print("\n>>> Phase A: SKIPPED (--skip-baselines) <<<")

    # ─── BAKE-OFF: Decide task order ───
    if state.get("task_order"):
        task_order = state["task_order"]
        print(f"\n  Using saved task order: {' -> '.join(task_order)}")
    else:
        task_order = run_bakeoff(baseline_results, requested_tasks)
        state["task_order"] = task_order
        save_pipeline_state(str(output_dir), state)

    # ─── PHASE B: BATCH ALL VARIATIONS ───
    if not args.skip_generation:
        run_variations(task_order, source_dir, str(output_dir), settings, state)

        if "generation" not in state["completed_phases"]:
            state["completed_phases"].append("generation")
        save_pipeline_state(str(output_dir), state)

        _, remaining = check_time_budget(start_time, "after_generation", budget, settings)
    else:
        print("\n>>> Phase B: SKIPPED (--skip-generation) <<<")

    # ─── PHASE C: SEQUENTIAL TRAIN + ASSESS + MERGE ───
    print("\n" + "=" * 60)
    print("  PHASE C: SEQUENTIAL TRAIN + ASSESS + MERGE")
    print("=" * 60)

    all_results = {}
    completed_tasks = state.get("completed_tasks", [])

    for i, task_name in enumerate(task_order):
        if task_name in completed_tasks:
            print(f"\n  {task_name}: already completed, skipping")
            # Load existing results
            final_f = Path(output_dir) / task_name / "final_results.json"
            if final_f.exists():
                with open(final_f) as f:
                    data = json.load(f)
                all_results[task_name] = {
                    "baseline_accuracy": baseline_results.get(task_name, {}).get("binary_accuracy", 0),
                    "final_accuracy": data.get("binary_accuracy", 0),
                    "baseline_fp": baseline_results.get(task_name, {}).get("fp", "?"),
                    "final_fp": data.get("fp", "?"),
                    "best_epoch": data.get("best_epoch", "?"),
                }
            continue

        # Time check: enough time for this task?
        elapsed_min = (time.time() - start_time) / 60
        remaining_min = budget - elapsed_min
        if remaining_min < 25:
            print(f"\n  SKIPPING {task_name}: only {remaining_min:.0f} min remaining (need 25)")
            continue

        # Run train + assess + merge
        result = run_train_task(
            task_name, source_dir, str(output_dir), settings, baseline_results
        )
        all_results[task_name] = result

        # Update state
        completed_tasks.append(task_name)
        state["completed_tasks"] = completed_tasks
        save_pipeline_state(str(output_dir), state)

        # Time checkpoint
        checkpoint_name = f"after_task_{i + 1}"
        check_time_budget(start_time, checkpoint_name, budget, settings)

    # ─── FINAL SUMMARY ───
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {total_elapsed / 60:.1f} min")
    print(f"  Tasks completed: {len(all_results)}/{len(task_order)}")

    if all_results:
        print_summary(all_results)

        # Save summary JSON
        summary = {
            "total_time_min": round(total_elapsed / 60, 1),
            "tasks_completed": len(all_results),
            "tasks_requested": len(task_order),
            "task_order": task_order,
            "settings": settings,
            "results": all_results,
        }
        with open(output_dir / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to: {output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
