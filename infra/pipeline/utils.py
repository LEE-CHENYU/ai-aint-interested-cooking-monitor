"""
Shared utilities for the VLM fine-tuning pipeline.

- VRAM management
- Prompt parsing
- Dataset building with oversampling
- VLM assessment (binary accuracy, FP, FN)
- Per-epoch assessment callback with early stopping
- Milestone reporting
"""

import gc
import json
import shutil
import time
from pathlib import Path

import torch
from PIL import Image


# =============================================================================
# VRAM Management
# =============================================================================

def clear_vram():
    """Aggressively clear GPU memory between phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def vram_stats():
    """Return current and peak VRAM usage in GB."""
    if not torch.cuda.is_available():
        return {"current_gb": 0, "peak_gb": 0}
    return {
        "current_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }


# =============================================================================
# Prompt Parsing
# =============================================================================

def parse_prompts(prompt_file):
    """Parse prompts.txt into {row/col: prompt} dict.

    Format: {row}/{col} | {prompt text}
    Lines starting with # or NEGATIVE: are skipped.
    """
    prompts = {}
    with open(prompt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("NEGATIVE"):
                continue
            if "|" not in line:
                continue
            key, prompt = line.split("|", 1)
            prompts[key.strip()] = prompt.strip()
    return prompts


# =============================================================================
# Image Path Resolution
# =============================================================================

def find_image(base_dir, row, col):
    """Find an image file, handling both flat and nested directory layouts.

    Checks (in order):
      1. {base_dir}/{row}/{col}.png       (nested: row subdirectory)
      2. {base_dir}/{row}_{col}.png       (flat: underscore-separated)
    Returns the Path if found, else None.
    """
    base = Path(base_dir)
    for candidate in [
        base / row / f"{col}.png",
        base / f"{row}_{col}.png",
    ]:
        if candidate.exists():
            return candidate
    return None


def find_variation(base_dir, row, col, v):
    """Find a variation image, handling both flat and nested layouts.

    Checks (in order):
      1. {base_dir}/{row}/{col}_v{v}.png  (nested)
      2. {base_dir}/{row}_{col}_v{v}.png  (flat)
    """
    base = Path(base_dir)
    for candidate in [
        base / row / f"{col}_v{v}.png",
        base / f"{row}_{col}_v{v}.png",
    ]:
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# JSON Response Parsing
# =============================================================================

def parse_json_response(response_text):
    """Extract first JSON object from model response text."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


# =============================================================================
# Dataset Building
# =============================================================================

def build_ground_truth(task_config, row, col):
    """Build the expected JSON response string for a training sample."""
    row_label = task_config["row_labels"][row]
    col_label = task_config["col_labels"][col]
    binary_key = task_config["binary_key"]

    response = {
        task_config["row_field"]: row_label,
        task_config["state_field"]: col_label["state"],
        binary_key: col_label[binary_key],
        "confidence": 1.0,
        "reason": col_label["reason"],
    }
    return json.dumps(response)


def build_training_dataset(task_config, source_dir, variations_dir=None):
    """Build training dataset from source images + optional variations.

    Returns list of dicts with 'messages' and 'images' keys,
    ready for HuggingFace Dataset + UnslothVisionDataCollator.

    Applies oversampling to unsafe/negative samples.
    """
    samples = []
    system_prompt = task_config["system_prompt"]
    unsafe_cols = set(task_config["unsafe_cols"])
    oversample = task_config.get("oversample_unsafe", 1)

    source_path = Path(source_dir)
    var_path = Path(variations_dir) if variations_dir else None

    for row in task_config["rows"]:
        for col in task_config["cols"]:
            gt_response = build_ground_truth(task_config, row, col)
            is_unsafe = col in unsafe_cols
            repeat = oversample if is_unsafe else 1

            # Source image (handles both flat and nested layouts)
            img_path = find_image(source_path, row, col)
            if img_path:
                for _ in range(repeat):
                    samples.append(_make_sample(img_path, system_prompt, gt_response))

            # Variation images
            if var_path:
                v = 1
                while True:
                    vimg = find_variation(var_path, row, col, v)
                    if not vimg:
                        break
                    for _ in range(repeat):
                        samples.append(_make_sample(vimg, system_prompt, gt_response))
                    v += 1

    return samples


def _make_sample(img_path, system_prompt, gt_response):
    """Create a single training sample dict."""
    image = Image.open(str(img_path)).convert("RGB")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": gt_response},
                ],
            },
        ],
        "images": [image],
    }


# =============================================================================
# Assessment / Scoring
# =============================================================================

def assess_task(model, tokenizer, task_config, source_dir):
    """Run model on source images for a task and compute metrics.

    Returns dict with binary_accuracy, correct, total, fp, fn, and per-image results.
    """
    results = []
    source_path = Path(source_dir)
    binary_key = task_config["binary_key"]
    system_prompt = task_config["system_prompt"]

    for row in task_config["rows"]:
        for col in task_config["cols"]:
            img_path = find_image(source_path, row, col)
            if not img_path:
                continue

            image = Image.open(str(img_path)).convert("RGB")

            # Build inference messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]
            input_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = tokenizer(
                text=input_text, images=[image], return_tensors="pt",
            ).to("cuda")

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            # Decode only generated tokens (skip input to avoid matching
            # the JSON template in the system prompt)
            response_text = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            predicted = parse_json_response(response_text)
            ground_truth = task_config["col_labels"][col][binary_key]
            predicted_value = predicted.get(binary_key)

            # Handle string "true"/"false" from model output
            if isinstance(predicted_value, str):
                predicted_value = predicted_value.lower() == "true"

            results.append({
                "row": row,
                "col": col,
                "ground_truth": ground_truth,
                "predicted": predicted_value,
                "correct": predicted_value == ground_truth,
                "response": response_text[:500],
            })

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    # FP: predicted True when ground truth is False (dangerous for safety tasks)
    fp = sum(1 for r in results if r["predicted"] is True and r["ground_truth"] is False)
    # FN: predicted False when ground truth is True
    fn = sum(1 for r in results if r["predicted"] is False and r["ground_truth"] is True)
    # Parse failures: predicted value is None
    parse_fail = sum(1 for r in results if r["predicted"] is None)

    return {
        "binary_accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "fp": fp,
        "fn": fn,
        "parse_fail": parse_fail,
        "results": results,
    }


# =============================================================================
# Training Callback — Per-Epoch Assessment + Early Stopping
# =============================================================================

def make_epoch_callback(tokenizer, task_config, source_dir, output_dir, patience=1):
    """Create a TrainerCallback for per-epoch assessment with early stopping.

    Import is deferred so this module can be imported without transformers installed.
    """
    from transformers import TrainerCallback

    class EpochAssessCallback(TrainerCallback):
        def __init__(self):
            self.best_accuracy = -1
            self.best_fp = float("inf")
            self.best_epoch = 0
            self.patience_counter = 0
            self.epoch_results = []
            self.inline_failed = False

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = int(state.epoch)
            max_e = int(args.num_train_epochs)

            # Save checkpoint
            ckpt_dir = Path(output_dir) / "checkpoints" / f"epoch-{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))

            # Attempt inline assessment
            try:
                was_training = model.training
                model.eval()  # standard PyTorch — disables dropout only
                metrics = assess_task(model, tokenizer, task_config, source_dir)
                if was_training:
                    model.train()
            except Exception as e:
                print(f"  Warning: inline assessment failed at epoch {epoch}: {e}")
                self.inline_failed = True
                metrics = {"binary_accuracy": 0, "fp": 999, "fn": 999, "parse_fail": 0}

            acc = metrics["binary_accuracy"]
            fp = metrics["fp"]
            fn = metrics.get("fn", 0)

            # Get training loss from log history
            loss = "?"
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        loss = round(entry["loss"], 4)
                        break

            # Check improvement: accuracy first, then lower FP as tiebreak
            improved = False
            if acc > self.best_accuracy or (
                acc == self.best_accuracy and fp < self.best_fp
            ):
                self.best_accuracy = acc
                self.best_fp = fp
                self.best_epoch = epoch
                self.patience_counter = 0
                improved = True

                # Copy to best_model dir
                best_dir = Path(output_dir) / "best_model"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(str(ckpt_dir), str(best_dir))
            else:
                self.patience_counter += 1

            # Save epoch assessment results
            result_path = Path(output_dir) / f"assess_epoch_{epoch}.json"
            result_data = {
                "epoch": epoch,
                "accuracy": acc,
                "fp": fp,
                "fn": fn,
                "loss": loss,
                "improved": improved,
                "best_epoch": self.best_epoch,
            }
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

            self.epoch_results.append(result_data)

            # Mini-report
            star = " ★" if improved else ""
            key = task_config["binary_key"]
            print(
                f"  Epoch {epoch}/{max_e}: loss={loss} | "
                f"{key}_acc={acc:.0%} | FP={fp} | FN={fn} | "
                f"best=epoch_{self.best_epoch}{star}"
            )

            # Early stopping
            if self.patience_counter >= patience:
                print(f"  Early stopping: no improvement for {patience} epoch(s)")
                control.should_training_stop = True

            return control

    return EpochAssessCallback()


# =============================================================================
# Milestone Reporting
# =============================================================================

def print_milestone(task_name, phase, metrics, elapsed_s=None):
    """Print a formatted milestone report box."""
    width = 58
    header = f" MILESTONE: {task_name} / {phase} "
    print()
    print("+" + "=" * width + "+")
    print(f"|{header:^{width}}|")
    print("+" + "=" * width + "+")

    for key, val in metrics.items():
        if key == "results":
            continue  # skip per-image details
        label = key.replace("_", " ").capitalize()
        if isinstance(val, float) and 0 <= val <= 1:
            line = f"  {label}: {val:.0%}"
        else:
            line = f"  {label}: {val}"
        print(f"| {line:<{width - 1}}|")

    if elapsed_s is not None:
        mins = int(elapsed_s // 60)
        secs = int(elapsed_s % 60)
        line = f"  Time: {mins}m {secs:02d}s"
        print(f"| {line:<{width - 1}}|")

    vram = vram_stats()
    line = f"  VRAM: {vram['current_gb']} GB (peak {vram['peak_gb']} GB)"
    print(f"| {line:<{width - 1}}|")

    print("+" + "=" * width + "+")
    print()


def print_summary(all_results):
    """Print cross-task comparison summary table."""
    print()
    print("+" + "=" * 68 + "+")
    print(f"|{'PIPELINE SUMMARY':^68}|")
    print("+" + "-" * 14 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 +
          "+" + "-" * 10 + "+" + "-" * 12 + "+")
    print(f"| {'Task':<12} | {'Baseline':>8} | {'After FT':>8} | {'Delta':>6} "
          f"| {'FP':>8} | {'Best Ep':>10} |")
    print("+" + "-" * 14 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 8 +
          "+" + "-" * 10 + "+" + "-" * 12 + "+")

    for task_name, result in all_results.items():
        short = task_name.split("_", 1)[1] if "_" in task_name else task_name
        base = result.get("baseline_accuracy", 0)
        final = result.get("final_accuracy", 0)
        delta = final - base
        fp_before = result.get("baseline_fp", "?")
        fp_after = result.get("final_fp", "?")
        best_ep = result.get("best_epoch", "?")

        print(
            f"| {short:<12} | {base:>7.0%} | {final:>7.0%} | "
            f"{delta:>+5.0%} | {fp_before}->{fp_after!s:>4} | {best_ep!s:>10} |"
        )

    print("+" + "=" * 68 + "+")
    print()


def print_bakeoff(task_a, metrics_a, task_b, metrics_b, winner, loser):
    """Print bake-off comparison between two similar tasks."""
    print()
    print("+" + "=" * 58 + "+")
    print(f"|{'BAKE-OFF: ' + task_a + ' vs ' + task_b:^58}|")
    print("+" + "=" * 58 + "+")
    short_a = task_a.split("_", 1)[1]
    short_b = task_b.split("_", 1)[1]
    acc_a = metrics_a["binary_accuracy"]
    acc_b = metrics_b["binary_accuracy"]
    fp_a = metrics_a["fp"]
    fp_b = metrics_b["fp"]
    print(f"|  {short_a:<12} baseline: {acc_a:>5.0%} acc, {fp_a} FP{' ':>22}|")
    print(f"|  {short_b:<12} baseline: {acc_b:>5.0%} acc, {fp_b} FP{' ':>22}|")
    print(f"|{' ':>58}|")
    win_short = winner.split("_", 1)[1]
    lose_short = loser.split("_", 1)[1]
    print(f"|  -> PRIMARY: {win_short} (lower baseline = more room to improve){' ':>3}|")
    print(f"|  -> OPTIONAL: {lose_short} (runs only if time permits){' ':>8}|")
    print("+" + "=" * 58 + "+")
    print()


# =============================================================================
# Pipeline State
# =============================================================================

def save_pipeline_state(output_dir, state):
    """Save pipeline state for resumability."""
    path = Path(output_dir) / "pipeline_state.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_pipeline_state(output_dir):
    """Load pipeline state, or return empty state if none exists."""
    path = Path(output_dir) / "pipeline_state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_phases": [], "completed_tasks": [], "task_order": []}
