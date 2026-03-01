# Automated Multi-Task VLM Fine-Tuning Pipeline

## Context

We have 4 kitchen safety detection tasks ready with 36 images each (generated from Gemini). The preliminary dish×state experiment showed fine-tuning works (75%→81% safety, 25%→61% state accuracy). Now we need an automated pipeline that runs baseline eval → generate variations → fine-tune with checkpoints → final eval → merge LoRA for each task. Single GPU (96GB VRAM) — true GPU parallelism is unsafe (CUDA context collisions with Unsloth), but **phase batching** saves ~33% time by loading each model once for all tasks.

## Tasks

| Task | Detection | Rows (6) | Columns (6) | Binary Key | Images |
|------|-----------|----------|-------------|------------|--------|
| task1_boilover | Pot overflow severity | vessels | severity levels | `safe` | 36 ✓ |
| task2_smoke | Smoke/fire detection | scenarios | smoke levels | `safe` | 36 ✓ |
| task3_person | Person present/absent | activities | presence levels | `present` | 36 ✓ |
| task6_boiling | Water boiling state | vessels | boil states | `boiling` | 36 ✓ |

Each: 3 safe columns + 3 unsafe columns. All images at `data/synthetic/matrix/task{N}_{name}/matrix_images/`

## Parallelism Analysis

### Why True GPU Parallelism Won't Work (Single GPU)

| Approach | VRAM Feasible? | Safe? | Verdict |
|----------|---------------|-------|---------|
| VLM train + SDXL gen concurrently | Yes (~20 GB / 96 GB) | **No** — Unsloth's gradient checkpointing uses shared CUDA contexts; PyTorch memory allocator fragments under concurrent processes | **REJECTED** |
| Two VLM training runs concurrently | Yes (~17 GB) | **No** — CUDA context collision, Unsloth not multi-process safe | **REJECTED** |
| VLM inference + SDXL gen concurrently | Yes (~18 GB) | Risky — could crash either process | **REJECTED** |

### What DOES Work: Phase Batching

Instead of completing task1 end-to-end before task2, **batch all same-phase work together** to minimize model load/unload cycles:

| Strategy | Model Loads | Est. Total Time | Risk |
|----------|-------------|-----------------|------|
| **Per-task interleaving** (old plan) | 20 loads | ~180 min | Low — get task1 results early |
| **Phase batching** (new plan) | 10 loads | ~120 min | Low — all baselines known upfront |
| **Phase batching + early stopping** | 10 loads | ~100 min | Low — patience=2 prevents wasted epochs |

**Phase batching saves ~60 min (33%)** by:
1. Loading VLM once for ALL 4 baselines (saves 3 loads × ~30s = 1.5 min + VRAM flush overhead)
2. Loading SDXL once for ALL 4 variation sets (saves 3 loads × ~30s = 1.5 min)
3. Combining train→eval→merge per task (saves 4 extra model loads for separate eval/merge phases)
4. Early stopping cuts wasted training epochs (~15-20 min across 4 tasks)

## Hard Constraint: 2-Hour Time Budget

### Time Budget Breakdown

| Phase | Work | Est. Time | Cumulative |
|-------|------|-----------|------------|
| **A** | Batch all baselines (4×36 = 144 inferences) | 15 min | 0:15 |
| **B** | Batch all variations (4×72 = 288 images @ 0.7s) | 5 min | 0:20 |
| **C1** | Train+eval+merge task1_boilover | 20 min | 0:40 |
| **C2** | Train+eval+merge task2_smoke | 20 min | 1:00 |
| **C3** | Train+eval+merge task3_person | 20 min | 1:20 |
| **C4** | Train+eval+merge task6_boiling | 20 min | 1:40 |
| | Summary + buffer | 20 min | **2:00** |

### Adaptive Time Management

The pipeline tracks elapsed time and adjusts automatically:

```python
# Time checkpoints in run_pipeline.py
BUDGET_MINUTES = 120
CHECKPOINTS = {
    "after_baselines":  {"deadline": 20,  "action": None},
    "after_generation": {"deadline": 30,  "action": "reduce_variations"},
    "after_task1":      {"deadline": 55,  "action": "reduce_epochs"},
    "after_task2":      {"deadline": 80,  "action": "reduce_epochs"},
    "after_task3":      {"deadline": 105, "action": "skip_gguf"},
}
```

**If running behind schedule:**
1. **Reduce variations** from 2 to 1 for remaining tasks (saves ~2 min/task generation + faster training)
2. **Reduce max epochs** from 3 to 2 for remaining tasks (saves ~5 min/task)
3. **Skip GGUF export** (saves ~2 min/task, can do later)
4. **Skip last task** if > 100 min elapsed after task3 (still get 3/4 models)

### Default Settings (tuned for 2-hour budget)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Variations | **2** (not 3) | 72 extra images per task is enough; saves generation + training time |
| Epochs | **3** (not 5) | Previous experiment: loss saturated by epoch 3 (0.05). 5 epochs is overkill |
| Patience | **1** | Stop after 1 epoch with no improvement. Aggressive but time-constrained |
| Batch size | 1 × 4 accum = 4 effective | Same as before |
| LoRA rank | 32 | Same as before |

## Architecture: Phase-Batched Orchestrator

Create `infra/pipeline/run_pipeline.py` — one entry point that batches same-phase work across tasks.

### Files to Create

```
infra/pipeline/
├── run_pipeline.py           # Orchestrator — phase-batched execution
├── task_configs.py           # Per-task config registry (labels, prompts, thresholds)
└── utils.py                  # VRAM cleanup, logging, prompt parsing, eval helpers
```

### Execution Plan (~100 min target, 120 min hard limit)

```
PHASE A: BATCH ALL BASELINES (~15 min, 1 VLM load)
├── Load Gemma 3 4B (unsloth, 4-bit) — ~7 GB VRAM
├── For each task [task1, task2, task3, task6]:
│   ├── Run inference on 36 source images (~1s/img × 36 = ~36s + overhead)
│   ├── Compute binary accuracy, confusion matrix
│   ├── Save baseline_results.json
│   └── >>> MILESTONE REPORT: baseline metrics <<<
├── >>> MILESTONE REPORT: cross-task baseline comparison <<<
├── ⏱ TIME CHECK: if > 20 min, reduce variations to 1
└── Unload model, clear VRAM

PHASE B: BATCH ALL VARIATIONS (~5 min, 1 SDXL load, 288 images)
├── Load SDXL img2img pipeline — ~11 GB VRAM
├── For each task [task1, task2, task3, task6]:
│   ├── Parse task-specific prompts.txt
│   ├── Generate 2 variations per image (72 per task, 288 total @ 0.7s/img)
│   ├── Save to variations/ + labels.json
│   └── >>> MILESTONE REPORT: generation count <<<
├── >>> MILESTONE REPORT: total generated + VRAM peak <<<
├── ⏱ TIME CHECK: if > 30 min, reduce epochs to 2
└── Unload SDXL, clear VRAM

PHASE C: SEQUENTIAL TRAIN + EVAL + MERGE (4 × ~20 min = ~80 min)
├── For each task [task1, task2, task3, task6]:
│   ├── ⏱ TIME CHECK: if remaining < 25 min, skip this task
│   ├── Load Gemma 3 4B + fresh LoRA (rank=32) — ~8.5 GB VRAM
│   ├── Build dataset: 36 originals + 72 variations = 108 samples
│   ├── Oversample unsafe 3x → ~162 balanced
│   ├── Train 3 epochs with early stopping (patience=1):
│   │   ├── SafetyEvalCallback: eval on 36 source images each epoch
│   │   ├── Track best model by binary accuracy (tie-break: lower FP)
│   │   ├── Stop early if no improvement for 1 epoch
│   │   └── >>> MINI REPORT per epoch: loss, accuracy, FP, best_epoch <<<
│   ├── >>> MILESTONE REPORT: training summary + best epoch <<<
│   ├── Final eval on 36 source images with best checkpoint
│   ├── Save final_eval_results.json
│   ├── >>> MILESTONE REPORT: before→after comparison table <<<
│   ├── model.save_pretrained_merged() → merged_model/ (16-bit)
│   ├── model.save_pretrained_gguf() → gguf/ (q4_k_m) [skip if time tight]
│   ├── >>> MILESTONE REPORT: export paths + sizes <<<
│   └── Unload model, clear VRAM
└── >>> MILESTONE REPORT: CROSS-TASK SUMMARY TABLE <<<
```

**Target: ~100 min | Hard limit: 120 min | Minimum deliverable: 3/4 tasks**

### Output Structure on GCP

```
/tmp/pipeline_output/
├── task1_boilover/
│   ├── baseline_results.json
│   ├── variations/{row}_{col}_v{1,2,3}.png + labels.json
│   ├── checkpoints/checkpoint-{step}/ + best_model/
│   ├── eval_epoch_{1..5}.json
│   ├── final_eval_results.json
│   ├── merged_model/          # LoRA merged into base weights
│   └── gguf/                  # q4_k_m quantized
├── task2_smoke/  (same)
├── task3_person/ (same)
├── task6_boiling/ (same)
├── pipeline_state.json        # Resumability tracking
└── pipeline_summary.json      # Cross-task comparison table
```

## Implementation Details

### Step 1: Create `task_configs.py`

Per-task config dictionary with:
- `rows`, `cols`: matrix dimensions
- `safe_cols`, `unsafe_cols`: which columns are safe/unsafe
- `col_labels`: ground truth per column (state name, binary value, reason)
- `row_labels`: human-readable row names for JSON responses
- `system_prompt`: task-specific VLM prompt
- `binary_key`: `"safe"` / `"present"` / `"boiling"` — which JSON field to evaluate
- `source_images`, `prompts_file`: paths to data
- Training hyperparams: `epochs=3`, `lr=2e-4`, `lora_rank=32`, `finetune_vision` (True for smoke/person, False for boilover/boiling), `oversample_unsafe=3`

### Step 2: Create `utils.py`

- `clear_vram()`: gc.collect + torch.cuda.empty_cache + reset_peak_memory
- `parse_prompts(file)`: parse `{row}/{col} | {prompt}` format from prompts.txt
- `build_training_samples(config, source_dir, variations_dir)`: build dataset from source + variations with oversampling
- `evaluate_task(model, tokenizer, config, image_dir)`: generic eval using config's binary_key
- `print_milestone(task, phase, metrics)`: formatted milestone report to stdout
- `SafetyEvalCallback(TrainerCallback)`: per-epoch eval + best model tracking

### Step 3: Create `run_pipeline.py`

Phase-batched orchestrator with:
- `--tasks` flag for subset (default: all 4)
- `--skip-baselines` to reuse existing baseline results
- `--skip-generation` to reuse existing variations
- `--resume` to continue from last completed phase (reads `pipeline_state.json`)
- `--output-dir` (default: `/tmp/pipeline_output`)
- `--variations` count (default: 2 — reduced from 3 to fit 2-hour budget)
- `--patience` early stopping patience (default: 1 — aggressive for time budget)
- `--epochs` max training epochs (default: 3 — loss saturates by epoch 3)
- `--time-budget` hard time limit in minutes (default: 120)
- `--source-dir` path to matrix images (default: `/tmp/matrix`)

Pipeline state tracking: writes `pipeline_state.json` after each phase/task completes. On `--resume`, skips completed work. Adaptive time management automatically reduces scope if running behind.

### Step 4: Milestone Reporting

**Every phase boundary** prints a structured report:

```
╔══════════════════════════════════════════════════════════╗
║ MILESTONE: task1_boilover / BASELINE EVAL               ║
╠══════════════════════════════════════════════════════════╣
║ Binary (safe) accuracy:  22/36 (61%)                    ║
║ State accuracy:          10/36 (28%)                    ║
║ FP (danger→safe):        8  << DANGEROUS                ║
║ FN (safe→danger):        6                              ║
║ VRAM peak:               6.2 GB                         ║
║ Time:                    4m 32s                          ║
╚══════════════════════════════════════════════════════════╝
```

**Per-epoch mini-report** during training:
```
  Epoch 1/5: loss=2.41 | safe_acc=67% | state_acc=42% | FP=6 | best=epoch_1
  Epoch 2/5: loss=0.34 | safe_acc=81% | state_acc=58% | FP=3 | best=epoch_2 ★
  Epoch 3/5: loss=0.08 | safe_acc=78% | state_acc=61% | FP=4 | best=epoch_2
```

**Cross-task summary** at the end:
```
╔════════════════════════════════════════════════════════════════════╗
║                    PIPELINE SUMMARY                               ║
╠══════════════╦══════════╦══════════╦════════╦════════╦════════════╣
║ Task         ║ Baseline ║ After FT ║ Delta  ║ FP     ║ Best Epoch ║
╠══════════════╬══════════╬══════════╬════════╬════════╬════════════╣
║ boilover     ║  61%     ║  89%     ║ +28%   ║ 9→2   ║ 2          ║
║ smoke        ║  55%     ║  83%     ║ +28%   ║ 11→3  ║ 3          ║
║ person       ║  50%     ║  86%     ║ +36%   ║ 10→1  ║ 2          ║
║ boiling      ║  72%     ║  92%     ║ +20%   ║ 5→1   ║ 3          ║
╚══════════════╩══════════╩══════════╩════════╩════════╩════════════╝
```

### Step 5: Upload & Run on GCP

```bash
# Upload pipeline + data
scp -r infra/pipeline/ hackathon@35.238.6.1:/tmp/pipeline/
scp -r data/synthetic/matrix/ hackathon@35.238.6.1:/tmp/matrix/

# Run all 4 tasks (~100 min, auto-adjusts if behind)
ssh hackathon@35.238.6.1 'python3 /tmp/pipeline/run_pipeline.py --output-dir /tmp/pipeline_output --time-budget 120'

# Resume after disconnect
ssh hackathon@35.238.6.1 'python3 /tmp/pipeline/run_pipeline.py --resume'
```

### Step 6: Download Results

```bash
# Download merged models + results
for task in task1_boilover task2_smoke task3_person task6_boiling; do
    scp -r hackathon@35.238.6.1:/tmp/pipeline_output/$task/merged_model/ models/$task/
    scp hackathon@35.238.6.1:/tmp/pipeline_output/$task/*results.json results/$task/
done
scp hackathon@35.238.6.1:/tmp/pipeline_output/pipeline_summary.json results/
```

## Checkpointing Strategy

| What | When | Why |
|------|------|-----|
| LoRA adapter | Every epoch end | Crash recovery + epoch selection |
| Best model copy | When binary accuracy improves | Avoid overfitting regression |
| Pipeline state | After each phase completes | Resume after disconnect/crash |
| Eval results | Every epoch + final | Track learning curve |
| Keep limit | All 3 epochs | Small LoRA files (~60MB each), disk is cheap |

**Best model selection**: Binary accuracy on source images (not variations). Tie-break: lower FP count wins.

## Key Differences from Previous Experiment

| Aspect | Previous (dish×state) | This Pipeline |
|--------|----------------------|---------------|
| Tasks | 1 combined task | 4 specialized tasks |
| Images | 36 (Gemini grid) | 36 per task (SDXL matrix) |
| Variations | None in training | 108 per task (3x img2img) |
| Execution | Single-task sequential | **Phase-batched across tasks** |
| Checkpoints | None saved | Per-epoch + best model |
| Eval | Only at end | Per-epoch during training |
| Early stopping | No | **patience=2** (stop if no improvement for 2 epochs) |
| Export | LoRA only | LoRA + merged 16-bit + GGUF |
| Binary key | `safe` only | Task-specific (`safe`/`present`/`boiling`) |
| Vision FT | Off | On for smoke/person tasks |
| Resumable | No | Yes (pipeline_state.json) |
| Est. time | N/A | **~100 min** (2-hour budget with adaptive reduction) |

## Verification

1. **Run full pipeline**: All 4 tasks in one shot — no time for test runs with 2-hour budget
2. **Monitor milestone reports**: Each phase prints structured output; watch for time warnings
3. **Verify merged model works**: Load merged model, run inference, confirm no LoRA loading needed
4. **Compare baseline vs final**: Final accuracy should exceed baseline for all completed tasks
5. **Download and spot-check**: Pull merged models locally, verify file sizes (~8GB for 16-bit, ~2GB for GGUF)
6. **Minimum success**: At least 3/4 tasks complete with merged models + eval results
