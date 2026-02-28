# Agentic Cooking Workflow Design

## Overview

A recipe-driven, step-by-step cooking assistant that guides users through preparing a dish. The user provides a dish name; the agent generates a recipe, breaks it into ordered steps, and reveals one step at a time — using vision (VLM), timers, and user confirmation to detect when each step completes before advancing.

## Constraints (from VLM Fine-Tuning Experiment)

| Constraint | Detail |
|---|---|
| **Inference latency** | 3-10s per frame (conservative estimate after LoRA merge) |
| **VLM strengths** | Water-based state detection (cold/simmer/boil/boil-over), structured JSON output |
| **VLM weaknesses** | Burnt food (50% miss), thick-liquid boil-over, "done" vs "cold" confusion, curry bias |
| **Training data** | 36 synthetic images, 6 dishes x 6 states |
| **Safety accuracy** | 81% overall, 100% for water-based boil-over |

These constraints shape every design decision below.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                         │
│  Input: "I want to cook mapo tofu"                              │
│  Output: One step at a time via Console + TTS                   │
└──────────┬──────────────────────────────────────┬───────────────┘
           │                                      ▲
           ▼                                      │
┌──────────────────┐                    ┌─────────────────────┐
│  Recipe Planner   │                    │   Action Executor    │
│  (LLM one-shot)  │                    │  • Console print     │
│  dish name →      │                    │  • TTS speak         │
│  structured YAML  │                    │  • Timer set/check   │
│  + cache to disk  │                    └─────────┬───────────┘
└──────────┬───────┘                              ▲
           │                                      │
           ▼                                      │
┌──────────────────────────────────────────────────────────────┐
│                      AGENT LOOP (inference-driven)            │
│                                                               │
│  ┌────────────┐    ┌──────────┐    ┌────────────────────┐    │
│  │ Image Feed  │───→│ Gemma 3  │───→│ State Manager      │    │
│  │ (synthetic  │    │ VLM      │    │ • current step     │    │
│  │  sequence)  │    │ (merged) │    │ • timer engine     │    │
│  └────────────┘    └──────────┘    │ • detection history │    │
│                                     └────────┬───────────┘    │
│                                              │                │
│                                              ▼                │
│                                     ┌────────────────────┐    │
│                                     │ Step Transition     │    │
│                                     │ Engine (rule-based) │    │
│                                     │ • VLM match?        │    │
│                                     │ • Timer expired?    │    │
│                                     │ • Safety alert?     │    │
│                                     └────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Design principle:** LLM is used once upfront (recipe generation) and for vision (VLM per frame). Step transitions are rule-based — no second LLM call per cycle. This keeps latency at 3-10s per cycle instead of 6-20s.

## Recipe Planner

**Flow:**
1. Check `configs/recipes/{dish_name}.yaml` — if cached, load it
2. If not cached, call Gemma 3 (text-only) to generate recipe in structured format
3. Save to `configs/recipes/{dish_name}.yaml` for future runs

**Recipe schema:**

```yaml
dish: mapo_tofu
servings: 2
estimated_time_minutes: 25

steps:
  - id: 1
    instruction: "Dice the tofu into 1-inch cubes"
    completion_type: vlm           # vlm | timer | user_confirm
    vlm_signal: "diced tofu on cutting board"
    depends_on: []
    parallel_group: null

  - id: 2
    instruction: "Mince garlic, ginger, and scallions"
    completion_type: user_confirm
    depends_on: []
    parallel_group: null

  - id: 3
    instruction: "Heat oil in wok until shimmering"
    completion_type: vlm
    vlm_signal: "oil shimmering in hot wok"
    depends_on: [1, 2]
    parallel_group: null

  - id: 4
    instruction: "Add doubanjiang and cook for 30 seconds"
    completion_type: timer
    timer_seconds: 30
    depends_on: [3]
    parallel_group: A

  - id: 5
    instruction: "Add tofu and broth, simmer for 5 minutes"
    completion_type: timer
    timer_seconds: 300
    depends_on: [4]
    parallel_group: A

  - id: 6
    instruction: "Add cornstarch slurry and stir until thickened"
    completion_type: vlm
    vlm_signal: "thick glossy sauce coating tofu"
    depends_on: [5]
    parallel_group: null
```

**Three completion types** align with model capabilities:
- **`vlm`**: VLM detects visual state change (used where model is reliable)
- **`timer`**: Step completes after N seconds (used for timed cooking phases)
- **`user_confirm`**: User taps/says "done" (used where VLM can't detect — e.g., mincing, seasoning)

## Agent Loop

Inference-driven: the loop speed matches VLM inference time (3-10s).

```
1. CAPTURE: Load next synthetic image from scripted sequence
      ↓
2. INFER: Send image + step-aware prompt to merged Gemma 3 VLM
      ↓
3. SMOOTH: Add result to temporal buffer (last 3 frames)
   Require 2/3 agreement before acting
      ↓
4. CHECK TRANSITIONS:
   a) Active VLM step → signal detected? → mark done
   b) Active timers → any expired? → mark done
   c) Safety signals → boil_over/smoke? → alert
      ↓
5. ADVANCE: If current step done:
   - Find next step (all dependencies satisfied)
   - If timer step → start timer in background
   - If parallel step available → start it silently
   - Announce new step via Console + TTS
      ↓
6. LOOP: Go to step 1
```

**Step state machine:**

```
pending → active → done
```

Steps remain `pending` until all `depends_on` steps are `done`.

**Parallel step handling:** One active step shown to user. Steps with timers run in the background and alert when they expire or need attention. The user stays focused on a single task.

## User Experience

```
🔊 "Step 1: Dice the tofu into 1-inch cubes"
   ... user dices tofu ...
   ... VLM detects diced tofu on cutting board (2/3 agreement) ...
🔊 "Great! Step 2: Mince garlic, ginger, and scallions.
     Say 'done' when ready."
   ... user says done ...
🔊 "Step 3: Heat oil in wok until shimmering"
   ... VLM detects shimmering oil ...
🔊 "Step 4: Add doubanjiang and cook for 30 seconds.
     Timer started."
   ... 30s timer runs ...
   ⏰ "Timer done! Step 5: Add tofu and broth,
      simmer for 5 minutes. Timer started."
   ... 5 min timer runs in background ...
   ⏰ "Simmer complete! Step 6: Add cornstarch slurry
      and stir until thickened"
   ... VLM detects thick glossy sauce ...
🔊 "Mapo tofu is done! Enjoy your meal."
```

Safety alerts can interrupt at any point:
```
⚠️ "WARNING: Boil-over detected! Reduce heat immediately."
```

## Demo Image Sequences

Scripted image sequences simulate a cooking session:

```
data/demo_sequences/mapo_tofu/
  ├── 001_empty_counter.png
  ├── 002_tofu_block_on_board.png
  ├── 003_diced_tofu_on_board.png         # Step 1 complete
  ├── 004_minced_aromatics.png            # Step 2 complete
  ├── 005_oil_in_cold_wok.png
  ├── 006_oil_shimmering_wok.png          # Step 3 complete
  ├── 007_doubanjiang_in_wok.png          # Step 4 (timer)
  ├── 008_tofu_in_broth_simmering.png     # Step 5 (timer)
  ├── 009_adding_cornstarch.png
  ├── 010_thick_glossy_mapo_tofu.png      # Step 6 complete
  └── 011_plated_mapo_tofu.png            # Done
```

Generate using the same Gemini-based approach from the fine-tuning experiment. Each transition image should have 2-3 variations for VLM robustness.

## Component Mapping to Existing Code

| Component | File | Status | Work Needed |
|---|---|---|---|
| Recipe Planner | `src/agent/agent_loop.py` | Stub | Add LLM recipe generation + YAML caching |
| Recipe Schema | `src/agent/schemas.py` | Partial | Add `RecipeStep` with completion_type, vlm_signal, timer_seconds, depends_on, parallel_group |
| Image Sequencer | `src/perception/camera.py` | Stub | Replace live camera with sequential image file loader |
| VLM Detector | `src/perception/detector.py` | Stub | Load merged Gemma 3 VLM, run inference, return JSON |
| Temporal Smoother | `src/perception/temporal_smoother.py` | Done | Adjust window to 3 |
| State Manager | `src/world_state/state.py` | Partial | Add step tracking (active, pending, completed) |
| Timer Engine | `src/world_state/timer_engine.py` | Done | Works as-is |
| Step Transition Engine | `src/agent/step_engine.py` | **New** | Rule-based: check VLM match, timer expiry, advance steps |
| Action Executor | `src/agent/tools.py` | Partial | Add TTS via pyttsx3 or system `say`, wire console output |
| Demo Runner | `scripts/run_demo.py` | Stub | Wire all components, load recipe, run loop |
| Demo Sequences | `data/demo_sequences/` | **New** | Generate scripted image sequences per recipe |

**New files:** 2 — `src/agent/step_engine.py` and `data/demo_sequences/` directory.

**Key dependency:** Merged Gemma 3 VLM model must be exported first via `infra/deployment/export_model.py`.

## Design Decisions Summary

| Aspect | Decision | Rationale |
|---|---|---|
| Primary function | Recipe-driven step-by-step guide | Core agent value prop |
| Recipe source | LLM-generated, cached as YAML | Flexible + reproducible |
| User sees | One step at a time | Progressive disclosure, less overwhelming |
| Step completion | VLM / timer / user confirm (hybrid) | Plays to model strengths, covers weaknesses |
| Vision model | Merged Gemma 3 VLM | Single model, no LoRA overhead at inference |
| Agent loop | Inference-driven | Loop speed = inference speed, no wasted cycles |
| Temporal smoothing | 2/3 over 3-frame buffer | Prevents false triggers, stays responsive |
| Parallel steps | Background timers, one active step shown | Keeps user focused |
| Output | Console + TTS audio | High impact, no frontend work needed |
| Safety | Boil-over/smoke as secondary alerts | Interrupt during any step |
| Input | Scripted synthetic image sequences | Reliable, reproducible demo |
| Demo dish | Water-based (pasta/ramen) primary, mapo tofu stretch | Plays to 100% boil-over detection rate |
