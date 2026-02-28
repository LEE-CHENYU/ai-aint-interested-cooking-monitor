# Agentic Cooking Workflow

## What It Does

A user says a dish name — like "mapo tofu" or "pasta" — and the agent takes over as a step-by-step cooking guide. It generates a recipe, breaks it into ordered steps, and reveals **one step at a time**. It watches the kitchen through a camera (or synthetic images for demo) to detect when each step is done, then announces the next one via console text and spoken audio.

The user never sees the full recipe. They just hear: "Step 1: Dice the tofu into 1-inch cubes" — and when the system sees diced tofu on the cutting board, it moves on.

## How It Works

There are four phases in every cooking session:

### Phase 1: Recipe Generation

The user provides a dish name. The agent checks if a cached recipe exists on disk (`configs/recipes/{dish}.yaml`). If not, it calls Gemma 3 (text-only, one-shot) to generate a structured recipe and saves it for future runs.

Each recipe step has one of three **completion types**:

| Type | When Used | Example |
|------|-----------|---------|
| **vlm** | The camera can see when it's done | "Boil water" — VLM detects rolling boil |
| **timer** | It takes a fixed amount of time | "Simmer for 5 minutes" — timer counts down |
| **user_confirm** | The camera can't detect it | "Mince garlic" — user says "done" |

Steps also have **dependencies** (step 3 can't start until step 1 is done) and optional **parallel groups** (two timer-based steps can run simultaneously in the background).

### Phase 2: Agent Loop

The loop is **inference-driven** — it runs as fast as the VLM can process images (conservatively 3-10 seconds per frame after LoRA merge). Each cycle:

1. **Capture** — load the next image from a scripted sequence
2. **Infer** — send image to merged Gemma 3 VLM with a step-aware prompt
3. **Smooth** — add result to a 3-frame buffer, require 2/3 agreement before acting
4. **Check transitions** — did the VLM detect the expected signal? Did a timer expire? Any safety hazard?
5. **Advance** — if the current step is done, activate the next eligible step and announce it

### Phase 3: Step Progression

The step transition engine is **rule-based** (no LLM call). It simply compares:

- For `vlm` steps: does the VLM output match the expected signal?
- For `timer` steps: has the timer expired?
- For `user_confirm` steps: did the user press Enter / say "done"?

When a step completes, the engine finds the next step whose dependencies are all satisfied, activates it, and announces it to the user.

**Parallel handling:** The user only sees one focused step at a time. Timer-based steps run silently in the background and alert when they need attention.

### Phase 4: Output

Two output channels:

- **Console** — printed text with priority prefixes (`>>`, `!!`, `** WARNING **`)
- **TTS audio** — spoken via macOS `say` command, with faster speech rate for critical alerts

Safety alerts (boil-over, smoke) can interrupt at any point, regardless of the current step.

## What the User Experiences

```
🔊 "Let's make mapo tofu! Step 1: Dice the tofu into 1-inch cubes"
   ... camera watches ...
   ... detects diced tofu on cutting board ...

🔊 "Step 2: Mince garlic, ginger, and scallions. Press Enter when done."
   ... user presses Enter ...

🔊 "Step 3: Heat oil in wok until shimmering"
   ... camera watches ...
   ... detects shimmering oil ...

🔊 "Step 4: Add doubanjiang and cook for 30 seconds. Timer started."
   ... 30 seconds ...
   ⏰ "Timer done! Step 5: Add tofu and broth, simmer for 5 minutes. Timer started."
   ... 5 minutes (background) ...

   ⏰ "Simmer complete! Step 6: Add cornstarch slurry and stir until thickened"
   ... camera watches ...
   ... detects thick glossy sauce ...

🔊 "Mapo tofu is done! Enjoy your meal."
```

## Model Capabilities & Limitations

Based on the VLM fine-tuning experiment (36 synthetic images, Gemma 3 4B):

**Works well:**
- Water-based boil-over detection — 100% for pasta, ramen, soup, rice
- State transitions for water-based dishes (cold → simmer → boil) — ~80%
- Structured JSON output — reliable after fine-tuning

**Doesn't work yet:**
- Burnt food detection — misses 50% of cases
- Thick-liquid boil-overs (curry, mapo tofu sauce) — misses these
- "Done" vs "cold" distinction — model confuses them
- Real-time speed under LoRA — 17-55s per image (must merge weights first)

The three completion types (vlm / timer / user_confirm) are designed around these limitations. Steps where the model is unreliable use timers or user confirmation instead.

## Architecture Diagram

```
User: "mapo tofu"
       │
       ▼
┌──────────────┐     ┌─────────────────────┐
│Recipe Planner│────→│ configs/recipes/     │
│ (LLM, once)  │     │ mapo_tofu.yaml      │
└──────────────┘     └─────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       AGENT LOOP              │
              │                               │
              │  Image ──→ VLM ──→ State      │
              │  Sequencer  (Gemma)  Manager  │
              │                    │          │
              │              Step Transition  │
              │              Engine (rules)   │
              │                    │          │
              │              Console + TTS    │
              └───────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| One step at a time | Keeps the user focused, less overwhelming |
| LLM used once (recipe gen), not per-cycle | Keeps latency at 3-10s instead of 6-20s |
| Rule-based step transitions | Reliable, fast, easy to debug |
| Three completion types | Plays to model strengths, covers weaknesses |
| Temporal smoothing (2/3 over 3 frames) | Prevents false triggers from noisy VLM output |
| Scripted image sequences for demo | Reproducible, reliable demo without real kitchen |
| Console + TTS output | High demo impact without frontend development |

## Files

| File | Purpose |
|------|---------|
| `src/agent/schemas.py` | RecipeStep and Recipe Pydantic models |
| `src/agent/recipe_loader.py` | Load recipes from YAML |
| `src/agent/step_engine.py` | Rule-based step transition engine |
| `src/agent/tts.py` | Text-to-speech via macOS `say` |
| `src/agent/tools.py` | Action executor (timers, TTS, UI) |
| `src/perception/detector.py` | Gemma 3 VLM inference (+ mock mode) |
| `src/perception/image_sequencer.py` | Sequential image file loader |
| `src/perception/temporal_smoother.py` | 3-frame majority voting filter |
| `src/world_state/state.py` | Kitchen state aggregation |
| `src/world_state/timer_engine.py` | Timer management |
| `scripts/run_demo.py` | End-to-end demo runner |
| `configs/recipes/*.yaml` | Cached recipe definitions |
| `data/demo_sequences/` | Scripted demo image sequences |
