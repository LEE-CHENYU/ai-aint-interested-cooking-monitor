# Phone UI & Demo Slide Design

## Overview

Two deliverables:
1. A responsive phone web UI that shows the current cooking step, timer, and done button — styled like the Overcooked game
2. A 5-minute hackathon demo slide script (6 slides)

## Phone UI

### Tech Stack

Single self-contained HTML file (`src/ui/index.html`) with inline CSS + JS. Served by the existing Python `UIServer`. Connects via WebSocket for real-time updates. Zero extra dependencies.

### Layout

Three zones stacked vertically:

```
┌──────────────────────────────┐
│  🍳 AI Kitchen Monitor       │  ← Header bar (warm orange)
│  ══════════════════════════  │
│                              │
│  ┌──────────────────────┐   │
│  │  STEP 3 of 6         │   │  ← Step progress chip
│  │                      │   │
│  │  🔥 Heat oil in wok  │   │  ← Current step card
│  │  until shimmering    │   │     (big, centered, rounded)
│  │                      │   │
│  │  👀 Watching...      │   │  ← Completion indicator
│  └──────────────────────┘   │
│                              │
│  ┌──────────────────────┐   │
│  │      ⏱ 04:32         │   │  ← Timer display (big numbers)
│  │    pasta cooking      │   │     (only visible when active)
│  └──────────────────────┘   │
│                              │
│  ┌──────────────────────┐   │
│  │     ✅ DONE!          │   │  ← Action button
│  └──────────────────────┘   │     (only for user_confirm steps)
│                              │
│  ⚠️ BOIL-OVER DETECTED!     │  ← Safety alert overlay
│     Reduce heat now!         │     (slides in from top, red)
└──────────────────────────────┘
```

### Visual Style (Overcooked Game Aesthetic)

- **Background:** Warm cream/beige (#FFF8E7) with subtle kitchen pattern
- **Step card:** Rounded corners (16px), white card with orange left border, drop shadow
- **Timer:** Bold chunky font, yellow/orange gradient background, pulsing animation when < 30s
- **Done button:** Big green rounded button (#4CAF50) with bouncy press animation
- **Safety alert:** Red banner sliding from top with shake animation, semi-transparent overlay
- **Typography:** Bold sans-serif, large sizes for kitchen readability
- **Step transitions:** Slide-out-left / slide-in-right animation when advancing steps
- **Completion indicators:**
  - `vlm` → pulsing eye icon with "Watching..."
  - `timer` → countdown numbers
  - `user_confirm` → green DONE button appears

### WebSocket Protocol

**Server → Phone:**

```json
{"type": "step", "step_id": 3, "total_steps": 6, "instruction": "Heat oil in wok until shimmering", "completion_type": "vlm", "dish": "mapo_tofu"}

{"type": "timer", "name": "step_4_timer", "remaining_seconds": 272, "total_seconds": 300}

{"type": "safety", "message": "Boil-over detected! Reduce heat now!", "severity": "critical"}

{"type": "done", "dish": "mapo_tofu"}
```

**Phone → Server:**

```json
{"type": "user_confirm", "step_id": 2}
```

### File Changes

| File | Action |
|------|--------|
| `src/ui/index.html` | **Create** — single-page phone UI |
| `src/ui/server.py` | **Modify** — add HTTP serving for index.html, add timer broadcast |

## Demo Slide Script

6 slides, 5-minute presentation.

### Slide 1: The Problem (30s)

**Title:** "Your Kitchen Doesn't Have an AI Copilot — Yet"

**Bullets:**
- Multitasking in the kitchen is stressful — juggling timers, parallel steps, and safety
- Existing recipe apps show a wall of text. You scroll with greasy hands
- No one watches your pot while you're chopping onions

**Speaker note:** "Who here has burned pasta because they forgot the timer? Or had a boil-over because they were busy with the sauce?"

### Slide 2: Our Solution (30s)

**Title:** "An On-Device AI Cooking Companion"

**Bullets:**
- Tells you one step at a time — no recipe overload
- Watches your cooking with a camera and detects when each step is done
- Alerts you if something goes wrong (boil-over, smoke)
- Runs entirely on-device — no cloud, no latency, no privacy concerns

**Visual:** Show the phone UI mockup

### Slide 3: Live Demo (2 min)

**Title:** "Let's Cook Mapo Tofu"

**Script:**
1. "I tell the system: mapo tofu"
2. Phone shows Step 1: "Dice the tofu"
3. Camera detects diced tofu → phone slides to Step 2
4. User taps "Done" for mincing step → Step 3
5. VLM detects shimmering oil → Step 4 with timer
6. Timer counts down → Step 5 with timer
7. Show a safety alert interrupting (boil-over)
8. Recipe completes → celebration screen

### Slide 4: How It Works (1 min)

**Title:** "Architecture: Thin Agent, Thick Rules"

**Key points:**
- Gemma 3 VLM (fine-tuned on 36 synthetic images) classifies kitchen state
- LLM generates recipe once, cached as YAML — no LLM call per cycle
- Rule-based step engine handles transitions — fast, reliable, debuggable
- 3 completion types play to model strengths: VLM for visual, timer for cooking, user confirm for the rest

**Speaker note:** "We use the LLM where it shines — recipe knowledge and vision — and deterministic code for everything else."

### Slide 5: Results (30s)

**Title:** "What We Learned from Fine-Tuning"

**Key metrics:**
- State classification: 25% → 61% after fine-tuning
- Water boil-over detection: 100% (pasta, ramen, soup, rice)
- Training: 6 min 45 sec on 48 samples, 8.5GB VRAM
- Model: Gemma 3 4B with QLoRA (0.69% of parameters trained)

**Speaker note:** "We fine-tuned with only 36 synthetic images and got meaningful improvements. More data will push this further."

### Slide 6: What's Next (30s)

**Title:** "From Demo to Kitchen"

**Bullets:**
- More training data (real kitchen photos, more dishes)
- Vision layer fine-tuning (currently language layers only)
- Deploy on edge device (Raspberry Pi + Coral TPU or phone with MediaPipe)
- Voice input ("Hey chef, I'm done")

**Closing:** "We built a cooking copilot that watches, guides, and protects — all on-device. Thank you!"
