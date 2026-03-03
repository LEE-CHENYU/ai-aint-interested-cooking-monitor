# 60-Second Presentation Script

## Timing Guide: ~60 seconds total

---

### WHAT (15 sec)

We built an **on-device AI cooking companion** that guides you through a recipe one step at a time using a camera. You say "mapo tofu"

### WHY (15 sec)

Cooking is multitasking — you're juggling timers, parallel steps, and safety hazards at the same time. Existing recipe apps just show a wall of text. Our system solves this by **watching your kitchen and reacting in real time** — it detects boil-overs and smoke *while* you're chopping onions. And because it's fully on-device, your kitchen video never leaves your home — **privacy by design, zero latency, works offline**.

### HOW (20 sec)

Under the hood: **Gemma 3 4B VLM**, fine-tuned with QLoRA on 36 synthetic images — trained in under 7 minutes on a single GPU. The model classifies kitchen state each frame. We pair it with a **rule-based agentic loop** — the LLM generates the recipe once, then deterministic rules handle step transitions, so we get fast 3-second inference cycles with no per-step LLM calls. Three completion types play to the model's strengths: **VLM detection** for visual steps, **timers** for cooking durations, and **user confirm** for what the camera can't see.

### RESULTS (10 sec)

After fine-tuning: state classification jumped from 25% to 61%, and **water-based boil-over detection hit 100%** — rice, pasta, soup, all caught. The system runs a live demo with a phone UI showing step cards, timer countdowns, and safety alerts, all driven by the agentic loop in real time.

---

## Anticipated Q&A

**Q: Why on-device instead of calling an API?**
A: Three reasons — privacy (kitchen video stays local), latency (safety alerts need sub-second response), and cost (no per-call pricing, runs indefinitely).

**Q: Why rule-based step transitions instead of using the LLM?**
A: Speed and reliability. The LLM is great at recipe knowledge and vision, but using it per-cycle would add 6-20 seconds of latency. Rules are instant and deterministic — no hallucinated transitions.

**Q: What are the three completion types?**
A: VLM (camera sees it's done, like a rolling boil), timer (fixed duration, like simmer 15 min), user_confirm (camera can't judge, user taps Done). This lets us play to the model's strengths instead of forcing it to detect everything.

**Q: How did you generate training data?**
A: 36 synthetic images from Gemini — 6 dishes times 6 states (cold, simmering, boiling, boil_over, smoke, done). Fine-tuned with Unsloth FastVisionModel, QLoRA 4-bit, language layers only, 0.69% of parameters.

**Q: What's the biggest limitation?**
A: Thick-liquid boil-overs (curry, mapo tofu sauce) — the model calls them "simmering." And burnt food detection is still at 50%. More diverse training data would fix both.

**Q: What's the latency?**
A: 3-5 seconds per frame after LoRA merge. Under unmerged LoRA it's 17-55s, which is why we always merge before deployment.

**Q: How does safety monitoring work?**
A: It runs every inference cycle independently of the cooking step. Both the VLM's `safe` flag and a deterministic rule engine check for hazards. If either fires, the alert interrupts immediately — safety never sleeps.
