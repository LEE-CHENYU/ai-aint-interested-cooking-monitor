# AI Kitchen Cooking Monitor

On-device AI system that monitors kitchen activity through camera feeds, tracks cooking procedure completion, and verifies compliance with safety guidelines. Built with Gemma models for the Google DeepMind x InstaLILY On-Device AI Hackathon.

## Why On-Device

- **Privacy**: Always-on kitchen cameras cannot stream to the cloud
- **Latency**: Safety hazards (smoke, boil-over) need sub-second reaction
- **Offline**: Kitchens have unreliable Wi-Fi; the system must keep working

## System Overview

```
Camera Feeds (top-down + front-facing)
        │
        ▼
┌─────────────────┐
│   Perception     │  Detect events: boiling, smoke, procedure steps
│   (Gemma 3n)     │
└────────┬────────┘
         │  Structured state JSON
         ▼
┌─────────────────┐
│   World State    │  Timers, step tracking, safety status
│   Engine         │
└────────┬────────┘
         │  Compact state snapshot
         ▼
┌─────────────────┐
│   Agent Loop     │  Decide + act autonomously
│   (FunctionGemma)│  Tool calls: set_timer, speak, alert
└────────┬────────┘
         │
         ▼
    Phone UI + Voice Output
```

## Two Purposes

1. **Procedure Compliance**: Track whether cooking steps are completed in order
2. **Safety Compliance**: Verify user actions follow safety guidelines (e.g., unattended stove, smoke detection)

## Team Structure

| Track | Owner | Directory | Focus |
|-------|-------|-----------|-------|
| Track 1 | Infra teammate | `infra/` | Fine-tuning on GCP, model deployment, GPU optimization |
| Track 2 | Agent teammate | `src/` | Perception pipeline, agent loop, data collection, UI |

## Setup

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### GCP Instance (Track 1)

```bash
ssh hackathon@35.238.6.1
# See infra/README.md for GPU setup instructions
pip install -r infra/requirements.txt
```

## Project Structure

```
├── configs/        # Shared: zone definitions, recipes, safety rules
├── infra/          # Track 1: fine-tuning, GCP, model deployment
├── src/            # Track 2: perception, agent, world state, UI
├── data/           # Shared: synthetic datasets and labels
├── scripts/        # Demo runner, utilities
└── docs/           # Plans and architecture docs
```

## Running the Demo

```bash
python scripts/run_demo.py --config configs/zones.yaml --recipe configs/recipes/pasta.yaml
```
