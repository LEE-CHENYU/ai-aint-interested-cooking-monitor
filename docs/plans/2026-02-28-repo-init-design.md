# Repo Initialization Design

**Date**: 2026-02-28
**Project**: AI Kitchen Cooking Monitor
**Hackathon**: Google DeepMind x InstaLILY On-Device AI Hackathon

## Context

On-device AI kitchen monitoring system using Gemma models. Two purposes:
1. Verify cooking procedure completion (step tracking)
2. Check compliance with safety/security guidelines

Two team tracks:
- **Track 1 (infra/)**: Fine-tuning, GCP GPU infrastructure, model deployment
- **Track 2 (src/)**: Data collection, labeling, agent flow, perception, UI

GCP VM: `hackathon-vm-ai-aint-interested` at `35.238.6.1`
Models: Gemma 3, Gemma 3n, FunctionGemma (on-device)
Data: Synthetic data from top-down and front-facing camera angles

## Directory Structure

```
ai-aint-interested-cooking-monitor/
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── plans/
│   └── architecture.md
├── configs/
│   ├── zones.yaml
│   ├── recipes/
│   └── safety_rules.yaml
├── infra/                          # TRACK 1
│   ├── README.md
│   ├── requirements.txt
│   ├── gcp/
│   │   └── setup.sh
│   ├── fine_tuning/
│   │   ├── train.py
│   │   ├── configs/
│   │   └── eval/
│   └── deployment/
│       └── export_model.py
├── src/                            # TRACK 2
│   ├── __init__.py
│   ├── perception/
│   │   ├── camera.py
│   │   ├── detector.py
│   │   └── temporal_smoother.py
│   ├── agent/
│   │   ├── agent_loop.py
│   │   ├── tools.py
│   │   └── schemas.py
│   ├── world_state/
│   │   ├── state.py
│   │   └── timer_engine.py
│   └── ui/
│       └── server.py
├── data/
│   ├── synthetic/
│   │   ├── top_view/
│   │   └── front_view/
│   ├── labels/
│   └── generate_synthetic.py
└── scripts/
    ├── run_demo.py
    └── record_backup.py
```

## Key Decisions

- **Two-Track Monorepo**: Maps team ownership to directory structure
- **Shared configs/**: Both tracks read zone definitions, recipes, safety rules
- **Shared data/**: Synthetic data generated once, consumed by both tracks
- **Credentials gitignored**: instance_info.txt excluded from version control
- **Python-only**: Entire stack in Python for hackathon speed
