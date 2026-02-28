# Track 1: Fine-Tuning & Infrastructure

## GCP Instance

- **VM**: `hackathon-vm-ai-aint-interested`
- **IP**: `35.238.6.1`
- **SSH**: `ssh hackathon@35.238.6.1`

## Setup

```bash
# On the GCP instance
cd ~
git clone https://github.com/LEE-CHENYU/ai-aint-interested-cooking-monitor.git
cd ai-aint-interested-cooking-monitor
pip install -r infra/requirements.txt
```

## Fine-Tuning Workflow

1. Generate synthetic training data: `python data/generate_synthetic.py`
2. Configure training: edit `infra/fine_tuning/configs/`
3. Run training: `python infra/fine_tuning/train.py`
4. Evaluate: `python infra/fine_tuning/eval/compare.py`
5. Export model: `python infra/deployment/export_model.py`

## Models

Target models from the hackathon handout:
- **FunctionGemma 270M**: Fine-tune for tool-calling (kitchen policy -> actions)
- **Gemma 3n E4B**: Vision model for event detection
- **Gemma 3 4B/12B**: Larger reasoning model (if GPU allows)

Fine-tuning approach: LoRA/QLoRA on the agent policy model
- Input: structured kitchen state + timers + recipe context
- Output: tool calls (set_timer, speak, alert, mark_step_done)
