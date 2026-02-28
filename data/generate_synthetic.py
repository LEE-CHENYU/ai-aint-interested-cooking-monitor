"""
Synthetic training data generator.

Generates state -> action pairs for fine-tuning the kitchen policy model.
Enumerates combinations of kitchen states and produces expected tool calls.

Target: ~150-300 training pairs covering:
- water_state (not_hot, simmering, boiling, boil_over_risk)
- 0/1/2 active timers
- Step statuses (pending, in_progress, done)
- Severity flags (smoke, boil-over)
"""

import itertools
import json
from pathlib import Path


WATER_STATES = ["not_hot", "simmering", "boiling", "boil_over_risk"]
TIMER_CONFIGS = [
    [],
    [{"name": "pasta", "remaining_s": 300}],
    [{"name": "pasta", "remaining_s": 0}],  # expired
    [
        {"name": "pasta", "remaining_s": 120},
        {"name": "sauce", "remaining_s": 300},
    ],
]
SMOKE_FLAGS = [False, True]
STEP_STATUSES = ["pending", "in_progress"]


def generate_state(water_state, timers, smoke, step_status):
    """Generate a kitchen state input."""
    return {
        "stove": {
            "water_state": water_state,
            "pot_present": water_state != "not_hot",
            "smoke_suspected": smoke,
            "steam_level": "high" if water_state in ("boiling", "boil_over_risk") else "low",
        },
        "timers": timers,
        "steps": [
            {"id": 1, "name": "boil water", "status": "done" if water_state == "boiling" else step_status},
            {"id": 2, "name": "add pasta", "status": step_status},
            {"id": 3, "name": "make sauce", "status": "pending"},
        ],
    }


def generate_expected_actions(state):
    """Generate expected agent actions for a given state.

    These are the 'ground truth' actions used for fine-tuning.
    Hand-edit the top 50 examples for crisp behavior.
    """
    actions = []
    stove = state["stove"]

    # Critical: smoke detected
    if stove["smoke_suspected"]:
        actions.append({
            "tool": "speak",
            "args": {"text": "Smoke detected! Check the stove immediately.", "priority": "critical"},
        })
        return {"actions": actions}

    # Boil-over risk
    if stove["water_state"] == "boil_over_risk":
        actions.append({
            "tool": "speak",
            "args": {"text": "Boil-over risk. Lower the heat.", "priority": "high"},
        })

    # Water just boiled - check for pending pasta step
    if stove["water_state"] == "boiling":
        pasta_step = next((s for s in state["steps"] if s["name"] == "add pasta" and s["status"] == "pending"), None)
        if pasta_step:
            actions.append({
                "tool": "speak",
                "args": {"text": "Water is boiling. Add pasta now.", "priority": "high"},
            })
            actions.append({
                "tool": "set_timer",
                "args": {"name": "pasta", "seconds": 480},
            })
            actions.append({
                "tool": "mark_step_done",
                "args": {"step_id": pasta_step["id"]},
            })

    # Expired timers
    for timer in state["timers"]:
        if timer["remaining_s"] <= 0:
            actions.append({
                "tool": "speak",
                "args": {"text": f"{timer['name']} timer is done!", "priority": "high"},
            })

    # No action needed
    if not actions:
        return {"actions": [], "reasoning": "No action needed at this time."}

    return {"actions": actions}


def main():
    output_dir = Path("data/labels")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for water, timers, smoke, step in itertools.product(
        WATER_STATES, TIMER_CONFIGS, SMOKE_FLAGS, STEP_STATUSES
    ):
        state = generate_state(water, timers, smoke, step)
        actions = generate_expected_actions(state)
        pairs.append({"input": state, "output": actions})

    output_path = output_dir / "training_pairs.json"
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated {len(pairs)} training pairs -> {output_path}")


if __name__ == "__main__":
    main()
