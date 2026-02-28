"""
Fine-tuning script for kitchen policy model.

Fine-tunes FunctionGemma / Gemma on kitchen state -> tool call mappings.
Input: structured kitchen state (JSON)
Output: agent tool calls (set_timer, speak, mark_step_done, etc.)
"""

import argparse
import json
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_training_data(data_path: str) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune kitchen policy model")
    parser.add_argument(
        "--config",
        default="infra/fine_tuning/configs/lora_config.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--data",
        default="data/labels/training_pairs.json",
        help="Training data path",
    )
    parser.add_argument(
        "--output-dir",
        default="infra/fine_tuning/checkpoints",
        help="Output directory for checkpoints",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Training config: {config}")

    # TODO: Implement training loop
    # 1. Load base model (FunctionGemma or Gemma 3)
    # 2. Apply LoRA adapter
    # 3. Load training data
    # 4. Train with SFTTrainer
    # 5. Save adapter weights

    print("Training script ready. Implement training loop with your model choice.")


if __name__ == "__main__":
    main()
