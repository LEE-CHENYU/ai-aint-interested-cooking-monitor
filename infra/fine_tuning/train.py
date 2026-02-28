"""
Fine-tuning script for kitchen policy model using Unsloth.

Fine-tunes FunctionGemma / Gemma 3 on kitchen state -> tool call mappings.
Input: structured kitchen state (JSON)
Output: agent tool calls (set_timer, speak, mark_step_done, etc.)

Uses Unsloth for 2x faster training and 60% less VRAM via optimized QLoRA.

Usage:
    python infra/fine_tuning/train.py
    python infra/fine_tuning/train.py --config infra/fine_tuning/configs/lora_config.yaml
    python infra/fine_tuning/train.py --model google/gemma-3-4b-it  # override model
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only


SYSTEM_PROMPT = (
    "You are a kitchen monitoring agent. Given the current kitchen state as JSON, "
    "decide what actions to take. Output a JSON object with an 'actions' array. "
    "Available tools: set_timer(name, seconds), adjust_timer(name, delta_seconds), "
    "speak(text, priority), show_card(title, bullets), mark_step_done(step_id), "
    "reorder_steps(new_order). Only act when necessary. If no action is needed, "
    "return an empty actions array."
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_and_format_data(data_path: str) -> list[dict]:
    """Load training pairs and convert to conversation format for Gemma."""
    with open(data_path) as f:
        pairs = json.load(f)

    conversations = []
    for pair in pairs:
        convo = {
            "conversations": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Current kitchen state:\n{json.dumps(pair['input'], indent=2)}",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(pair["output"], indent=2),
                },
            ]
        }
        conversations.append(convo)

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Fine-tune kitchen policy model with Unsloth")
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
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_name = args.model or config["model"]["name"]
    print(f"Model: {model_name}")
    print(f"Config: {config}")

    # -----------------------------------------------------------
    # 1. Load model with Unsloth (handles quantization internally)
    # -----------------------------------------------------------
    max_seq_length = config["training"]["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=config["model"]["quantization"] == "4bit",
        dtype=None,  # Auto-detect
    )

    # -----------------------------------------------------------
    # 2. Apply LoRA adapter via Unsloth
    # -----------------------------------------------------------
    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
    )

    # -----------------------------------------------------------
    # 3. Set up Gemma chat template
    # -----------------------------------------------------------
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # -----------------------------------------------------------
    # 4. Load and format training data
    # -----------------------------------------------------------
    conversations = load_and_format_data(args.data)
    dataset = Dataset.from_list(conversations)

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    print(f"Training samples: {len(dataset)}")
    print(f"Example formatted:\n{dataset[0]['text'][:500]}...")

    # -----------------------------------------------------------
    # 5. Configure SFTTrainer
    # -----------------------------------------------------------
    training_cfg = config["training"]
    output_cfg = config["output"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=training_cfg["epochs"],
            per_device_train_batch_size=training_cfg["batch_size"],
            gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
            learning_rate=training_cfg["learning_rate"],
            warmup_steps=training_cfg["warmup_steps"],
            max_seq_length=max_seq_length,
            save_steps=output_cfg["save_steps"],
            logging_steps=output_cfg["logging_steps"],
            fp16=True,
            dataset_text_field="text",
        ),
    )

    # Only train on model responses, not the user prompt
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # -----------------------------------------------------------
    # 6. Train
    # -----------------------------------------------------------
    print("Starting training...")
    stats = trainer.train()
    print(f"Training complete. Stats: {stats}")

    # -----------------------------------------------------------
    # 7. Save adapter weights
    # -----------------------------------------------------------
    adapter_path = Path(args.output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to: {adapter_path}")

    # -----------------------------------------------------------
    # 8. Quick sanity check inference
    # -----------------------------------------------------------
    FastLanguageModel.for_inference(model)
    test_state = {
        "stove": {"water_state": "boiling", "pot_present": True, "smoke_suspected": False},
        "timers": [],
        "steps": [{"id": 1, "name": "add pasta", "status": "pending"}],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current kitchen state:\n{json.dumps(test_state, indent=2)}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nSanity check response:\n{response}")


if __name__ == "__main__":
    main()
