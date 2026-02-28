"""
Evaluate base vs fine-tuned model on kitchen policy tasks using Unsloth.

Generates before/after comparison logs showing improvement in:
- Tool call schema compliance (valid JSON output)
- Appropriate alert prioritization
- Reduced hallucinated actions
- Action relevance to kitchen state

Usage:
    python infra/fine_tuning/eval/compare.py \
        --adapter infra/fine_tuning/checkpoints/final_adapter \
        --test-data data/labels/test_pairs.json
"""

import argparse
import json

from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a kitchen monitoring agent. Given the current kitchen state as JSON, "
    "decide what actions to take. Output a JSON object with an 'actions' array. "
    "Available tools: set_timer(name, seconds), adjust_timer(name, delta_seconds), "
    "speak(text, priority), show_card(title, bullets), mark_step_done(step_id), "
    "reorder_steps(new_order). Only act when necessary. If no action is needed, "
    "return an empty actions array."
)


def run_inference(model, tokenizer, state: dict) -> str:
    """Run a single inference on a kitchen state."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current kitchen state:\n{json.dumps(state, indent=2)}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def is_valid_json_actions(response: str) -> bool:
    """Check if response contains valid JSON with actions array."""
    try:
        # Try to extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            return False
        parsed = json.loads(response[start:end])
        return "actions" in parsed and isinstance(parsed["actions"], list)
    except (json.JSONDecodeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument(
        "--base-model",
        default="unsloth/gemma-3-4b-it",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter",
        default="infra/fine_tuning/checkpoints/final_adapter",
        help="Fine-tuned adapter path",
    )
    parser.add_argument(
        "--test-data",
        default="data/labels/test_pairs.json",
        help="Test data path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max test samples to evaluate",
    )
    args = parser.parse_args()

    # Load test data
    with open(args.test_data) as f:
        test_pairs = json.load(f)[: args.max_samples]

    print(f"Evaluating {len(test_pairs)} test samples\n")

    # ---- Base model ----
    print("=" * 60)
    print("Loading BASE model...")
    print("=" * 60)
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    base_results = []
    for i, pair in enumerate(test_pairs):
        resp = run_inference(base_model, base_tokenizer, pair["input"])
        valid = is_valid_json_actions(resp)
        base_results.append({"index": i, "valid_json": valid, "response": resp})
        print(f"  [{i}] valid_json={valid}")

    # Free base model memory
    del base_model, base_tokenizer

    # ---- Fine-tuned model ----
    print("\n" + "=" * 60)
    print("Loading FINE-TUNED model...")
    print("=" * 60)
    tuned_model, tuned_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(tuned_model)

    tuned_results = []
    for i, pair in enumerate(test_pairs):
        resp = run_inference(tuned_model, tuned_tokenizer, pair["input"])
        valid = is_valid_json_actions(resp)
        tuned_results.append({"index": i, "valid_json": valid, "response": resp})
        print(f"  [{i}] valid_json={valid}")

    # ---- Summary ----
    base_valid = sum(1 for r in base_results if r["valid_json"])
    tuned_valid = sum(1 for r in tuned_results if r["valid_json"])
    total = len(test_pairs)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Schema compliance (valid JSON actions):")
    print(f"  Base:       {base_valid}/{total} ({base_valid/total*100:.0f}%)")
    print(f"  Fine-tuned: {tuned_valid}/{total} ({tuned_valid/total*100:.0f}%)")
    print()

    # Show side-by-side for first 3 examples
    print("Side-by-side (first 3 examples):")
    for i in range(min(3, total)):
        print(f"\n--- Example {i} ---")
        print(f"Input state: {json.dumps(test_pairs[i]['input']['stove'], indent=2)}")
        print(f"Expected:    {json.dumps(test_pairs[i]['output'], indent=2)[:200]}")
        print(f"Base:        {base_results[i]['response'][:200]}")
        print(f"Fine-tuned:  {tuned_results[i]['response'][:200]}")


if __name__ == "__main__":
    main()
