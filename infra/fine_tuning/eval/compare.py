"""
Evaluate base vs fine-tuned model on kitchen policy tasks.

Generates before/after comparison logs showing improvement in:
- Tool call consistency
- Appropriate alert prioritization
- Reduced hallucinated actions
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--tuned-model", required=True, help="Fine-tuned model path")
    parser.add_argument(
        "--test-data",
        default="data/labels/test_pairs.json",
        help="Test data path",
    )
    args = parser.parse_args()

    # TODO: Implement comparison
    # 1. Load both models
    # 2. Run same test inputs through both
    # 3. Compare: schema compliance, action relevance, hallucination rate
    # 4. Output side-by-side comparison

    print("Evaluation script ready. Implement comparison logic.")


if __name__ == "__main__":
    main()
