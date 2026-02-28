"""
Export fine-tuned model for on-device deployment.

Merges LoRA weights with base model and exports in a format
suitable for local inference (GGUF, SafeTensors, etc.)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter path")
    parser.add_argument("--base-model", required=True, help="Base model name/path")
    parser.add_argument("--output-path", required=True, help="Export output path")
    parser.add_argument(
        "--format",
        choices=["safetensors", "gguf"],
        default="safetensors",
        help="Export format",
    )
    args = parser.parse_args()

    # TODO: Implement export
    # 1. Load base model + LoRA adapter
    # 2. Merge weights
    # 3. Export in target format
    # 4. Quantize if needed (4-bit for on-device)

    print("Export script ready. Implement model merging and export.")


if __name__ == "__main__":
    main()
