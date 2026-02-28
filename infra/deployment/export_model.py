"""
Export fine-tuned model for on-device deployment using Unsloth.

Merges LoRA adapter with base model and exports to:
- GGUF format (for llama.cpp, Ollama, LM Studio)
- SafeTensors (for HuggingFace / direct loading)

Usage:
    # Export to GGUF (recommended for on-device)
    python infra/deployment/export_model.py \
        --adapter infra/fine_tuning/checkpoints/final_adapter \
        --format gguf --quantization q4_k_m

    # Export to SafeTensors
    python infra/deployment/export_model.py \
        --adapter infra/fine_tuning/checkpoints/final_adapter \
        --format safetensors
"""

import argparse

from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model with Unsloth")
    parser.add_argument(
        "--adapter",
        default="infra/fine_tuning/checkpoints/final_adapter",
        help="LoRA adapter path",
    )
    parser.add_argument(
        "--output-dir",
        default="infra/deployment/exported_model",
        help="Export output directory",
    )
    parser.add_argument(
        "--format",
        choices=["gguf", "safetensors"],
        default="gguf",
        help="Export format",
    )
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        help="GGUF quantization method (q4_k_m, q8_0, q5_k_m, f16)",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="HuggingFace repo to push to (e.g., 'username/model-name')",
    )
    args = parser.parse_args()

    # Load the fine-tuned model
    print(f"Loading adapter from: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=1024,
        load_in_4bit=True,
    )

    if args.format == "gguf":
        print(f"Exporting to GGUF ({args.quantization})...")
        if args.push_to_hub:
            model.push_to_hub_gguf(
                args.push_to_hub,
                tokenizer,
                quantization_method=[args.quantization],
            )
            print(f"Pushed to: https://huggingface.co/{args.push_to_hub}")
        else:
            model.save_pretrained_gguf(
                args.output_dir,
                tokenizer,
                quantization_method=args.quantization,
            )
            print(f"GGUF saved to: {args.output_dir}")

    elif args.format == "safetensors":
        print("Merging adapter and exporting to SafeTensors...")
        model.save_pretrained_merged(
            args.output_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to: {args.output_dir}")

    print("Export complete.")


if __name__ == "__main__":
    main()
