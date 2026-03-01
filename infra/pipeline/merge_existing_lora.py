#!/usr/bin/env python3
"""Merge existing LoRA adapter into base model for download."""

import os
import torch

# Monkey-patch torch.compile for Blackwell GPU compatibility
_orig = torch.compile
def _noop(f=None, *a, **kw):
    return f if f is not None else (lambda fn: fn)
torch.compile = _noop

from unsloth import FastVisionModel

ADAPTER_DIR = "/tmp/vlm_ft_lora"
OUTPUT_DIR = "/tmp/vlm_ft_merged"

print("Loading model from LoRA adapter directory...")
model, tokenizer = FastVisionModel.from_pretrained(
    ADAPTER_DIR,
    load_in_4bit=True,
    max_seq_length=2048,
)
vram = torch.cuda.memory_allocated() / 1e9
is_peft = hasattr(model, "peft_config")
print(f"Model loaded. VRAM: {vram:.1f} GB | PeftModel: {is_peft}")

print("Merging LoRA into base model (16-bit)...")
model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method="merged_16bit",
)

total = sum(
    os.path.getsize(os.path.join(d, f))
    for d, _, files in os.walk(OUTPUT_DIR)
    for f in files
)
print(f"Merged model saved to {OUTPUT_DIR}")
print(f"Total size: {total / 1e9:.1f} GB")
