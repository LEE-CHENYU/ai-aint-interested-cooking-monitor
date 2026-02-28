"""
Generate synthetic kitchen images directly on GCP instance using SDXL.
Runs entirely on-device — no API credits needed.

Usage (on instance):
    python generate_on_instance.py                    # all scenes
    python generate_on_instance.py --max-scenes 5     # quick test
    python generate_on_instance.py --batch-size 2     # save VRAM
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

# ── Scene definitions (subset covering key categories) ──────────────────────
# Full prompts for each critical scene type our agent needs to recognize.

SCENES = [
    # ---- COOKING: Boiling stages ----
    {
        "id": "water_cold",
        "angle": "top_view",
        "prompt": "Top-down overhead photo of stainless steel pot on gas stove burner, clear still water, no bubbles, flame visible around pot base, realistic kitchen, sharp detail",
        "label": {"zone": "stove", "water_state": "cold", "pot_present": True, "smoke_suspected": False},
    },
    {
        "id": "water_simmering",
        "angle": "top_view",
        "prompt": "Top-down overhead photo of stainless steel pot on gas stove, gentle bubbles rising slowly through water, light steam wisps, moderate flame, realistic kitchen",
        "label": {"zone": "stove", "water_state": "simmering", "pot_present": True, "smoke_suspected": False},
    },
    {
        "id": "water_rolling_boil",
        "angle": "top_view",
        "prompt": "Top-down overhead photo of stainless steel pot on gas stove, vigorous large rolling boil, significant steam rising, active bubbles breaking surface, realistic kitchen",
        "label": {"zone": "stove", "water_state": "boiling", "pot_present": True, "smoke_suspected": False},
    },
    {
        "id": "pasta_in_boiling_water",
        "angle": "top_view",
        "prompt": "Top-down photo of pot with spaghetti pasta in boiling water on stove, steam rising, pasta partially submerged, tongs nearby, realistic kitchen",
        "label": {"zone": "stove", "water_state": "boiling", "pot_present": True, "contents": "pasta"},
    },
    {
        "id": "boil_over",
        "angle": "top_view",
        "prompt": "Top-down photo of pot boiling over on gas stove, starchy water spilling over pot rim onto stovetop, steam and mess, urgent situation, realistic kitchen",
        "label": {"zone": "stove", "water_state": "boil_over", "pot_present": True, "smoke_suspected": False},
    },
    # ---- COOKING: Frying/Wok stages ----
    {
        "id": "oil_shimmering_pan",
        "angle": "top_view",
        "prompt": "Top-down photo of frying pan on gas stove with shimmering hot oil, oil rippling and moving, ready for cooking, no food yet, realistic kitchen",
        "label": {"zone": "stove", "oil_state": "shimmering", "pan_present": True, "smoke_suspected": False},
    },
    {
        "id": "oil_smoking",
        "angle": "top_view",
        "prompt": "Top-down photo of frying pan on gas stove with smoking oil, visible wisps of smoke rising from oil surface, dangerously hot, realistic kitchen",
        "label": {"zone": "stove", "oil_state": "smoking", "pan_present": True, "smoke_suspected": True},
    },
    {
        "id": "wok_stir_fry",
        "angle": "top_view",
        "prompt": "Top-down photo of wok on high flame with vegetables and protein being stir fried, tossing action, colorful ingredients, high heat searing, Asian kitchen",
        "label": {"zone": "stove", "wok_state": "stir_frying", "temperature": "very_high"},
    },
    {
        "id": "food_browning_pan",
        "angle": "top_view",
        "prompt": "Top-down photo of chicken breast browning in frying pan, one side golden brown visible, oil sizzling, tongs nearby on counter, realistic kitchen",
        "label": {"zone": "stove", "food_state": "browning_one_side", "pan_present": True},
    },
    {
        "id": "food_burning",
        "angle": "top_view",
        "prompt": "Top-down photo of food burning in frying pan on stove, dark charred spots visible, smoke rising from pan, overcooked, realistic kitchen, concerning situation",
        "label": {"zone": "stove", "food_state": "burning", "smoke_suspected": True},
    },
    # ---- PREP: Cutting/Chopping ----
    {
        "id": "onion_being_diced",
        "angle": "front_view",
        "prompt": "Front-facing photo of hands with proper claw grip dicing onion on wooden cutting board, chef knife in motion, partially diced pieces, warm kitchen light",
        "label": {"zone": "counter", "ingredient": "onion", "prep_state": "dicing", "hands_active": True, "knife_in_use": True},
    },
    {
        "id": "mise_en_place",
        "angle": "front_view",
        "prompt": "Front-facing photo of kitchen counter with multiple small prep bowls containing diced vegetables, minced garlic, measured spices, organized mise en place, clean workspace",
        "label": {"zone": "counter", "prep_state": "mise_en_place_complete", "hands_active": False},
    },
    {
        "id": "garlic_mincing",
        "angle": "front_view",
        "prompt": "Front-facing photo of hands mincing garlic cloves on cutting board with chef knife, very fine pieces, garlic press nearby, warm kitchen",
        "label": {"zone": "counter", "ingredient": "garlic", "prep_state": "mincing", "hands_active": True, "knife_in_use": True},
    },
    # ---- SAFETY: Smoke/Fire ----
    {
        "id": "heavy_smoke_stove",
        "angle": "top_view",
        "prompt": "Top-down photo of kitchen stove with heavy smoke rising from pan, thick grey smoke obscuring view, dangerous situation, no person visible, abandoned cooking, realistic",
        "label": {"zone": "stove", "smoke_suspected": True, "person_present": False, "severity": "critical"},
    },
    {
        "id": "towel_near_flame",
        "angle": "front_view",
        "prompt": "Front-facing photo of kitchen stove with cloth towel draped dangerously close to gas flame burner, fire hazard, tea towel about to catch fire, realistic kitchen",
        "label": {"zone": "stove", "safety_hazard": "towel_near_flame", "severity": "high"},
    },
    # ---- SAFETY: Unattended ----
    {
        "id": "unattended_pot_boiling",
        "angle": "top_view",
        "prompt": "Top-down photo of pot boiling vigorously on gas stove, no person visible, no hands in frame, empty kitchen, unattended cooking, steam rising, realistic",
        "label": {"zone": "stove", "water_state": "boiling", "person_present": False, "unattended": True},
    },
    {
        "id": "unattended_frying",
        "angle": "top_view",
        "prompt": "Top-down photo of frying pan with oil on active gas flame, no person visible, completely unattended, kitchen empty, dangerous situation, realistic",
        "label": {"zone": "stove", "oil_state": "hot", "person_present": False, "unattended": True},
    },
    # ---- CUISINE: Chinese ----
    {
        "id": "wok_hei_moment",
        "angle": "top_view",
        "prompt": "Top-down photo of carbon steel wok on high flame, tossing fried rice with visible wok hei flame licking up sides, egg and rice flying, intense heat, Chinese cooking",
        "label": {"zone": "stove", "wok_state": "wok_hei", "cuisine": "chinese", "temperature": "very_high"},
    },
    {
        "id": "doubanjiang_frying",
        "angle": "top_view",
        "prompt": "Top-down photo of wok with red chili bean paste doubanjiang frying in hot oil, sichuan peppercorns visible, deep red color, aromatic, Chinese Sichuan cooking",
        "label": {"zone": "stove", "cuisine": "chinese", "dish": "mapo_tofu", "step": "fry_doubanjiang"},
    },
    # ---- CUISINE: Thai ----
    {
        "id": "pad_thai_wok",
        "angle": "top_view",
        "prompt": "Top-down photo of wok with pad thai noodles being tossed, rice noodles with shrimp, bean sprouts, egg, tamarind sauce, lime wedge nearby, Thai cooking",
        "label": {"zone": "stove", "cuisine": "thai", "dish": "pad_thai", "wok_state": "tossing"},
    },
    # ---- CUISINE: Japanese ----
    {
        "id": "tempura_frying",
        "angle": "top_view",
        "prompt": "Top-down photo of deep pot with golden tempura pieces frying in hot oil, bubbles around battered vegetables and shrimp, chopsticks holding piece, Japanese cooking",
        "label": {"zone": "stove", "cuisine": "japanese", "dish": "tempura", "oil_state": "at_temp"},
    },
    # ---- NORMAL: No action needed ----
    {
        "id": "normal_cooking_person",
        "angle": "front_view",
        "prompt": "Front-facing photo of person cooking at stove, stirring pot with wooden spoon, normal safe cooking scene, everything under control, well-lit kitchen",
        "label": {"zone": "stove", "person_present": True, "hands_active": True, "normal_operation": True},
    },
    {
        "id": "normal_chopping",
        "angle": "front_view",
        "prompt": "Front-facing photo of person safely chopping carrots on cutting board, proper technique, no hazards, organized clean kitchen, natural lighting",
        "label": {"zone": "counter", "person_present": True, "hands_active": True, "normal_operation": True},
    },
]

# Total: 23 base scenes × (1 base + 4 variations) = 115 images


def generate(args):
    from diffusers import StableDiffusionXLPipeline

    output_dir = Path(args.output_dir)

    negative = (
        "cartoon, anime, illustration, drawing, painting, CGI, render, "
        "low quality, blurry, text, watermark, logo, distorted hands, "
        "extra fingers, deformed, unrealistic colors"
    )

    # Load SDXL
    print("Loading SDXL pipeline...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    scenes = SCENES[: args.max_scenes] if args.max_scenes else SCENES
    labels = []
    total = len(scenes) * (1 + args.num_variations)
    count = 0

    for scene in scenes:
        angle_dir = output_dir / scene["angle"]
        angle_dir.mkdir(parents=True, exist_ok=True)

        prompts_batch = [scene["prompt"]]
        suffixes = ["_base"]

        # Add variation prompts with lighting/style modifiers
        modifiers = [
            "warm natural window light",
            "dim evening ambient lighting",
            "bright overhead fluorescent light",
            "harsh single-source light with shadows",
            "modern white clean kitchen",
            "small apartment cramped kitchen",
            "rustic wood and stone kitchen",
            "busy cluttered counter with spice bottles",
        ]
        import random
        for v in range(args.num_variations):
            mod = random.choice(modifiers)
            prompts_batch.append(f"{scene['prompt']}, {mod}")
            suffixes.append(f"_var{v:03d}")

        for prompt, suffix in zip(prompts_batch, suffixes):
            count += 1
            img_path = angle_dir / f"{scene['id']}{suffix}.png"

            if img_path.exists() and not args.overwrite:
                print(f"  [{count}/{total}] SKIP (exists): {img_path.name}")
                labels.append({
                    "file": str(img_path.relative_to(output_dir)),
                    "scene_id": scene["id"],
                    "angle": scene["angle"],
                    "is_variation": suffix != "_base",
                    "label": scene["label"],
                })
                continue

            t1 = time.time()
            image = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=768,
                width=768,
            ).images[0]
            gen_time = time.time() - t1

            image.save(str(img_path))
            labels.append({
                "file": str(img_path.relative_to(output_dir)),
                "scene_id": scene["id"],
                "angle": scene["angle"],
                "is_variation": suffix != "_base",
                "label": scene["label"],
            })
            print(f"  [{count}/{total}] {img_path.name} ({gen_time:.1f}s)")

    # Save labels
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone: {count} images -> {output_dir}")
    print(f"Labels: {labels_path}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/synthetic_images")
    parser.add_argument("--num-variations", type=int, default=4)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
