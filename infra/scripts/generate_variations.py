"""
Generate image variations using SDXL img2img on existing cropped images.
Takes each cropped cell + its prompt, generates N variations with controlled noise.

Usage:
    python /tmp/generate_variations.py /tmp/cropped/ --prompts /tmp/matrix_prompts.txt --variations 3
"""

import argparse
import json
import time
import torch
from pathlib import Path
from PIL import Image


NEGATIVE = (
    "cartoon, anime, illustration, drawing, painting, CGI, render, "
    "low quality, blurry, text, watermark, logo, distorted, "
    "extra objects, unrealistic colors, side angle, diagonal view"
)


def parse_prompts(prompt_file):
    """Parse matrix_prompts.txt into {dish/state: prompt} dict."""
    prompts = {}
    with open(prompt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("NEGATIVE"):
                continue
            if "|" not in line:
                continue
            key, prompt = line.split("|", 1)
            key = key.strip()       # e.g. "pasta/1_cold"
            prompt = prompt.strip()
            prompts[key] = prompt
    return prompts


def generate(args):
    from diffusers import StableDiffusionXLImg2ImgPipeline

    img_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = parse_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    print("Loading SDXL img2img pipeline...")
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    print(f"Loaded in {time.time() - t0:.1f}s | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    dishes = ["pasta", "ramen", "mapo_tofu", "curry", "soup", "rice"]
    states = ["1_cold", "2_simmer", "3_boil", "4_BOIL_OVER", "5_done_ok", "6_FAILED"]

    total_expected = 0
    for dish in dishes:
        for state in states:
            src = img_dir / dish / f"{state}.png"
            if src.exists():
                total_expected += args.variations

    count = 0
    labels = []

    for dish in dishes:
        dish_out = output_dir / dish
        dish_out.mkdir(parents=True, exist_ok=True)

        for state in states:
            src_path = img_dir / dish / f"{state}.png"
            if not src_path.exists():
                print(f"  SKIP: {dish}/{state}.png not found")
                continue

            key = f"{dish}/{state}"
            prompt = prompts.get(key, "")
            if not prompt:
                print(f"  SKIP: no prompt for {key}")
                continue

            # Load and resize source image to SDXL native resolution
            src_image = Image.open(src_path).convert("RGB")
            src_image = src_image.resize((768, 768), Image.LANCZOS)

            is_failure = "BOIL_OVER" in state or "FAILED" in state

            for v in range(args.variations):
                count += 1
                fname = f"{state}_v{v+1}.png"
                out_path = dish_out / fname

                if out_path.exists() and not args.overwrite:
                    print(f"  [{count}/{total_expected}] SKIP: {dish}/{fname}")
                    labels.append({
                        "file": f"{dish}/{fname}",
                        "dish": dish, "state": state,
                        "is_failure": is_failure, "variation": v + 1,
                    })
                    continue

                seed = 42 + v * 1000 + hash(key) % 10000
                generator = torch.Generator("cuda").manual_seed(seed)

                t1 = time.time()
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE,
                    image=src_image,
                    strength=args.strength,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]
                gen_time = time.time() - t1

                result.save(str(out_path))
                labels.append({
                    "file": f"{dish}/{fname}",
                    "dish": dish, "state": state,
                    "is_failure": is_failure, "variation": v + 1,
                })

                tag = "FAIL" if is_failure else "OK"
                print(f"  [{count}/{total_expected}] {dish}/{fname} [{tag}] ({gen_time:.1f}s) seed={seed}")

    # Save labels
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {count} variations -> {output_dir}")
    print(f"Labels: {labels_path}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_dir", help="Directory with existing cropped images (dish/state.png)")
    p.add_argument("--prompts", required=True, help="Path to matrix_prompts.txt")
    p.add_argument("--output-dir", default="/tmp/variations")
    p.add_argument("--variations", type=int, default=3)
    p.add_argument("--strength", type=float, default=0.35,
                   help="img2img strength (0=identical, 1=fully new). 0.3-0.4 recommended")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
