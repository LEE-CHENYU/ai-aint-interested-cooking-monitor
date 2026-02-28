"""
Synthetic image generation for kitchen monitoring training data.

Generates base images from text prompts, then creates variations using img2img.
Images are labeled and organized by camera angle (top_view / front_view).

Supports two backends:
  1. Local: Stable Diffusion XL via diffusers (needs GPU)
  2. API:  Google Gemini Imagen / Replicate API (no local GPU needed)

Usage:
    # Generate with local SDXL (on GCP GPU instance)
    python data/generate_images.py --backend local

    # Generate with Replicate API (from laptop, no GPU needed)
    python data/generate_images.py --backend replicate --api-key your_key

    # Generate specific scene only
    python data/generate_images.py --scene-id water_boiling --num-variations 5
"""

import argparse
import json
import random
from pathlib import Path

import yaml


def load_prompts(config_path: str = "data/synthetic/image_prompts.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_variation_prompt(base_prompt: str, settings: dict) -> str:
    """Create a varied prompt by adding random modifiers."""
    modifiers = settings.get("modifiers", {})
    additions = []
    for category, options in modifiers.items():
        additions.append(random.choice(options))
    return f"{base_prompt}, {', '.join(additions)}"


def generate_local(prompts_config: dict, output_dir: Path, scene_filter: str | None = None):
    """Generate images using local Stable Diffusion XL."""
    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    # Load SDXL text-to-image pipeline
    print("Loading SDXL pipeline...")
    txt2img = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Load img2img pipeline (shares model weights)
    img2img = StableDiffusionXLImg2ImgPipeline(
        vae=txt2img.vae,
        text_encoder=txt2img.text_encoder,
        text_encoder_2=txt2img.text_encoder_2,
        tokenizer=txt2img.tokenizer,
        tokenizer_2=txt2img.tokenizer_2,
        unet=txt2img.unet,
        scheduler=txt2img.scheduler,
    ).to("cuda")

    settings = prompts_config["variation_settings"]
    negative_prompt = settings["negative_prompt"]
    labels = []

    for use_case_name, use_case in prompts_config.items():
        if use_case_name == "variation_settings":
            continue

        for zone_name, zone in use_case.items():
            angle = zone["camera_angle"]
            angle_dir = output_dir / angle
            angle_dir.mkdir(parents=True, exist_ok=True)

            for scene in zone["scenes"]:
                scene_id = scene["id"]
                if scene_filter and scene_id != scene_filter:
                    continue

                print(f"\nGenerating: {use_case_name}/{zone_name}/{scene_id}")

                # Generate base image
                base_image = txt2img(
                    prompt=scene["prompt"],
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=settings["guidance_scale"],
                    height=768,
                    width=768,
                ).images[0]

                base_path = angle_dir / f"{scene_id}_base.png"
                base_image.save(str(base_path))
                labels.append({
                    "file": str(base_path.relative_to(output_dir)),
                    "scene_id": scene_id,
                    "angle": angle,
                    "is_variation": False,
                    "label": scene["label"],
                })
                print(f"  Saved base: {base_path}")

                # Generate variations via img2img
                num_vars = scene.get("variations", 5)
                strength_min, strength_max = settings["strength_range"]

                for v in range(num_vars):
                    strength = random.uniform(strength_min, strength_max)
                    var_prompt = build_variation_prompt(scene["prompt"], settings)

                    var_image = img2img(
                        prompt=var_prompt,
                        negative_prompt=negative_prompt,
                        image=base_image,
                        strength=strength,
                        guidance_scale=settings["guidance_scale"],
                        num_inference_steps=25,
                    ).images[0]

                    var_path = angle_dir / f"{scene_id}_var{v:03d}.png"
                    var_image.save(str(var_path))
                    labels.append({
                        "file": str(var_path.relative_to(output_dir)),
                        "scene_id": scene_id,
                        "angle": angle,
                        "is_variation": True,
                        "variation_strength": round(strength, 2),
                        "label": scene["label"],
                    })
                    print(f"  Saved variation {v}: {var_path} (strength={strength:.2f})")

    return labels


def generate_replicate(prompts_config: dict, output_dir: Path, api_key: str, scene_filter: str | None = None):
    """Generate images using Replicate API (no local GPU needed)."""
    import replicate
    import urllib.request

    client = replicate.Client(api_token=api_key)
    settings = prompts_config["variation_settings"]
    negative_prompt = settings["negative_prompt"]
    labels = []

    for use_case_name, use_case in prompts_config.items():
        if use_case_name == "variation_settings":
            continue

        for zone_name, zone in use_case.items():
            angle = zone["camera_angle"]
            angle_dir = output_dir / angle
            angle_dir.mkdir(parents=True, exist_ok=True)

            for scene in zone["scenes"]:
                scene_id = scene["id"]
                if scene_filter and scene_id != scene_filter:
                    continue

                print(f"\nGenerating: {use_case_name}/{zone_name}/{scene_id}")
                num_total = 1 + scene.get("variations", 5)

                for i in range(num_total):
                    is_variation = i > 0
                    prompt = (
                        build_variation_prompt(scene["prompt"], settings)
                        if is_variation
                        else scene["prompt"]
                    )

                    output = client.run(
                        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                        input={
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "width": 768,
                            "height": 768,
                            "num_inference_steps": 30,
                        },
                    )

                    suffix = f"_var{i-1:03d}" if is_variation else "_base"
                    img_path = angle_dir / f"{scene_id}{suffix}.png"
                    urllib.request.urlretrieve(output[0], str(img_path))

                    labels.append({
                        "file": str(img_path.relative_to(output_dir)),
                        "scene_id": scene_id,
                        "angle": angle,
                        "is_variation": is_variation,
                        "label": scene["label"],
                    })
                    print(f"  Saved: {img_path}")

    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic kitchen images")
    parser.add_argument(
        "--backend",
        choices=["local", "replicate"],
        default="local",
        help="Generation backend",
    )
    parser.add_argument("--api-key", default=None, help="API key for replicate backend")
    parser.add_argument("--scene-id", default=None, help="Generate only this scene")
    parser.add_argument(
        "--output-dir",
        default="data/synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--prompts",
        default="data/synthetic/image_prompts.yaml",
        help="Prompts config path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prompts_config = load_prompts(args.prompts)

    if args.backend == "local":
        labels = generate_local(prompts_config, output_dir, args.scene_id)
    elif args.backend == "replicate":
        if not args.api_key:
            print("Error: --api-key required for replicate backend")
            return
        labels = generate_replicate(prompts_config, output_dir, args.api_key, args.scene_id)

    # Save labels manifest
    labels_path = Path("data/labels") / "image_labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone. Generated {len(labels)} images.")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
