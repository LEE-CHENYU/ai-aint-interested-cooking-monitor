"""
Generate dish × state matrix images on GCP instance.
Each row = different dish, each column = different cooking state.
Focus: normal vs failure states a 4B model can reliably distinguish.

Output: grid of images organized as data/matrix/{dish}/{state}.png
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

# ── MATRIX DEFINITION ────────────────────────────────────────────────────────
# Rows: dishes | Columns: states from OK → NOT OK
# All prompts are top-down overhead angle for consistency.

BASE = "Top-down overhead photo of {pot_type} on gas stove burner, {scene}, realistic kitchen, sharp detail, natural lighting"

DISHES = [
    {
        "id": "pasta",
        "pot_type": "large stainless steel pot",
        "states": {
            "cold_water":       "clear still water with dry spaghetti pasta bundle resting on counter beside pot, no bubbles, blue flame visible",
            "simmering":        "pasta spaghetti submerged in water with small gentle bubbles rising, light steam, moderate flame",
            "rolling_boil":     "pasta spaghetti in vigorous rolling boil, large bubbles breaking surface, significant steam, starchy foam at edges",
            "boil_over":        "starchy white foamy water spilling over pot rim onto stovetop, pasta visible inside, messy overflow dripping down sides, urgent situation",
            "done_perfect":     "cooked al dente spaghetti being lifted with tongs from pot, steam rising, clear golden broth",
            "failed_mush":      "overcooked mushy clumped pasta stuck together in murky starchy water, falling apart, grey and lifeless",
        },
    },
    {
        "id": "ramen",
        "pot_type": "black pot with ramen broth",
        "states": {
            "cold_water":       "cold pork bone broth with visible fat layer on surface, ramen noodle packet beside pot, no heat, still liquid",
            "simmering":        "ramen broth gently simmering with small bubbles, milky tonkotsu color, soft steam, noodles just added curling in broth",
            "rolling_boil":     "ramen broth at vigorous boil, noodles floating and swirling, foam forming at edges, heavy steam rising",
            "boil_over":        "milky ramen broth overflowing pot rim onto stovetop, foamy white liquid spilling everywhere, noodles visible, messy overflow",
            "done_perfect":     "beautiful ramen bowl being assembled, noodles in rich broth, soft boiled egg half, chashu pork, green onions, nori",
            "failed_mush":      "overcooked bloated ramen noodles disintegrating in cloudy murky broth, no structure left, mushy mess",
        },
    },
    {
        "id": "mapo_tofu",
        "pot_type": "carbon steel wok",
        "states": {
            "cold_water":       "cubed soft white tofu pieces sitting in cold water in wok, not yet heated, doubanjiang paste in small bowl nearby",
            "simmering":        "mapo tofu gently simmering in red chili oil sauce, tofu cubes intact, small bubbles, ground pork visible, sichuan peppercorns",
            "rolling_boil":     "mapo tofu sauce at vigorous boil, red oil splashing, tofu cubes bouncing in bubbling sauce, heavy steam",
            "boil_over":        "red spicy mapo tofu sauce overflowing wok onto stovetop, chili oil spilling over rim, tofu cubes falling out, messy red overflow",
            "done_perfect":     "beautiful finished mapo tofu in wok, silky tofu in glossy red sauce, garnished with green scallions and sichuan pepper, vibrant red color",
            "failed_mush":      "mapo tofu completely fallen apart into mushy paste, tofu disintegrated into grainy mush, sauce separated, unappetizing brown mess",
        },
    },
    {
        "id": "curry",
        "pot_type": "heavy dutch oven pot",
        "states": {
            "cold_water":       "raw chunked vegetables potatoes carrots and chicken pieces in cold water in pot, curry roux blocks sitting beside pot, not heated",
            "simmering":        "japanese curry gently simmering, golden brown sauce with potato and carrot chunks, small bubbles, rich thick sauce",
            "rolling_boil":     "curry at vigorous boil, sauce splattering, vegetables bouncing in bubbling thick sauce, heavy steam and splashes on stovetop",
            "boil_over":        "thick brown curry sauce overflowing pot rim onto stovetop, sticky sauce dripping down sides, potato chunks falling out, messy brown overflow",
            "done_perfect":     "beautiful thick golden japanese curry in pot, perfectly cooked potatoes and carrots, rich glossy sauce, steam rising gently",
            "failed_mush":      "burnt curry stuck to bottom of pot, black charred patches visible through broken sauce, scorched smell implied, dried out and ruined",
        },
    },
    {
        "id": "soup",
        "pot_type": "large stainless steel stock pot",
        "states": {
            "cold_water":       "clear cold water with raw chicken bones and whole vegetables celery carrots onion sitting in pot, not heated, clean clear water",
            "simmering":        "chicken soup gently simmering with golden color, small bubbles at edges, vegetables softened, herbs floating, clear broth",
            "rolling_boil":     "soup at vigorous rolling boil, turbulent surface with vegetables tumbling, foam and scum rising, heavy steam",
            "boil_over":        "chicken soup overflowing pot rim onto stovetop, broth with vegetables spilling over sides, liquid pooling on burner, messy overflow",
            "done_perfect":     "beautiful clear golden chicken soup in pot, perfectly cooked vegetables, fresh herbs floating, appetizing warm color",
            "failed_mush":      "murky grey cloudy soup with disintegrated mushy vegetables, unappetizing dark color, overcooked to mush, unappealing",
        },
    },
    {
        "id": "rice",
        "pot_type": "medium saucepan with glass lid beside it",
        "states": {
            "cold_water":       "dry uncooked white rice grains in saucepan with cold water just added, clear water with rice settled at bottom, glass lid beside pot",
            "simmering":        "rice absorbing water at gentle simmer, small bubbles visible at edges, steam escaping, water level dropping, rice grains swelling",
            "rolling_boil":     "rice water at vigorous boil, starchy white foam bubbling up aggressively, rice grains churning, heavy steam and foam",
            "boil_over":        "starchy white rice water foaming over saucepan rim onto stovetop, white foamy overflow dripping down sides, rice grains in spillage, messy",
            "done_perfect":     "perfectly cooked fluffy white rice in pot, each grain separate and distinct, steam rising gently, no excess water",
            "failed_mush":      "overcooked mushy sticky rice clumped into gluey mass stuck to pot bottom, burnt brown crust at edges, ruined texture",
        },
    },
]

STATES_ORDER = ["cold_water", "simmering", "rolling_boil", "boil_over", "done_perfect", "failed_mush"]
STATE_LABELS = {
    "cold_water": "1_cold",
    "simmering": "2_simmer",
    "rolling_boil": "3_boil",
    "boil_over": "4_BOIL_OVER",
    "done_perfect": "5_done_ok",
    "failed_mush": "6_FAILED",
}

NEGATIVE = (
    "cartoon, anime, illustration, drawing, painting, CGI, render, "
    "low quality, blurry, text, watermark, logo, distorted, "
    "extra objects, unrealistic colors, side angle, diagonal view"
)


def generate(args):
    from diffusers import StableDiffusionXLPipeline

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SDXL pipeline...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    print(f"Loaded in {time.time() - t0:.1f}s | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    dishes = DISHES[:args.max_dishes] if args.max_dishes else DISHES
    total = len(dishes) * len(STATES_ORDER) * args.variations
    count = 0
    labels = []

    for dish in dishes:
        dish_dir = output_dir / dish["id"]
        dish_dir.mkdir(parents=True, exist_ok=True)

        for state_key in STATES_ORDER:
            scene_desc = dish["states"][state_key]
            prompt = BASE.format(pot_type=dish["pot_type"], scene=scene_desc)
            state_label = STATE_LABELS[state_key]

            # Is this a failure state?
            is_failure = state_key in ("boil_over", "failed_mush")

            for v in range(args.variations):
                count += 1
                suffix = "" if args.variations == 1 else f"_v{v}"
                fname = f"{state_label}{suffix}.png"
                img_path = dish_dir / fname

                if img_path.exists() and not args.overwrite:
                    print(f"  [{count}/{total}] SKIP: {dish['id']}/{fname}")
                    labels.append({
                        "file": f"{dish['id']}/{fname}",
                        "dish": dish["id"],
                        "state": state_key,
                        "is_failure": is_failure,
                        "is_boil_over": state_key == "boil_over",
                    })
                    continue

                t1 = time.time()
                image = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE,
                    num_inference_steps=28,
                    guidance_scale=7.5,
                    height=768,
                    width=768,
                ).images[0]
                gen_time = time.time() - t1

                image.save(str(img_path))
                labels.append({
                    "file": f"{dish['id']}/{fname}",
                    "dish": dish["id"],
                    "state": state_key,
                    "is_failure": is_failure,
                    "is_boil_over": state_key == "boil_over",
                })
                tag = "FAIL" if is_failure else "OK"
                print(f"  [{count}/{total}] {dish['id']}/{fname} [{tag}] ({gen_time:.1f}s)")

    # Save labels
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    # Print matrix summary
    print(f"\n{'=' * 70}")
    print("MATRIX SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'DISH':<12}" + "".join(f"{STATE_LABELS[s]:>14}" for s in STATES_ORDER)
    print(header)
    print("-" * len(header))
    for dish in dishes:
        row = f"{dish['id']:<12}"
        for s in STATES_ORDER:
            tag = "FAIL" if s in ("boil_over", "failed_mush") else "ok"
            row += f"{'[' + tag + ']':>14}"
        print(row)

    print(f"\nTotal: {count} images -> {output_dir}")
    print(f"Labels: {labels_path}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/matrix_images")
    parser.add_argument("--variations", type=int, default=1, help="Images per cell")
    parser.add_argument("--max-dishes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
