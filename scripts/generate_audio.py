#!/usr/bin/env python3
"""
Pre-generate ElevenLabs audio files for recipe steps and common announcements.

Usage:
    python scripts/generate_audio.py --recipe configs/recipes/rice.yaml --output data/audio/rice/
    python scripts/generate_audio.py --recipe configs/recipes/pasta_and_sauce.yaml

Generates MP3 files:
    step_1.mp3, step_2.mp3, ...        — step instruction announcements
    step_1_done.mp3, step_2_done.mp3   — step completion announcements
    safety_boilover.mp3                 — safety alerts
    recipe_start.mp3, recipe_done.mp3   — bookend announcements

Requires ELEVENLABS_API_KEY environment variable (or .env file).
"""

import argparse
import sys
from pathlib import Path

import yaml

# Allow running from repo root: python scripts/generate_audio.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent.tts import TTSEngine


def generate_recipe_audio(recipe_path: str, output_dir: str | None = None):
    """Generate audio files for every step in a recipe."""
    recipe_path = Path(recipe_path)
    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)

    dish = recipe["dish"]
    steps = recipe["steps"]

    if output_dir is None:
        slug = dish.lower().replace(" ", "_")
        output_dir = f"data/audio/{slug}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tts = TTSEngine(audio=False)  # No local playback, just file generation

    print(f"Generating audio for: {dish} ({len(steps)} steps)")
    print(f"Output directory: {out}\n")

    # Recipe start
    _gen(tts, f"Let's start cooking {dish}!", out / "recipe_start.mp3")

    # Per-step announcements
    for step in steps:
        sid = step["id"]
        instruction = step["instruction"]

        # Step activation
        _gen(
            tts,
            f"Step {sid}: {instruction}",
            out / f"step_{sid}.mp3",
        )

        # Step completion
        _gen(
            tts,
            f"Step {sid} complete: {instruction}",
            out / f"step_{sid}_done.mp3",
        )

    # Safety alerts
    _gen(tts, "Boil-over detected! Reduce heat immediately!",
         out / "safety_boilover.mp3")
    _gen(tts, "Smoke detected! Check the stove immediately!",
         out / "safety_smoke.mp3")

    # Recipe done
    _gen(tts, f"{dish} is ready! Enjoy your meal!",
         out / "recipe_done.mp3")

    print(f"\nDone! Generated audio in {out}")


def _gen(tts: TTSEngine, text: str, path: Path):
    """Generate a single audio file, skipping if it already exists."""
    if path.exists():
        print(f"  SKIP  {path.name} (already exists)")
        return
    print(f"  GEN   {path.name}: {text[:60]}...")
    tts.generate_file(text, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-generate ElevenLabs audio for recipe steps")
    parser.add_argument(
        "--recipe", required=True,
        help="Path to recipe YAML (e.g. configs/recipes/rice.yaml)")
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: data/audio/<dish>/)")
    args = parser.parse_args()

    generate_recipe_audio(args.recipe, args.output)
