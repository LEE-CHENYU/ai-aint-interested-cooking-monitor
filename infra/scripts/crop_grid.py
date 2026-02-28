"""
Crop a 6×6 grid image into individual cells.
Each row = dish, each column = state.

Usage:
    python infra/scripts/crop_grid.py data/synthetic/grid.png
    python infra/scripts/crop_grid.py data/synthetic/grid.png --output data/synthetic/cropped
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

ROWS = ["pasta", "ramen", "mapo_tofu", "curry", "soup", "rice"]
COLS = ["1_cold", "2_simmer", "3_boil", "4_BOIL_OVER", "5_done_ok", "6_FAILED"]

LABELS = {
    "1_cold":       {"status": "normal", "cooking_active": False},
    "2_simmer":     {"status": "normal", "cooking_active": True},
    "3_boil":       {"status": "normal", "cooking_active": True},
    "4_BOIL_OVER":  {"status": "failure", "cooking_active": True, "boil_over": True},
    "5_done_ok":    {"status": "normal", "cooking_active": False, "done": True},
    "6_FAILED":     {"status": "failure", "cooking_active": False, "ruined": True},
}


def crop(args):
    img = Image.open(args.input)
    w, h = img.size
    print(f"Grid image: {w}×{h}")

    cell_w = w / len(COLS)
    cell_h = h / len(ROWS)
    print(f"Cell size: {cell_w:.0f}×{cell_h:.0f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    labels = []

    for r, dish in enumerate(ROWS):
        dish_dir = output_dir / dish
        dish_dir.mkdir(parents=True, exist_ok=True)

        for c, state in enumerate(COLS):
            left = int(c * cell_w)
            top = int(r * cell_h)
            right = int((c + 1) * cell_w)
            bottom = int((r + 1) * cell_h)

            cell = img.crop((left, top, right, bottom))
            fname = f"{state}.png"
            cell_path = dish_dir / fname
            cell.save(str(cell_path))

            label = {
                "file": f"{dish}/{fname}",
                "dish": dish,
                "state": state,
                **LABELS[state],
            }
            labels.append(label)
            tag = "FAIL" if LABELS[state]["status"] == "failure" else "ok"
            print(f"  [{tag:>4}] {dish}/{fname} ({cell.size[0]}×{cell.size[1]})")

    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nCropped {len(labels)} cells -> {output_dir}")
    print(f"Labels: {labels_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the grid image")
    parser.add_argument("--output", default="data/synthetic/cropped", help="Output directory")
    args = parser.parse_args()
    crop(args)


if __name__ == "__main__":
    main()
