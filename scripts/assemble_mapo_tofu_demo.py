"""
Assemble mapo tofu demo image sequences at 60, 120, and 600 frames.

Creates numbered frame sequences in data/demo_sequence/mapo_tofu/ by
distributing source images across cooking phases with proportional
frame repetition to simulate a real-time camera feed.

Source images:
  data/synthetic/matrix/matrix_images/mapo_tofu/   (768x768 base states)
  data/synthetic/variations/mapo_tofu/              (768x768 variations)
  data/synthetic/matrix/task1_boilover/matrix_images/wok_*  (boilover detail)

Usage:
  python scripts/assemble_mapo_tofu_demo.py
"""

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "data" / "synthetic" / "matrix" / "matrix_images" / "mapo_tofu"
VARIATIONS = ROOT / "data" / "synthetic" / "variations" / "mapo_tofu"
BOILOVER_WOK = ROOT / "data" / "synthetic" / "matrix" / "task1_boilover" / "matrix_images"
OUT = ROOT / "data" / "demo_sequence" / "mapo_tofu"

TARGET_SIZE = 768

# ── COOKING PHASES ──────────────────────────────────────────────
# Each phase: (label, time_weight, [(source_dir, filename), ...])
#
# Mapo tofu timeline:
#   Phase 1: Prep         — raw tofu, aromatics on counter
#   Phase 2: Cook base    — aromatics + meat simmering in wok
#   Phase 3: Active cook  — sauce boiling, tofu added
#   Phase 4: Safety       — boil-over event (with wok close-ups)
#   Phase 5: Recovery     — reduce heat, gentle simmer
#   Phase 6: Done         — garnished, ready to serve

PHASES = [
    ("prep", 15, [
        (MATRIX,     "1_cold.png"),
        (VARIATIONS, "1_cold_v1.png"),
        (VARIATIONS, "1_cold_v2.png"),
        (VARIATIONS, "1_cold_v3.png"),
    ]),
    ("cook_base", 18, [
        (VARIATIONS, "2_simmer_v1.png"),
        (VARIATIONS, "2_simmer_v2.png"),
        (MATRIX,     "2_simmer.png"),
        (VARIATIONS, "2_simmer_v3.png"),
    ]),
    ("active_cook", 18, [
        (VARIATIONS, "3_boil_v1.png"),
        (VARIATIONS, "3_boil_v2.png"),
        (MATRIX,     "3_boil.png"),
        (VARIATIONS, "3_boil_v3.png"),
    ]),
    ("boil_over", 14, [
        (BOILOVER_WOK, "wok_3_rising.png"),
        (MATRIX,       "4_BOIL_OVER.png"),
        (BOILOVER_WOK, "wok_4_AT_RIM.png"),
        (VARIATIONS,   "4_BOIL_OVER_v1.png"),
        (BOILOVER_WOK, "wok_5_SPILLING.png"),
        (VARIATIONS,   "4_BOIL_OVER_v2.png"),
        (BOILOVER_WOK, "wok_6_OVERFLOW.png"),
        (VARIATIONS,   "4_BOIL_OVER_v3.png"),
    ]),
    ("recovery", 13, [
        (BOILOVER_WOK, "wok_2_safe_simmer.png"),
        (VARIATIONS,   "2_simmer_v3.png"),
        (VARIATIONS,   "2_simmer_v1.png"),
        (VARIATIONS,   "2_simmer_v2.png"),
    ]),
    ("done", 22, [
        (VARIATIONS, "5_done_ok_v1.png"),
        (VARIATIONS, "5_done_ok_v2.png"),
        (MATRIX,     "5_done_ok.png"),
        (VARIATIONS, "5_done_ok_v3.png"),
    ]),
]


def resize_to_square(src: Path, dst: Path, size: int = TARGET_SIZE):
    """Resize image to square, center-cropping if needed. No text added."""
    img = Image.open(src).convert("RGB")
    w, h = img.size

    if w == size and h == size:
        shutil.copy2(src, dst)
        return

    # Center-crop to square aspect ratio, then resize
    if w > h:
        left = (w - h) // 2
        img = img.crop((left, 0, left + h, h))
    elif h > w:
        top = (h - w) // 2
        img = img.crop((0, top, w, top + w))

    img = img.resize((size, size), Image.LANCZOS)
    img.save(dst, "PNG")


def distribute_frames(total_frames: int) -> list[tuple[Path, str]]:
    """Distribute total_frames across phases, repeating images as needed."""
    total_weight = sum(w for _, w, _ in PHASES)
    result = []

    frames_assigned = 0
    for i, (_, weight, images) in enumerate(PHASES):
        # Last phase gets remaining frames to avoid rounding gaps
        if i == len(PHASES) - 1:
            phase_frames = total_frames - frames_assigned
        else:
            phase_frames = round(total_frames * weight / total_weight)
        frames_assigned += phase_frames

        if phase_frames <= 0:
            continue

        # Distribute phase frames evenly across its images
        n_images = len(images)
        base_per_image = phase_frames // n_images
        remainder = phase_frames % n_images

        for j, (src_dir, filename) in enumerate(images):
            count = base_per_image + (1 if j < remainder else 0)
            for _ in range(count):
                result.append((src_dir, filename))

    return result


def assemble(name: str, total_frames: int):
    """Build a frame sequence directory for the given frame count."""
    out_dir = OUT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    frames = distribute_frames(total_frames)
    width = len(str(total_frames))

    for idx, (src_dir, filename) in enumerate(frames, 1):
        src = src_dir / filename
        dst = out_dir / f"frame_{idx:0{width}d}.png"
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue
        resize_to_square(src, dst)

    actual = len(list(out_dir.iterdir()))
    print(f"  {name}/  — {actual} frames")


def main():
    print("Assembling mapo tofu demo sequences...")
    assemble("60", 60)
    assemble("120", 120)
    assemble("600", 600)
    print(f"\nDone. Output: {OUT}")
    print(f"\nRun with:")
    print(f"  python scripts/run_demo.py --images data/demo_sequence/mapo_tofu/60  --mock")
    print(f"  python scripts/run_demo.py --images data/demo_sequence/mapo_tofu/120 --mock")
    print(f"  python scripts/run_demo.py --images data/demo_sequence/mapo_tofu/600 --mock")


if __name__ == "__main__":
    main()
