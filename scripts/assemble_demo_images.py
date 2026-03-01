"""
Assemble demo image sequence from existing synthetic rice images.

Creates numbered symlinks in data/demo_sequence/ that map to the
demo timeline. Three sets are generated:

  60s/   — 9 images  (1 per next_image() call)
  120s/  — 15 images (extra variation frames for pacing)
  5m/    — 24 images (full set with multiple variations per state)

Usage:
  python scripts/assemble_demo_images.py
  python src/ui/server.py --duration 60s --images data/demo_sequence/60s
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "data" / "synthetic" / "matrix" / "matrix_images" / "rice"
VARIATIONS = ROOT / "data" / "synthetic" / "variations" / "rice"
OUT = ROOT / "data" / "demo_sequence"

# ── IMAGE MAPPING ────────────────────────────────────────────────
# Each entry: (source_dir, filename, demo_label)
#
# Demo timeline for rice:
#   Step 1 (VLM):  rinse rice          — cold images (prep phase)
#   Step 2 (user): add water to pot    — cold variation
#   Step 3 (VLM):  bring to boil       — cold → boil transition
#   Step 4 (timer): simmer 15 min      — simmer images
#   SAFETY:         boil-over!         — boil_over images
#   Step 5 (timer): let stand 5 min    — simmer (post-recovery)
#   Step 6 (VLM):  fluff and serve     — done images

SEQUENCE_60S = [
    # 9 images — one per next_image() call in demo
    (MATRIX,     "1_cold.png",        "01_step1_rice_uncooked.png"),
    (VARIATIONS, "1_cold_v1.png",     "02_step1_rice_rinsed.png"),
    (VARIATIONS, "1_cold_v2.png",     "03_step2_water_added.png"),
    (MATRIX,     "1_cold.png",        "04_step3_cold_water.png"),
    (MATRIX,     "3_boil.png",        "05_step3_rolling_boil.png"),
    (MATRIX,     "2_simmer.png",      "06_step4_simmering.png"),
    (MATRIX,     "4_BOIL_OVER.png",   "07_safety_boil_over.png"),
    (VARIATIONS, "2_simmer_v1.png",   "08_step5_steaming.png"),
    (MATRIX,     "5_done_ok.png",     "09_step6_done.png"),
]

SEQUENCE_120S = [
    # 15 images — extra variations for comfortable 2-min pacing
    (MATRIX,     "1_cold.png",        "01_step1_rice_uncooked.png"),
    (VARIATIONS, "1_cold_v1.png",     "02_step1_rice_rinsed.png"),
    (VARIATIONS, "1_cold_v2.png",     "03_step2_water_added.png"),
    (VARIATIONS, "1_cold_v3.png",     "04_step2_pot_ready.png"),
    (MATRIX,     "1_cold.png",        "05_step3_cold_water.png"),
    (VARIATIONS, "3_boil_v1.png",     "06_step3_heating_up.png"),
    (MATRIX,     "3_boil.png",        "07_step3_rolling_boil.png"),
    (MATRIX,     "2_simmer.png",      "08_step4_simmering.png"),
    (VARIATIONS, "2_simmer_v1.png",   "09_step4_simmer_steady.png"),
    (MATRIX,     "4_BOIL_OVER.png",   "10_safety_boil_over.png"),
    (VARIATIONS, "4_BOIL_OVER_v1.png","11_safety_boil_over_v2.png"),
    (VARIATIONS, "2_simmer_v2.png",   "12_step5_steaming.png"),
    (VARIATIONS, "2_simmer_v3.png",   "13_step5_resting.png"),
    (VARIATIONS, "5_done_ok_v1.png",  "14_step6_almost_done.png"),
    (MATRIX,     "5_done_ok.png",     "15_step6_done.png"),
]

SEQUENCE_5M = [
    # 24 images — full visual variety for 5-minute narrated demo
    (MATRIX,     "1_cold.png",        "01_step1_rice_uncooked.png"),
    (VARIATIONS, "1_cold_v1.png",     "02_step1_measuring.png"),
    (VARIATIONS, "1_cold_v2.png",     "03_step1_rinsed.png"),
    (VARIATIONS, "1_cold_v3.png",     "04_step2_water_added.png"),
    (MATRIX,     "1_cold.png",        "05_step2_pot_ready.png"),
    (VARIATIONS, "1_cold_v1.png",     "06_step3_cold_start.png"),
    (VARIATIONS, "3_boil_v1.png",     "07_step3_warming.png"),
    (VARIATIONS, "3_boil_v2.png",     "08_step3_bubbles.png"),
    (MATRIX,     "3_boil.png",        "09_step3_rolling_boil.png"),
    (MATRIX,     "2_simmer.png",      "10_step4_reduce_heat.png"),
    (VARIATIONS, "2_simmer_v1.png",   "11_step4_simmering.png"),
    (VARIATIONS, "2_simmer_v2.png",   "12_step4_gentle_simmer.png"),
    (VARIATIONS, "2_simmer_v3.png",   "13_step4_covered.png"),
    (MATRIX,     "4_BOIL_OVER.png",   "14_safety_boil_over.png"),
    (VARIATIONS, "4_BOIL_OVER_v1.png","15_safety_spilling.png"),
    (VARIATIONS, "4_BOIL_OVER_v2.png","16_safety_overflow.png"),
    (VARIATIONS, "2_simmer_v1.png",   "17_step5_recovered.png"),
    (VARIATIONS, "2_simmer_v2.png",   "18_step5_steaming.png"),
    (VARIATIONS, "2_simmer_v3.png",   "19_step5_resting.png"),
    (VARIATIONS, "5_done_ok_v1.png",  "20_step6_lid_off.png"),
    (VARIATIONS, "5_done_ok_v2.png",  "21_step6_fluffing.png"),
    (VARIATIONS, "5_done_ok_v3.png",  "22_step6_almost_done.png"),
    (MATRIX,     "5_done_ok.png",     "23_step6_done.png"),
    (MATRIX,     "5_done_ok.png",     "24_step6_final.png"),
]


def assemble(name: str, sequence: list):
    """Copy images into a numbered demo sequence directory."""
    out_dir = OUT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for src_dir, src_name, dst_name in sequence:
        src = src_dir / src_name
        dst = out_dir / dst_name
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue
        shutil.copy2(src, dst)

    print(f"  {name}/  — {len(list(out_dir.iterdir()))} images")


def main():
    print("Assembling demo image sequences from rice...")
    assemble("60s", SEQUENCE_60S)
    assemble("120s", SEQUENCE_120S)
    assemble("5m", SEQUENCE_5M)
    print(f"\nDone. Run with:")
    print(f"  python src/ui/server.py --duration 60s  --images data/demo_sequence/60s")
    print(f"  python src/ui/server.py --duration 120s --images data/demo_sequence/120s")
    print(f"  python src/ui/server.py --duration 5m   --images data/demo_sequence/5m")


if __name__ == "__main__":
    main()
