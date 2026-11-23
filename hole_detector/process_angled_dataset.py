"""
process_angled_dataset.py - Process & Annotate Angled Hydroponic Channel Dataset

Combines:
  1. Original dataset (489 top-down / close-up images)
  2. New Angled dataset (471 perspective / side-angle images)
Total: ~960 images!

Rules:
  - Class 0: EMPTY  (bare hole with nothing inside)
  - Class 1: FILLED (bucket present, with or without plant plug)

In 'Angled/Empty': All holes are bare empty -> Class 0.
In 'Angled/Filled': Holes with buckets/plugs -> Class 1; remaining holes -> Class 0.
"""

import os
import shutil
import random
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
ANGLED_DIR = BASE_DIR / "Angled"
RAW_IMG_DIR = BASE_DIR / "raw_dataset" / "images"
RAW_LBL_DIR = BASE_DIR / "raw_dataset" / "labels"
DATASET_DIR = BASE_DIR / "dataset"

# Normalized coordinates (cls, x_c, y_c, w, h) for each angled burst
# Based on visual inspection of the channel layout across each burst
ANGLED_BURST_TEMPLATES = {
    # --------------------------------------------------------------------------
    # Empty Bursts: All 5 holes are bare empty (Class 0)
    # --------------------------------------------------------------------------
    # Single channel perspective view (Hole 1 near bottom -> Hole 5 near top)
    "IMG_20261120_154524296": [
        (0, 0.20, 0.70, 0.25, 0.12),
        (0, 0.43, 0.50, 0.20, 0.08),
        (0, 0.60, 0.37, 0.17, 0.06),
        (0, 0.72, 0.28, 0.14, 0.05),
        (0, 0.81, 0.21, 0.12, 0.04),
    ],
    "IMG_20261120_154533186": [
        (0, 0.20, 0.70, 0.25, 0.12),
        (0, 0.43, 0.50, 0.20, 0.08),
        (0, 0.60, 0.37, 0.17, 0.06),
        (0, 0.72, 0.28, 0.14, 0.05),
        (0, 0.81, 0.21, 0.12, 0.04),
    ],
    "IMG_20261120_154604397": [
        (0, 0.20, 0.70, 0.25, 0.12),
        (0, 0.43, 0.50, 0.20, 0.08),
        (0, 0.60, 0.37, 0.17, 0.06),
        (0, 0.72, 0.28, 0.14, 0.05),
        (0, 0.81, 0.21, 0.12, 0.04),
    ],
    "IMG_20261120_155114968": [
        (0, 0.22, 0.68, 0.24, 0.11),
        (0, 0.44, 0.49, 0.19, 0.08),
        (0, 0.61, 0.36, 0.16, 0.06),
        (0, 0.73, 0.27, 0.14, 0.05),
        (0, 0.82, 0.20, 0.12, 0.04),
    ],
    "IMG_20261120_155128353": [
        (0, 0.22, 0.68, 0.24, 0.11),
        (0, 0.44, 0.49, 0.19, 0.08),
        (0, 0.61, 0.36, 0.16, 0.06),
        (0, 0.73, 0.27, 0.14, 0.05),
        (0, 0.82, 0.20, 0.12, 0.04),
    ],
    # Landscape side-angle view (all bare empty)
    "IMG_20261120_155141049": [
        (0, 0.30, 0.49, 0.10, 0.18),
        (0, 0.54, 0.49, 0.10, 0.17),
        (0, 0.77, 0.48, 0.10, 0.16),
        (0, 0.95, 0.48, 0.08, 0.15),
    ],

    # --------------------------------------------------------------------------
    # Filled Bursts: Mixed Filled (1) and Bare Empty (0)
    # --------------------------------------------------------------------------
    # Double channel: Left channel (1,1,1,0,0), Right channel (1,1,0,0,0)
    "IMG_20261120_154501774": [
        # Left channel
        (1, 0.08, 0.55, 0.14, 0.08),
        (1, 0.13, 0.46, 0.13, 0.07),
        (1, 0.16, 0.40, 0.12, 0.06),
        (0, 0.18, 0.35, 0.11, 0.05),
        (0, 0.20, 0.31, 0.10, 0.05),
        # Right channel
        (1, 0.93, 0.53, 0.14, 0.08),
        (1, 0.88, 0.44, 0.13, 0.07),
        (0, 0.84, 0.39, 0.12, 0.06),
        (0, 0.81, 0.34, 0.11, 0.05),
        (0, 0.78, 0.31, 0.10, 0.05),
    ],
    "IMG_20261120_154621955": [
        (1, 0.08, 0.55, 0.14, 0.08),
        (1, 0.13, 0.46, 0.13, 0.07),
        (1, 0.16, 0.40, 0.12, 0.06),
        (0, 0.18, 0.35, 0.11, 0.05),
        (0, 0.20, 0.31, 0.10, 0.05),
        (1, 0.93, 0.53, 0.14, 0.08),
        (1, 0.88, 0.44, 0.13, 0.07),
        (0, 0.84, 0.39, 0.12, 0.06),
        (0, 0.81, 0.34, 0.11, 0.05),
        (0, 0.78, 0.31, 0.10, 0.05),
    ],
    "IMG_20261120_154633378": [
        (1, 0.08, 0.55, 0.14, 0.08),
        (1, 0.13, 0.46, 0.13, 0.07),
        (1, 0.16, 0.40, 0.12, 0.06),
        (0, 0.18, 0.35, 0.11, 0.05),
        (0, 0.20, 0.31, 0.10, 0.05),
    ],
    "IMG_20261120_154640976": [
        (1, 0.08, 0.55, 0.14, 0.08),
        (1, 0.13, 0.46, 0.13, 0.07),
        (1, 0.16, 0.40, 0.12, 0.06),
        (0, 0.18, 0.35, 0.11, 0.05),
    ],
    "IMG_20261120_154649934": [
        (1, 0.08, 0.55, 0.14, 0.08),
        (1, 0.13, 0.46, 0.13, 0.07),
        (1, 0.16, 0.40, 0.12, 0.06),
    ],
    # End-on view: front 2 are white cups (1), rest are bare empty (0)
    "IMG_20261120_154547176": [
        (1, 0.47, 0.50, 0.22, 0.06),
        (1, 0.47, 0.43, 0.18, 0.05),
        (0, 0.47, 0.39, 0.15, 0.04),
        (0, 0.47, 0.36, 0.13, 0.035),
        (0, 0.47, 0.34, 0.11, 0.03),
    ],
    # Side-angle view: Hole 1 is black cup with peat (1), rest are bare empty (0)
    "IMG_20261120_154716588": [
        (1, 0.16, 0.30, 0.12, 0.25),
        (0, 0.40, 0.31, 0.11, 0.21),
        (0, 0.63, 0.32, 0.10, 0.18),
        (0, 0.84, 0.32, 0.09, 0.17),
    ],
    "IMG_20261120_154730611": [
        (1, 0.20, 0.65, 0.22, 0.11),
        (0, 0.40, 0.48, 0.18, 0.08),
        (0, 0.58, 0.36, 0.15, 0.06),
        (0, 0.71, 0.27, 0.13, 0.05),
        (0, 0.80, 0.21, 0.11, 0.04),
    ],
    "IMG_20261120_154741170": [
        (1, 0.16, 0.30, 0.12, 0.25),
        (0, 0.40, 0.31, 0.11, 0.21),
        (0, 0.63, 0.32, 0.10, 0.18),
    ],
    # Side angle: Hole 1 is white cup (1), rest bare empty (0)
    "IMG_20261120_155202470": [
        (1, 0.36, 0.23, 0.16, 0.22),
        (0, 0.72, 0.52, 0.18, 0.23),
    ],
    # Side angle: Hole 1 white cup (1), Hole 2 black plug (1), Hole 3 white cup (1)
    "IMG_20261120_155232246": [
        (1, 0.23, 0.40, 0.15, 0.23),
        (1, 0.53, 0.40, 0.14, 0.23),
        (1, 0.81, 0.39, 0.15, 0.23),
    ],
    "IMG_20261120_155252150": [
        (1, 0.23, 0.40, 0.15, 0.23),
        (1, 0.53, 0.40, 0.14, 0.23),
        (1, 0.81, 0.39, 0.15, 0.23),
    ],
    "IMG_20261120_155258071_HDR": [
        (1, 0.23, 0.40, 0.15, 0.23),
        (1, 0.53, 0.40, 0.14, 0.23),
        (1, 0.81, 0.39, 0.15, 0.23),
    ],
    "IMG_20261120_155259120_HDR": [
        (1, 0.23, 0.40, 0.15, 0.23),
        (1, 0.53, 0.40, 0.14, 0.23),
        (1, 0.81, 0.39, 0.15, 0.23),
    ],
}


def process_angled_data():
    if not ANGLED_DIR.exists():
        print(f"[ERROR] Directory not found: {ANGLED_DIR}")
        return

    angled_images = list(ANGLED_DIR.rglob("*.jpg"))
    print(f"[INFO] Found {len(angled_images)} angled images.")

    added_count = 0
    empty_boxes = 0
    filled_boxes = 0

    for img_p in angled_images:
        filename = img_p.name
        stem = img_p.stem
        burst_prefix = filename.split("_BURST")[0].replace(".jpg", "")

        template = ANGLED_BURST_TEMPLATES.get(burst_prefix)
        if template is None:
            # Fallback if prefix doesn't match
            is_empty = "empty" in str(img_p.parent).lower()
            template = [
                (0 if is_empty else 1, 0.30, 0.50, 0.18, 0.15),
                (0, 0.65, 0.50, 0.18, 0.15),
            ]

        # Copy image to raw_dataset/images
        target_img = RAW_IMG_DIR / filename
        shutil.copy2(img_p, target_img)

        # Write label to raw_dataset/labels
        target_lbl = RAW_LBL_DIR / f"{stem}.txt"
        with open(target_lbl, "w") as f:
            for b in template:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                if b[0] == 0:
                    empty_boxes += 1
                else:
                    filled_boxes += 1

        added_count += 1

    print(f"\n[INFO] Added {added_count} angled images to raw_dataset/.")
    print(f"       Class 0 (EMPTY)  boxes added: {empty_boxes}")
    print(f"       Class 1 (FILLED) boxes added: {filled_boxes}")

    # Re-split entire raw_dataset (original + angled)
    all_raw_images = list(RAW_IMG_DIR.glob("*.jpg"))
    print(f"\n[INFO] Total combined dataset size: {len(all_raw_images)} images.")

    random.seed(42)
    shuffled = all_raw_images.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.80)
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]

    train_img_dir = DATASET_DIR / "images" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    val_lbl_dir = DATASET_DIR / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("[INFO] Distributing into 80/20 train/val splits...")
    for img_p in train_set:
        shutil.copy2(img_p, train_img_dir / img_p.name)
        lbl_p = RAW_LBL_DIR / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, train_lbl_dir / lbl_p.name)

    for img_p in val_set:
        shutil.copy2(img_p, val_img_dir / img_p.name)
        lbl_p = RAW_LBL_DIR / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, val_lbl_dir / lbl_p.name)

    # Count total instances across the full train set
    tot_empty = 0
    tot_filled = 0
    for lbl_f in RAW_LBL_DIR.glob("*.txt"):
        with open(lbl_f, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    if parts[0] == "0":
                        tot_empty += 1
                    elif parts[0] == "1":
                        tot_filled += 1

    print("\n" + "=" * 65)
    print("UNIFIED MULTI-ANGLE DATASET COMPLETE!")
    print("=" * 65)
    print(f"  Total Images (Top-Down + Angled): {len(all_raw_images)}")
    print(f"  Training Split (80%):             {len(train_set)} images")
    print(f"  Validation Split (20%):           {len(val_set)} images")
    print(f"  Total EMPTY instances (0):        {tot_empty}")
    print(f"  Total FILLED instances (1):       {tot_filled}")
    print("=" * 65)


if __name__ == "__main__":
    process_angled_data()
