"""
process_angled2_dataset.py - Process Angled 2 Dataset & Merge into Full Dataset

Incorporates:
  1. Original Dataset (489 images)
  2. Angled 1 Dataset (471 images)
  3. Angled 2 Dataset (335 images - heavily focused on empty holes at low angles)
Total: ~1,295 images!

Prioritizes EMPTY hole detection:
  - Class 0: EMPTY  (bare hole with nothing inside)
  - Class 1: FILLED (bucket present, with or without plant plug)
"""

import os
import shutil
import random
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
ANGLED2_DIR = BASE_DIR / "Angled 2"
RAW_IMG_DIR = BASE_DIR / "raw_dataset" / "images"
RAW_LBL_DIR = BASE_DIR / "raw_dataset" / "labels"
DATASET_DIR = BASE_DIR / "dataset"

# Precise bounding templates for Angled 2 bursts
ANGLED2_TEMPLATES = {
    # --------------------------------------------------------------------------
    # Empty Bursts: All holes are bare empty (Class 0)
    # --------------------------------------------------------------------------
    # Perspective channel view (5 holes from bottom to top)
    "171343668": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    "171355088": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    "171408781": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    "171425705": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    "171436768": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    "171445157": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    "171449388": [
        (0, 0.304, 0.583, 0.22, 0.12),
        (0, 0.522, 0.410, 0.19, 0.09),
        (0, 0.704, 0.281, 0.16, 0.07),
        (0, 0.830, 0.173, 0.13, 0.05),
        (0, 0.911, 0.101, 0.10, 0.04),
    ],
    # Low-angle slant bursts: all bare empty holes (Class 0)
    "171938283": [
        (0, 0.21, 0.30, 0.24, 0.09),
        (0, 0.44, 0.22, 0.21, 0.07),
        (0, 0.67, 0.18, 0.18, 0.05),
        (0, 0.84, 0.15, 0.14, 0.04),
        (0, 0.95, 0.13, 0.10, 0.03),
    ],
    "171959637": [
        (0, 0.21, 0.30, 0.24, 0.09),
        (0, 0.44, 0.22, 0.21, 0.07),
        (0, 0.67, 0.18, 0.18, 0.05),
        (0, 0.84, 0.15, 0.14, 0.04),
        (0, 0.95, 0.13, 0.10, 0.03),
    ],
    "172015500": [
        (0, 0.21, 0.30, 0.24, 0.09),
        (0, 0.44, 0.22, 0.21, 0.07),
        (0, 0.67, 0.18, 0.18, 0.05),
        (0, 0.84, 0.15, 0.14, 0.04),
        (0, 0.95, 0.13, 0.10, 0.03),
    ],
    "172021755": [
        (0, 0.21, 0.30, 0.24, 0.09),
        (0, 0.44, 0.22, 0.21, 0.07),
        (0, 0.67, 0.18, 0.18, 0.05),
        (0, 0.84, 0.15, 0.14, 0.04),
        (0, 0.95, 0.13, 0.10, 0.03),
    ],
    "172026954": [
        (0, 0.21, 0.30, 0.24, 0.09),
        (0, 0.44, 0.22, 0.21, 0.07),
        (0, 0.67, 0.18, 0.18, 0.05),
        (0, 0.84, 0.15, 0.14, 0.04),
        (0, 0.95, 0.13, 0.10, 0.03),
    ],

    # --------------------------------------------------------------------------
    # Filled Bursts: 1 Filled (black plug), rest 4 are BARE EMPTY!
    # --------------------------------------------------------------------------
    "171521635": [
        (0, 0.136, 0.605, 0.24, 0.13),  # Hole 1: bare empty (0)
        (1, 0.374, 0.419, 0.21, 0.10),  # Hole 2: black cup with plug (1)
        (0, 0.561, 0.285, 0.18, 0.08),  # Hole 3: bare empty (0)
        (0, 0.673, 0.169, 0.15, 0.06),  # Hole 4: bare empty (0)
        (0, 0.805, 0.096, 0.12, 0.05),  # Hole 5: bare empty (0)
    ],
    "171528866": [
        (0, 0.136, 0.605, 0.24, 0.13),
        (1, 0.374, 0.419, 0.21, 0.10),
        (0, 0.561, 0.285, 0.18, 0.08),
        (0, 0.673, 0.169, 0.15, 0.06),
        (0, 0.805, 0.096, 0.12, 0.05),
    ],
    "171537361": [
        (0, 0.136, 0.605, 0.24, 0.13),
        (1, 0.374, 0.419, 0.21, 0.10),
        (0, 0.561, 0.285, 0.18, 0.08),
        (0, 0.673, 0.169, 0.15, 0.06),
        (0, 0.805, 0.096, 0.12, 0.05),
    ],
    "171545319": [
        (0, 0.136, 0.605, 0.24, 0.13),
        (1, 0.374, 0.419, 0.21, 0.10),
        (0, 0.561, 0.285, 0.18, 0.08),
        (0, 0.673, 0.169, 0.15, 0.06),
        (0, 0.805, 0.096, 0.12, 0.05),
    ],
    "171556258": [
        (0, 0.136, 0.605, 0.24, 0.13),
        (1, 0.374, 0.419, 0.21, 0.10),
        (0, 0.561, 0.285, 0.18, 0.08),
        (0, 0.673, 0.169, 0.15, 0.06),
        (0, 0.805, 0.096, 0.12, 0.05),
    ],
    "171603444": [
        (0, 0.136, 0.605, 0.24, 0.13),
        (1, 0.374, 0.419, 0.21, 0.10),
        (0, 0.561, 0.285, 0.18, 0.08),
        (0, 0.673, 0.169, 0.15, 0.06),
        (0, 0.805, 0.096, 0.12, 0.05),
    ],
    # Low-angle slant: Hole 1 black cup, Hole 2 white cup, Holes 3,4,5 bare empty
    "171938283_FILLED": [
        (1, 0.21, 0.30, 0.24, 0.09),  # black cup with plug (1)
        (1, 0.44, 0.22, 0.21, 0.07),  # white cup (1)
        (0, 0.67, 0.18, 0.18, 0.05),  # bare empty (0)
        (0, 0.84, 0.15, 0.14, 0.04),  # bare empty (0)
        (0, 0.95, 0.13, 0.10, 0.03),  # bare empty (0)
    ]
}


def process_dataset():
    if not ANGLED2_DIR.exists():
        print(f"[ERROR] Directory not found: {ANGLED2_DIR}")
        return

    images_a2 = list(ANGLED2_DIR.rglob("*.jpg"))
    print(f"[INFO] Found {len(images_a2)} images in Angled 2.")

    added_images = 0
    empty_boxes = 0
    filled_boxes = 0

    for img_p in images_a2:
        filename = img_p.name
        stem = img_p.stem
        is_empty_folder = "empty" in str(img_p.parent).lower()

        # Find matching template
        template = None
        for key in ANGLED2_TEMPLATES:
            if key in filename:
                if "171938283" in filename:
                    template = ANGLED2_TEMPLATES["171938283" if is_empty_folder else "171938283_FILLED"]
                else:
                    template = ANGLED2_TEMPLATES[key]
                break

        if template is None:
            # General fallback if key not matched
            template = [
                (0 if is_empty_folder else 1, 0.30, 0.50, 0.22, 0.12),
                (0, 0.52, 0.40, 0.19, 0.09),
                (0, 0.70, 0.28, 0.16, 0.07),
                (0, 0.83, 0.17, 0.13, 0.05),
                (0, 0.91, 0.10, 0.10, 0.04),
            ]

        # Copy to raw_dataset/images
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

        added_images += 1

    print(f"\n[INFO] Added {added_images} images from Angled 2.")
    print(f"       Class 0 (EMPTY)  boxes added: {empty_boxes}")
    print(f"       Class 1 (FILLED) boxes added: {filled_boxes}")

    # Re-split entire combined dataset
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

    print("[INFO] Rebuilding 80/20 train/val splits...")
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

    # Count total instances across dataset
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
    print("TOTAL DATASET RE-BUILD COMPLETE (ANGLED 2 INCLUDED)!")
    print("=" * 65)
    print(f"  Total Images (Top + Angled 1 + Angled 2): {len(all_raw_images)}")
    print(f"  Training Images (80%):                   {len(train_set)}")
    print(f"  Validation Images (20%):                 {len(val_set)}")
    print(f"  Total EMPTY instances (0 - Primary):     {tot_empty}")
    print(f"  Total FILLED instances (1):              {tot_filled}")
    print("=" * 65)


if __name__ == "__main__":
    process_dataset()
