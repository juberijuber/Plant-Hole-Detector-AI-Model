"""
2_split_dataset.py - Split Raw Images and YOLO Labels into Train/Val Sets (80/20)

Dataset Structure Created:
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import config

def split_dataset(train_ratio=0.8, seed=42):
    random.seed(seed)

    raw_images_dir = config.RAW_DATASET_DIR / "images"
    raw_labels_dir = config.RAW_DATASET_DIR / "labels"

    if not raw_images_dir.exists():
        print(f"[ERROR] Raw images directory does not exist: {raw_images_dir.resolve()}")
        print("Please run 1_capture.py first to collect images.")
        return

    # Valid image extensions
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f for f in raw_images_dir.iterdir() if f.suffix.lower() in image_exts])

    if not image_files:
        print(f"[ERROR] No images found in {raw_images_dir.resolve()}.")
        return

    print(f"[INFO] Found {len(image_files)} raw images.")

    # Target directories
    train_img_dir = config.DATASET_DIR / "images" / "train"
    val_img_dir = config.DATASET_DIR / "images" / "val"
    train_lbl_dir = config.DATASET_DIR / "labels" / "train"
    val_lbl_dir = config.DATASET_DIR / "labels" / "val"

    # Reset / create clean target directories
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Shuffle for unbiased split
    shuffled_images = image_files.copy()
    random.shuffle(shuffled_images)

    split_idx = int(len(shuffled_images) * train_ratio)
    train_images = shuffled_images[:split_idx]
    val_images = shuffled_images[split_idx:]

    print(f"[INFO] Splitting: {len(train_images)} train ({train_ratio*100:.0f}%), {len(val_images)} val ({(1-train_ratio)*100:.0f}%)...")

    missing_labels = 0
    empty_class_count = 0
    filled_class_count = 0

    def copy_pair(img_path, target_img_dir, target_lbl_dir):
        nonlocal missing_labels, empty_class_count, filled_class_count

        # Copy image
        shutil.copy2(img_path, target_img_dir / img_path.name)

        # Look for matching .txt label file
        label_file = raw_labels_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, target_lbl_dir / label_file.name)
            # Count class annotations
            try:
                with open(label_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id == 0:
                                empty_class_count += 1
                            elif cls_id == 1:
                                filled_class_count += 1
            except Exception:
                pass
        else:
            missing_labels += 1
            # Create an empty label file if none exists (negative sample)
            (target_lbl_dir / f"{img_path.stem}.txt").touch()

    # Process train pairs
    for img in train_images:
        copy_pair(img, train_img_dir, train_lbl_dir)

    # Process val pairs
    for img in val_images:
        copy_pair(img, val_img_dir, val_lbl_dir)

    print("\n" + "=" * 50)
    print("DATASET SPLIT COMPLETE")
    print("=" * 50)
    print(f"  Train images: {len(train_images)} -> {train_img_dir}")
    print(f"  Val images:   {len(val_images)} -> {val_img_dir}")
    print(f"  Empty instances (0):  {empty_class_count}")
    print(f"  Filled instances (1): {filled_class_count}")
    if missing_labels > 0:
        print(f"\n[WARNING] {missing_labels} image(s) had no corresponding .txt file in {raw_labels_dir}.")
        print("          Empty .txt files were created. Make sure you labeled your images in Roboflow/LabelImg.")
    print("\n[READY] You can now run 3_train.py to train YOLOv8n.")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and val subsets.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    split_dataset(train_ratio=args.ratio, seed=args.seed)
