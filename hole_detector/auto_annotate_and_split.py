"""
auto_annotate_and_split.py - Automatically Annotate and Split User Dataset

Classification Rule (Updated per user specification):
  - Class 0: EMPTY  -> ONLY when there is NOTHING in the hole (bare empty hole).
  - Class 1: FILLED -> When a bucket / net cup is present in the hole
                       (whether the bucket is empty OR contains a peat/plant plug).
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np
import cv2

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "Dataset for hole detection"
RAW_IMG_DIR = BASE_DIR / "raw_dataset" / "images"
RAW_LBL_DIR = BASE_DIR / "raw_dataset" / "labels"
DATASET_DIR = BASE_DIR / "dataset"

# Clean up raw_dataset target folders
RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
RAW_LBL_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth mapping based on:
# 0 = EMPTY (Nothing in hole / bare hole)
# 1 = FILLED (Bucket/cup present, with or without plug)
BURST_CONFIGS = {
    # Empty tray bursts (both holes are bare holes with NOTHING inside)
    "IMG_20261120_114657452": {"type": "channel", "left": 0, "right": 0},
    "IMG_20261120_114915018": {"type": "channel", "left": 0, "right": 0},
    "IMG_20261120_114927243": {"type": "channel", "left": 0, "right": 0},

    # Channel bursts where 1 hole has a bucket and 1 hole is bare empty:
    "IMG_20261120_114752614": {"type": "channel", "left": 1, "right": 0}, # left=black bucket (1), right=bare empty (0)
    "IMG_20261120_114816163": {"type": "channel", "left": 1, "right": 0}, # left=black bucket (1), right=bare empty (0)
    "IMG_20261120_114830286": {"type": "channel", "left": 1, "right": 0}, # left=white bucket (1), right=bare empty (0)

    # Channel bursts where both holes have buckets (empty bucket or plug):
    "IMG_20261120_114729760": {"type": "channel", "left": 1, "right": 1}, # left=white bucket (1), right=bucket+plug (1)
    "IMG_20261120_114847590": {"type": "channel", "left": 1, "right": 1}, # left=white bucket (1), right=bucket+plug (1)

    # Buckets on table (all have bucket present -> Class 1: FILLED)
    "IMG_20261120_114627644": {"type": "single", "cls": 1},          # White bucket -> FILLED (1)
    "IMG_20261120_114509359": {"type": "single", "cls": 1},          # Bucket with peat -> FILLED (1)
    "IMG_20261120_114557173": {"type": "single", "cls": 1},          # Bucket with peat -> FILLED (1)
    "IMG_20261120_114222132": {"type": "double_vertical", "cls": 1}, # Two buckets -> FILLED (1)
    "IMG_20261120_114238542": {"type": "channel", "left": 1, "right": 1}, # Two buckets -> FILLED (1)
    "IMG_20261120_114249999": {"type": "channel", "left": 1, "right": 1}, # Two buckets -> FILLED (1)
}


def detect_channel_circles(im):
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(w * 0.25),
        param1=50, param2=30, minRadius=int(h * 0.12), maxRadius=int(h * 0.30)
    )
    left_c = None
    right_c = None

    if circles is not None:
        c = np.round(circles[0, :]).astype("int")
        left_candidates = [pt for pt in c if 0.10 * w <= pt[0] <= 0.45 * w and 0.2 * h <= pt[1] <= 0.8 * h]
        right_candidates = [pt for pt in c if 0.55 * w <= pt[0] <= 0.90 * w and 0.2 * h <= pt[1] <= 0.8 * h]

        if left_candidates:
            left_c = max(left_candidates, key=lambda pt: pt[2])
        if right_candidates:
            right_c = max(right_candidates, key=lambda pt: pt[2])

    # Fallback to standard tray geometry if one or both circles missed by Hough
    if left_c is not None and right_c is None:
        right_c = (int(left_c[0] + 0.43 * w), left_c[1], left_c[2])
    elif right_c is not None and left_c is None:
        left_c = (int(right_c[0] - 0.43 * w), right_c[1], right_c[2])
    elif left_c is None and right_c is None:
        left_c = (int(0.26 * w), int(0.50 * h), int(0.21 * h))
        right_c = (int(0.69 * w), int(0.50 * h), int(0.21 * h))

    return left_c, right_c


def circle_to_yolo_bbox(cx, cy, r, w, h, padding=1.05):
    """Converts circle center and radius to normalized YOLO format (x_c y_c bw bh)."""
    box_w = min(w, 2 * r * padding)
    box_h = min(h, 2 * r * padding)
    norm_x = cx / w
    norm_y = cy / h
    norm_w = box_w / w
    norm_h = box_h / h
    return norm_x, norm_y, norm_w, norm_h


def process_dataset():
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        return

    all_images = list(SOURCE_DIR.rglob("*.jpg"))
    print(f"[INFO] Found {len(all_images)} total images in {SOURCE_DIR.name}.")

    empty_box_count = 0
    filled_box_count = 0
    annotated_images = []

    for img_path in sorted(all_images):
        filename = img_path.name
        stem = img_path.stem
        burst_prefix = filename.split("_BURST")[0]

        cfg = BURST_CONFIGS.get(burst_prefix)
        if cfg is None:
            is_empty_folder = "empty" in str(img_path.parent).lower()
            cfg = {"type": "channel", "left": 0 if is_empty_folder else 1, "right": 0}

        im = cv2.imread(str(img_path))
        if im is None:
            continue

        h, w = im.shape[:2]
        boxes_yolo = []

        if cfg["type"] == "channel":
            left_c, right_c = detect_channel_circles(im)
            # Left hole
            lx, ly, lw, lh = circle_to_yolo_bbox(left_c[0], left_c[1], left_c[2], w, h)
            boxes_yolo.append((cfg["left"], lx, ly, lw, lh))
            # Right hole
            rx, ry, rw, rh = circle_to_yolo_bbox(right_c[0], right_c[1], right_c[2], w, h)
            boxes_yolo.append((cfg["right"], rx, ry, rw, rh))

        elif cfg["type"] == "single":
            cls_id = cfg["cls"]
            cx, cy = int(w * 0.50), int(h * 0.45)
            r = int(min(w, h) * 0.35)
            nx, ny, nw, nh = circle_to_yolo_bbox(cx, cy, r, w, h)
            boxes_yolo.append((cls_id, nx, ny, nw, nh))

        elif cfg["type"] == "double_vertical":
            cls_id = cfg["cls"]
            # Top cup
            c1_x, c1_y, c1_r = int(w * 0.45), int(h * 0.32), int(w * 0.30)
            nx1, ny1, nw1, nh1 = circle_to_yolo_bbox(c1_x, c1_y, c1_r, w, h)
            boxes_yolo.append((cls_id, nx1, ny1, nw1, nh1))
            # Bottom cup
            c2_x, c2_y, c2_r = int(w * 0.48), int(h * 0.63), int(w * 0.32)
            nx2, ny2, nw2, nh2 = circle_to_yolo_bbox(c2_x, c2_y, c2_r, w, h)
            boxes_yolo.append((cls_id, nx2, ny2, nw2, nh2))

        # Copy image to raw_dataset/images/
        target_img_path = RAW_IMG_DIR / filename
        shutil.copy2(img_path, target_img_path)

        # Write YOLO label .txt file
        target_lbl_path = RAW_LBL_DIR / f"{stem}.txt"
        with open(target_lbl_path, "w") as f:
            for b in boxes_yolo:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                if b[0] == 0:
                    empty_box_count += 1
                else:
                    filled_box_count += 1

        annotated_images.append(target_img_path)

    print(f"\n[INFO] Generated YOLO annotations for {len(annotated_images)} images.")
    print(f"       Class 0 (EMPTY - bare hole):          {empty_box_count}")
    print(f"       Class 1 (FILLED - bucket/plug present): {filled_box_count}")

    # --------------------------------------------------------------------------
    # Split into 80% train and 20% val
    # --------------------------------------------------------------------------
    random.seed(42)
    shuffled = annotated_images.copy()
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

    print("\n[INFO] Copying files to 80/20 train/val dataset...")
    for img_p in train_set:
        shutil.copy2(img_p, train_img_dir / img_p.name)
        lbl_p = RAW_LBL_DIR / f"{img_p.stem}.txt"
        shutil.copy2(lbl_p, train_lbl_dir / lbl_p.name)

    for img_p in val_set:
        shutil.copy2(img_p, val_img_dir / img_p.name)
        lbl_p = RAW_LBL_DIR / f"{img_p.stem}.txt"
        shutil.copy2(lbl_p, val_lbl_dir / lbl_p.name)

    # Update data.yaml
    data_yaml_path = BASE_DIR / "data.yaml"
    data_yaml_content = f"""# YOLOv8 Dataset Configuration
path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: empty
  1: filled
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)

    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETE (UPDATED RULE)!")
    print("=" * 65)
    print(f"  Total Images Processed: {len(annotated_images)}")
    print(f"  Training Set (80%):     {len(train_set)} images")
    print(f"  Validation Set (20%):   {len(val_set)} images")
    print(f"  Class 0 (EMPTY - bare hole):          {empty_box_count} boxes")
    print(f"  Class 1 (FILLED - bucket/plug present): {filled_box_count} boxes")
    print(f"  Data Config File:       {data_yaml_path.resolve()}")
    print("=" * 65)
    print("\n[READY] You can now train YOLOv8n with:")
    print("  python 3_train.py --epochs 80 --imgsz 640")


if __name__ == "__main__":
    process_dataset()
