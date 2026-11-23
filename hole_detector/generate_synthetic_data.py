"""
generate_synthetic_data.py - Synthetic Dataset Generator for Testing

Generates 16 realistic synthetic hydroponic tray images (with 4 planting holes)
and their corresponding YOLOv8 annotation labels:
  - Class 0: empty (white cup with dark center or bare hole)
  - Class 1: filled (peat/coco plug with green sprout)

This allows testing the complete pipeline (2_split_dataset.py, 3_train.py, 4_live_detect.py)
immediately without needing physical hardware.
"""

import os
import random
from pathlib import Path
import numpy as np
import cv2

import config

def create_synthetic_image(image_idx, w=640, h=480):
    # Background: hydroponic tray (light gray / off-white plastic texture)
    img = np.full((h, w, 3), random.randint(215, 235), dtype=np.uint8)

    # Add subtle lighting gradient / noise
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4 Hole positions arranged in a horizontal row or 2x2 grid
    # Let's arrange 4 holes horizontally across the tray
    margin_x = 80
    spacing_x = (w - 2 * margin_x) // 3
    center_y = h // 2 + random.randint(-15, 15)

    radius = random.randint(40, 48)
    boxes_yolo = []

    # Decide state for each hole: randomly filled or empty
    for i in range(4):
        cx = margin_x + i * spacing_x + random.randint(-8, 8)
        cy = center_y + random.randint(-6, 6)

        # Class: 0 = empty, 1 = filled
        # Make sure at least one is empty and one is filled across images
        if image_idx % 4 == i:
            cls_id = 0  # Empty
        elif (image_idx + 1) % 4 == i:
            cls_id = 1  # Filled
        else:
            cls_id = random.choice([0, 1])

        # Outer rim of hole (black/dark plastic net cup rim)
        cv2.circle(img, (cx, cy), radius + 4, (40, 40, 40), 3)

        if cls_id == 0:
            # EMPTY HOLE:
            # Variant A: bare hole (dark background beneath tray)
            # Variant B: white net cup with empty dark cavity
            if random.random() > 0.5:
                # White net cup rim
                cv2.circle(img, (cx, cy), radius, (240, 240, 240), -1)
                cv2.circle(img, (cx, cy), radius - 8, (35, 35, 35), -1)
            else:
                # Bare dark circular hole
                cv2.circle(img, (cx, cy), radius, (25, 25, 25), -1)
        else:
            # FILLED HOLE:
            # Net cup with brown peat/coco plug
            brown_color = (random.randint(25, 45), random.randint(55, 80), random.randint(90, 130))
            cv2.circle(img, (cx, cy), radius, brown_color, -1)

            # Peat texture spots
            for _ in range(30):
                px = cx + random.randint(-radius + 10, radius - 10)
                py = cy + random.randint(-radius + 10, radius - 10)
                cv2.circle(img, (px, py), random.randint(2, 5), (15, 35, 55), -1)

            # Small green seedling leaves in center
            cv2.circle(img, (cx - 4, cy - 2), random.randint(5, 9), (30, 160, 40), -1)
            cv2.circle(img, (cx + 6, cy + 3), random.randint(4, 8), (40, 180, 50), -1)

        # Bounding box in YOLO normalized format: [class_id, x_center, y_center, width, height]
        box_w = (radius * 2 + 10) / w
        box_h = (radius * 2 + 10) / h
        norm_cx = cx / w
        norm_cy = cy / h

        boxes_yolo.append((cls_id, norm_cx, norm_cy, box_w, box_h))

    return img, boxes_yolo


def main():
    num_samples = 16
    raw_img_dir = config.RAW_DATASET_DIR / "images"
    raw_lbl_dir = config.RAW_DATASET_DIR / "labels"

    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_lbl_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"GENERATING {num_samples} SYNTHETIC PLANTING TRAY IMAGES & LABELS")
    print("=" * 60)

    for i in range(num_samples):
        img, boxes = create_synthetic_image(i)
        stem = f"synthetic_tray_{i+1:03d}"
        img_file = raw_img_dir / f"{stem}.jpg"
        lbl_file = raw_lbl_dir / f"{stem}.txt"

        cv2.imwrite(str(img_file), img)

        with open(lbl_file, "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        print(f"Generated sample: {img_file.name} (4 holes labeled)")

    print("\n[SUCCESS] Synthetic dataset ready in raw_dataset/.")
    print("Run 'python 2_split_dataset.py' next to split into train/val.")

if __name__ == "__main__":
    main()
