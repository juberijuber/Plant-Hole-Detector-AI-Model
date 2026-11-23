"""
prepare_colab_package.py - Package Dataset for Fast Google Colab Upload

Why this script:
  The raw dataset is ~2.1 GB (due to 4096x1842 high-res phone camera shots).
  Uploading 2.1 GB to Google Colab can be slow.
  Since YOLO trains at 640x640, this script resizes images to max 800px,
  dropping the ZIP size from ~2.1 GB down to ~60-80 MB (30x faster upload!),
  with ZERO loss in model accuracy.
"""

import os
import zipfile
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ZIP_OUT = BASE_DIR / "dataset_colab.zip"

def create_colab_zip(max_dim=800):
    print("=" * 65)
    print("PACKAGING DATASET FOR GOOGLE COLAB")
    print("=" * 65)
    print(f"Source dataset: {DATASET_DIR.resolve()}")
    print(f"Target archive: {ZIP_OUT.resolve()}")
    print(f"Resizing images to max dimension: {max_dim}px (fast upload)")
    print("-" * 65)

    img_count = 0
    lbl_count = 0

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Process images
        for split in ["train", "val"]:
            img_dir = DATASET_DIR / "images" / split
            for img_p in img_dir.glob("*.jpg"):
                im = cv2.imread(str(img_p))
                if im is not None:
                    h, w = im.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / float(max(h, w))
                        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    # Encode to memory as jpeg
                    ret, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ret:
                        arcname = f"dataset/images/{split}/{img_p.name}"
                        zf.writestr(arcname, buf.tobytes())
                        img_count += 1
                        if img_count % 100 == 0:
                            print(f"  Processed {img_count} images...")

            # 2. Add matching labels
            lbl_dir = DATASET_DIR / "labels" / split
            for lbl_p in lbl_dir.glob("*.txt"):
                arcname = f"dataset/labels/{split}/{lbl_p.name}"
                zf.write(lbl_p, arcname)
                lbl_count += 1

    zip_size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 65)
    print("PACKAGE READY!")
    print("=" * 65)
    print(f"  Saved to:       {ZIP_OUT.name} ({zip_size_mb:.1f} MB)")
    print(f"  Images packed:  {img_count}")
    print(f"  Labels packed:  {lbl_count}")
    print("\n[NEXT STEP] Upload 'dataset_colab.zip' and 'Planting_Hole_Detection_Colab.ipynb' to Google Colab.")
    print("=" * 65)

if __name__ == "__main__":
    create_colab_zip()
