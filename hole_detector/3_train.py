"""
3_train.py - YOLOv8n Model Training Script for Planting Hole Detection

Train YOLOv8n detector with 2 classes:
  0 = empty
  1 = filled

Default setup: 80 epochs, imgsz 640. Works locally on laptop, Jetson (if PyTorch is installed),
or on Google Colab GPU.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Project configuration
import config

def train_model(epochs=80, imgsz=640, batch=16, device="", export_onnx=True, base_model="yolov5nu.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n" + "=" * 60)
        print("[ERROR] 'ultralytics' package is not installed.")
        print("Please install requirements using:")
        print("    pip install -r requirements.txt")
        print("=" * 60 + "\n")
        sys.exit(1)

    data_yaml_path = config.BASE_DIR / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {data_yaml_path.resolve()}")
        sys.exit(1)

    # Check if dataset images exist
    train_images_dir = config.DATASET_DIR / "images" / "train"
    if not train_images_dir.exists() or len(list(train_images_dir.glob("*.*"))) == 0:
        print("\n" + "=" * 60)
        print("[ERROR] No training images found in dataset/images/train/!")
        print("Please complete the following steps first:")
        print("  1. Run 'python 1_capture.py' to capture images.")
        print("  2. Label them (Roboflow or LabelImg in YOLO format).")
        print("  3. Run 'python 2_split_dataset.py' to prepare the dataset.")
        print("  (Tip: Run 'python generate_synthetic_data.py' to test with sample data immediately!)")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Normalize model name for YOLOv5n
    if base_model.lower() in ["yolov5n", "yolov5n.pt"]:
        base_model = "yolov5nu.pt"

    run_name = f"planting_hole_{Path(base_model).stem}"

    print("\n" + "=" * 60)
    print(f"STARTING MODEL TRAINING: {base_model.upper()}")
    print("=" * 60)
    print(f"  Base Model:         {base_model}")
    print(f"  Configuration file: {data_yaml_path.resolve()}")
    print(f"  Epochs:             {epochs}")
    print(f"  Image size:         {imgsz}")
    print(f"  Batch size:         {batch}")
    print(f"  Device:             {device if device else 'Auto-detect'}")
    print("=" * 60 + "\n")

    # Load baseline model (YOLOv8n or YOLOv5nu)
    model = YOLO(base_model)

    # Train model
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=str(config.BASE_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        degrees=15.0,        # rotation +/- 15 deg
        perspective=0.0005,  # perspective jitter for angled views
        shear=5.0,           # shear +/- 5 deg
        fliplr=0.5,          # horizontal flip
        scale=0.5            # scale +/- 50% for depth/distance invariance
    )

    # Best weights path
    runs_best_pt = config.BASE_DIR / "runs" / run_name / "weights" / "best.pt"

    if runs_best_pt.exists():
        # Copy best.pt to models/ directory for easy deployment
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target_pt = config.MODEL_PATH
        shutil.copy2(runs_best_pt, target_pt)
        print(f"\n[SUCCESS] Best model weights saved to: {target_pt.resolve()}")

        # Export to ONNX if requested (faster inference on Jetson Nano / Raspberry Pi)
        if export_onnx:
            try:
                print("\n[INFO] Exporting trained model to ONNX for edge optimization...")
                exported_onnx = model.export(format="onnx", imgsz=imgsz, simplify=True)
                if Path(exported_onnx).exists():
                    shutil.copy2(exported_onnx, config.ONNX_MODEL_PATH)
                    print(f"[SUCCESS] ONNX model ready at: {config.ONNX_MODEL_PATH.resolve()}")
            except Exception as e:
                print(f"[WARNING] ONNX export skipped or encountered an error: {e}")
                print("Standard PyTorch .pt model is still ready and usable.")
    else:
        print("[WARNING] Could not find runs/planting_hole_yolov8n/weights/best.pt.")

    print("\n[NEXT STEP] Run 'python 4_live_detect.py' to run live detection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for planting hole detection.")
    parser.add_argument("--model", type=str, default="yolov5nu.pt",
                        help="Base model to train (default: 'yolov5nu.pt' / YOLOv5n)")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="", help="Device: '0', '0,1', 'cpu'")
    parser.add_argument("--no-onnx", action="store_true", help="Skip automatic ONNX export")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        export_onnx=not args.no_onnx,
        base_model=args.model
    )
