"""
1_capture.py - Collect Images for Planting Hole Dataset

Instructions:
  1. Position your DOFBOT in the fixed SCAN_POSE (or mount camera overhead).
  2. Run this script to view the live camera feed.
  3. Press [SPACE] to capture an image.
  4. Rearrange plugs (e.g. all filled, some empty, all empty, vary lighting).
  5. Press [Q] to quit when finished.
"""

import sys
import os
import time
import argparse
from pathlib import Path
import cv2

# Import project configuration
import config

def get_camera_capture():
    """Initializes OpenCV VideoCapture with fallback diagnostics."""
    if config.USE_CSI_CAMERA:
        print("[INFO] Initializing Jetson CSI Camera with GStreamer pipeline...")
        cap = cv2.VideoCapture(config.CSI_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        print(f"[INFO] Initializing USB / Standard Camera (Index: {config.CAMERA_INDEX})...")
        # On Windows, cv2.CAP_DSHOW can prevent delay
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    if not cap.isOpened():
        print("\n" + "=" * 60)
        print("[ERROR] Could not open camera!")
        print("Troubleshooting checklist:")
        print(f"  1. Is camera connected? Check USB cable or ribbon cable.")
        print(f"  2. Current index is {config.CAMERA_INDEX}. Try changing CAMERA_INDEX in config.py (e.g. 1 or 2).")
        print(f"  3. On Linux/Jetson, check permissions: sudo usermod -aG video $USER")
        print(f"  4. On Windows, make sure no other app (e.g. Camera app, Zoom) is using it.")
        print("=" * 60 + "\n")
        return None
    return cap


def main():
    parser = argparse.ArgumentParser(description="Capture dataset images for planting hole detection.")
    parser.add_argument("--out-dir", type=str, default=str(config.RAW_DATASET_DIR / "images"),
                        help="Target directory to save raw captured images.")
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (captures frames automatically at intervals).")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between automatic captures in headless mode.")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Max images to capture in headless mode.")
    args = parser.parse_args()

    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = get_camera_capture()
    if cap is None:
        sys.exit(1)

    print(f"\n[READY] Saving images to: {save_dir.resolve()}")
    if not args.headless:
        print("[CONTROLS] Press [SPACE] to capture an image | Press [Q] or [ESC] to exit.\n")
    else:
        print(f"[HEADLESS] Automatic capture every {args.interval}s up to {args.max_images} images.\n")

    saved_count = 0
    last_auto_capture_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from camera. Retrying...")
                time.sleep(0.1)
                continue

            display_frame = frame.copy()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hole_img_{timestamp}_{saved_count:04d}.jpg"
            file_path = save_dir / filename

            should_save = False

            if args.headless:
                now = time.time()
                if now - last_auto_capture_time >= args.interval:
                    should_save = True
                    last_auto_capture_time = now
            else:
                # Add on-screen UI guide
                cv2.rectangle(display_frame, (0, 0), (config.FRAME_WIDTH, 40), (40, 40, 40), -1)
                status_text = f"Captured: {saved_count} | [SPACE] Save | [Q] Quit"
                cv2.putText(display_frame, status_text, (15, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("DOFBOT Dataset Capture", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # 27 is ESC
                    print("\n[INFO] Capture session stopped by user.")
                    break
                elif key == 32:  # SPACE
                    should_save = True

            if should_save:
                cv2.imwrite(str(file_path), frame)
                saved_count += 1
                print(f"[{saved_count:03d}] Saved: {file_path.name}")

                if not args.headless:
                    # Flash visual feedback
                    flash_overlay = display_frame.copy()
                    cv2.putText(flash_overlay, "SAVED!", (int(config.FRAME_WIDTH/2) - 60, int(config.FRAME_HEIGHT/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.imshow("DOFBOT Dataset Capture", flash_overlay)
                    cv2.waitKey(100)

                if args.headless and saved_count >= args.max_images:
                    print(f"[INFO] Reached maximum target of {args.max_images} images.")
                    break

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"\n[DONE] Finished. Total images saved: {saved_count}")
        print(f"[NEXT STEP] Label the images in Roboflow or LabelImg, then place them in raw_dataset/labels.")

if __name__ == "__main__":
    main()
