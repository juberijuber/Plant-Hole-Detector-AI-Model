"""
6_fallback_roi.py - Non-AI Fallback: Fixed ROI + HSV Color Ratio Detection

Use this if you haven't trained a YOLO model yet, or need an ultra-lightweight
rule-based detector that runs at maximum speed on Raspberry Pi / Jetson Nano.

Workflow:
  1. Calibration (--calibrate):
       Draw bounding boxes around each hole once with cv2.selectROI.
       Saved to roi_config.json.
  2. Live Detection:
       Crops each hole ROI, extracts brown peat plug pixels using HSV thresholding,
       and classifies:
         - Ratio >= FILLED_RATIO_THRESHOLD -> FILLED (Green)
         - Ratio <  FILLED_RATIO_THRESHOLD -> EMPTY (Red)
       Shows "Hole N: LABEL" and summary line "Empty holes: [...]".
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

import config


def get_camera():
    if config.USE_CSI_CAMERA:
        cap = cv2.VideoCapture(config.CSI_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera (Index: {config.CAMERA_INDEX}).")
        return None
    return cap


def calibrate_rois(roi_path=config.ROI_CONFIG_FILE):
    """Interactive tool to select bounding boxes for Hole 1..N using cv2.selectROI."""
    cap = get_camera()
    if cap is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ROI CALIBRATION MODE")
    print("=" * 60)
    print("1. An image frame will be frozen.")
    print("2. For each hole, click and drag a bounding box.")
    print("3. Press [SPACE] or [ENTER] to confirm each selection.")
    print("4. Press [ESC] or [C] when all holes have been defined.")
    print("=" * 60 + "\n")

    # Read a few frames to let camera auto-exposure stabilize
    for _ in range(15):
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to read frame for calibration.")
        sys.exit(1)

    rois = []
    hole_num = 1

    while True:
        display = frame.copy()
        # Draw already confirmed ROIs
        for idx, (rx, ry, rw, rh) in enumerate(rois, start=1):
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(display, f"Hole {idx}", (rx, max(15, ry - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        window_name = f"Select Hole {hole_num} (Drag box, then SPACE. ESC when done)"
        r = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        rx, ry, rw, rh = r
        if rw == 0 or rh == 0:
            print(f"[INFO] Finished ROI selection. Total holes defined: {len(rois)}")
            break

        rois.append([int(rx), int(ry), int(rw), int(rh)])
        print(f"[OK] Added Hole {hole_num}: box = [x={rx}, y={ry}, w={rw}, h={rh}]")
        hole_num += 1

    if not rois:
        print("[WARNING] No ROIs were defined. Exiting.")
        return

    # Save to config file
    with open(roi_path, "w") as f:
        json.dump({"rois": rois}, f, indent=2)

    print(f"\n[SAVED] ROI configuration written to {roi_path.resolve()}")
    print("You can now run 'python 6_fallback_roi.py' for live detection.")


def run_fallback_detection(headless=False):
    roi_path = config.ROI_CONFIG_FILE
    if not roi_path.exists():
        print(f"[ERROR] ROI configuration file not found: {roi_path.resolve()}")
        print("Please run calibration first: python 6_fallback_roi.py --calibrate")
        sys.exit(1)

    with open(roi_path, "r") as f:
        data = json.load(f)
    rois = data.get("rois", [])

    if not rois:
        print(f"[ERROR] No ROIs found in {roi_path.resolve()}. Run --calibrate first.")
        sys.exit(1)

    cap = get_camera()
    if cap is None:
        sys.exit(1)

    print(f"\n[INFO] Loaded {len(rois)} calibrated hole ROIs.")
    print("[INFO] Fallback HSV brown-ratio detection active.")
    if headless:
        print("[MODE] Running HEADLESS (press Ctrl+C to stop).")
    else:
        print("[CONTROLS] Press [Q] or [ESC] to quit.")
    print("-" * 65)

    lower_brown = np.array(config.HSV_BROWN_LOWER, dtype=np.uint8)
    upper_brown = np.array(config.HSV_BROWN_UPPER, dtype=np.uint8)

    prev_time = time.time()
    last_print_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            overlay = frame.copy()

            empty_holes = []
            hole_statuses = []

            for idx, (x, y, w, h) in enumerate(rois, start=1):
                # Clamp coordinates to frame
                x1 = max(0, min(x, frame.shape[1] - 1))
                y1 = max(0, min(y, frame.shape[0] - 1))
                x2 = max(0, min(x + w, frame.shape[1]))
                y2 = max(0, min(y + h, frame.shape[0]))

                roi_hsv = hsv[y1:y2, x1:x2]
                if roi_hsv.size == 0:
                    continue

                # Threshold for brown peat/coco plug
                mask = cv2.inRange(roi_hsv, lower_brown, upper_brown)
                brown_pixel_count = cv2.countNonZero(mask)
                total_pixels = (x2 - x1) * (y2 - y1)
                ratio = brown_pixel_count / max(1, total_pixels)

                is_filled = ratio >= config.FILLED_RATIO_THRESHOLD
                cls_name = "FILLED" if is_filled else "EMPTY"
                color = config.CLASS_COLORS[1] if is_filled else config.CLASS_COLORS[0]

                if not is_filled:
                    empty_holes.append(idx)
                hole_statuses.append(f"Hole {idx}: {cls_name} ({ratio*100:.1f}%)")

                if not headless:
                    # Draw box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    label = f"Hole {idx}: {cls_name}"
                    cv2.putText(overlay, label, (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            if not headless:
                # Top summary banner
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 42), (30, 30, 30), -1)
                empty_str = str(sorted(empty_holes)) if empty_holes else "None"
                banner_text = f"Empty holes: {empty_str}  |  FPS: {fps:.1f} (Fallback HSV)"
                banner_color = (0, 0, 255) if empty_holes else (0, 255, 0)
                cv2.putText(overlay, banner_text, (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2, cv2.LINE_AA)

                cv2.imshow("DOFBOT Fallback ROI Detection", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break

            # Headless print
            if curr_time - last_print_time >= 1.0:
                print(f"[FPS: {fps:4.1f}] Empty holes: {sorted(empty_holes)} | {', '.join(hole_statuses)}")
                last_print_time = curr_time

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-AI Fallback ROI Planting Hole Detector.")
    parser.add_argument("--calibrate", action="store_true", help="Interactively define hole bounding boxes")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_rois()
    else:
        run_fallback_detection(headless=args.headless)
