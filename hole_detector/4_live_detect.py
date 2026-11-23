"""
4_live_detect.py - Live Planting Hole Detection, Spatial Numbering & Overlay

Key Capabilities:
  1. Detects EMPTY (class 0) and FILLED (class 1) planting holes.
  2. Numbers holes in spatial order:
       - Groups into rows by y (within ROW_TOLERANCE_PX).
       - Sorts rows from top to bottom.
       - Sorts holes within each row from left to right.
  3. Stable numbering & majority-vote smoothing across frames to prevent flicker.
  4. Live OpenCV overlay showing:
       - "Hole N: FILLED" in Green
       - "Hole N: EMPTY" in Red
       - Summary line: "Empty holes: [4]"
       - Real-time FPS counter
  5. --headless mode for running over SSH without a display.
  6. Jetson Nano / Raspberry Pi friendly: supports small imgsz and ONNX runtime.
"""

import sys
import time
import argparse
from collections import deque, Counter
from pathlib import Path
import numpy as np
import cv2

# Import configuration
import config


class HoleTracker:
    """
    Tracks hole detections across frames using spatial proximity and
    smooths classification (EMPTY vs FILLED) via majority vote over N frames.
    """
    def __init__(self, window_size=5, max_dist=50):
        self.window_size = window_size
        self.max_dist = max_dist
        # Dict mapping stable_id -> deque of past class predictions
        self.history = {}
        # Dict mapping stable_id -> last known centroid (cx, cy)
        self.centroids = {}
        self.next_id = 1

    def update(self, raw_detections):
        """
        raw_detections: list of dicts: [{'box': (x1, y1, x2, y2), 'cx': cx, 'cy': cy, 'cls': int, 'conf': float}]
        Returns smoothed detections with stable classifications.
        """
        if not raw_detections:
            return []

        updated_detections = []
        unmatched_raw = list(range(len(raw_detections)))
        matched_ids = set()

        # Match current detections with existing centroids by Euclidean distance
        for sid, last_pt in list(self.centroids.items()):
            best_idx = None
            best_dist = float('inf')
            for idx in unmatched_raw:
                det = raw_detections[idx]
                dist = np.hypot(det['cx'] - last_pt[0], det['cy'] - last_pt[1])
                if dist < self.max_dist and dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                det = raw_detections[best_idx]
                unmatched_raw.remove(best_idx)
                matched_ids.add(sid)
                self.centroids[sid] = (det['cx'], det['cy'])
                self.history[sid].append(det['cls'])
                # Majority vote for class
                voted_cls = Counter(self.history[sid]).most_common(1)[0][0]
                det['tracked_id'] = sid
                det['smoothed_cls'] = voted_cls
                updated_detections.append(det)

        # Assign new IDs to remaining unmatched detections
        for idx in unmatched_raw:
            det = raw_detections[idx]
            sid = self.next_id
            self.next_id += 1
            self.centroids[sid] = (det['cx'], det['cy'])
            self.history[sid] = deque([det['cls']], maxlen=self.window_size)
            det['tracked_id'] = sid
            det['smoothed_cls'] = det['cls']
            updated_detections.append(det)

        # Cleanup lost tracks
        all_sids = list(self.centroids.keys())
        active_ids = {d['tracked_id'] for d in updated_detections}
        for sid in all_sids:
            if sid not in active_ids:
                del self.centroids[sid]
                del self.history[sid]

        return updated_detections


def sort_holes_spatially(detections, row_tol=config.ROW_TOLERANCE_PX):
    """
    Sorts detections into reading order:
      1. Group into rows by y coordinate using row_tol.
      2. Sort rows from top to bottom (ascending y).
      3. Sort holes within each row from left to right (ascending x).
      4. Assign hole_number (1..N).
    """
    if not detections:
        return []

    # Sort all initially by cy
    sorted_by_y = sorted(detections, key=lambda d: d['cy'])

    rows = []
    current_row = [sorted_by_y[0]]

    for det in sorted_by_y[1:]:
        # Calculate mean y of current row
        row_mean_y = sum(d['cy'] for d in current_row) / len(current_row)
        if abs(det['cy'] - row_mean_y) <= row_tol:
            current_row.append(det)
        else:
            rows.append(current_row)
            current_row = [det]
    if current_row:
        rows.append(current_row)

    # Sort each row left-to-right (ascending cx)
    ordered_detections = []
    hole_counter = 1
    for row in rows:
        row_sorted = sorted(row, key=lambda d: d['cx'])
        for det in row_sorted:
            det['hole_num'] = hole_counter
            hole_counter += 1
            ordered_detections.append(det)

    return ordered_detections


def open_camera():
    """Opens video capture with diagnostic error messages."""
    if config.USE_CSI_CAMERA:
        print("[INFO] Attempting to open Jetson CSI camera...")
        cap = cv2.VideoCapture(config.CSI_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        print(f"[INFO] Attempting to open camera index {config.CAMERA_INDEX}...")
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("\n" + "=" * 60)
        print("[ERROR] Failed to open video feed.")
        print(f"Please check that camera index {config.CAMERA_INDEX} is correct.")
        print("Tip: If running on Jetson Nano with CSI camera, set USE_CSI_CAMERA = True in config.py.")
        print("=" * 60 + "\n")
        return None
    return cap


def draw_visual_overlay(frame, detections, fps):
    """Draws bounding boxes, Hole N labels, and top summary banner."""
    overlay = frame.copy()
    h, w, _ = frame.shape

    empty_holes = []
    filled_holes = []

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls_id = det.get('smoothed_cls', det['cls'])
        hole_num = det.get('hole_num', '?')
        cls_name = config.CLASS_NAMES.get(cls_id, "UNKNOWN")
        color = config.CLASS_COLORS.get(cls_id, (255, 255, 255))

        if cls_id == 0:
            empty_holes.append(hole_num)
        else:
            filled_holes.append(hole_num)

        # Draw box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label pill
        label = f"Hole {hole_num}: {cls_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        pill_y1 = max(0, y1 - text_h - 10)
        pill_y2 = y1
        cv2.rectangle(overlay, (x1, pill_y1), (x1 + text_w + 10, pill_y2), color, -1)
        cv2.putText(overlay, label, (x1 + 5, pill_y2 - 5),
                    font, font_scale, (0, 0, 0) if cls_id == 1 else (255, 255, 255),
                    thickness, cv2.LINE_AA)

    # Top summary banner
    cv2.rectangle(overlay, (0, 0), (w, 42), (30, 30, 30), -1)

    empty_str = str(sorted(empty_holes)) if empty_holes else "None"
    banner_text = f"Empty holes: {empty_str}  |  Total: {len(detections)}  |  FPS: {fps:.1f}"
    banner_color = (0, 0, 255) if empty_holes else (0, 255, 0)

    cv2.putText(overlay, banner_text, (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2, cv2.LINE_AA)

    return overlay, empty_holes


def run_detection(headless=False, imgsz=config.INFERENCE_IMGSZ, use_onnx=False, conf_thresh=config.CONFIDENCE_THRESHOLD):
    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    weights_to_load = None
    if use_onnx and config.ONNX_MODEL_PATH.exists():
        weights_to_load = str(config.ONNX_MODEL_PATH)
        print(f"[INFO] Using ONNX model: {weights_to_load}")
    elif config.MODEL_PATH.exists():
        weights_to_load = str(config.MODEL_PATH)
        print(f"[INFO] Using trained PyTorch weights: {weights_to_load}")
    else:
        weights_to_load = config.FALLBACK_BASE_MODEL
        print(f"[WARNING] Trained model '{config.MODEL_PATH}' not found.")
        print(f"[INFO] Falling back to base '{weights_to_load}'. Run 3_train.py to train your model!")

    model = YOLO(weights_to_load)
    cap = open_camera()
    if cap is None:
        sys.exit(1)

    tracker = HoleTracker(window_size=config.SMOOTHING_WINDOW_FRAMES,
                          max_dist=config.TRACKING_MAX_DISTANCE_PX)

    print("\n[READY] Live Planting Hole Detection Started.")
    print(f"[CONFIG] Confidence Threshold: {conf_thresh:.2f} | Image Size: {imgsz}")
    if headless:
        print("[MODE] Running HEADLESS (console output only, press Ctrl+C to stop).")
    else:
        print("[CONTROLS] Press [Q] or [ESC] on the video window to quit.")
    print("-" * 65)

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

            # Run YOLO inference
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                iou=config.IOU_THRESHOLD,
                imgsz=imgsz,
                verbose=False
            )

            # Parse bounding boxes
            raw_detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    raw_detections.append({
                        'box': (x1, y1, x2, y2),
                        'cx': cx,
                        'cy': cy,
                        'cls': cls_id,
                        'conf': conf
                    })

            # Temporal smoothing
            smoothed_detections = tracker.update(raw_detections)

            # Spatial ordering (Row by row, Left-to-Right)
            ordered_detections = sort_holes_spatially(smoothed_detections, row_tol=config.ROW_TOLERANCE_PX)

            empty_list = [d['hole_num'] for d in ordered_detections if d.get('smoothed_cls', d['cls']) == 0]

            # Periodic console reporting (for SSH / Headless)
            if curr_time - last_print_time >= 1.0:
                status_summary = ", ".join([
                    f"Hole {d['hole_num']}: {config.CLASS_NAMES.get(d.get('smoothed_cls', d['cls']))}"
                    for d in ordered_detections
                ])
                print(f"[FPS: {fps:4.1f}] Holes: {len(ordered_detections)} | Empty holes: {sorted(empty_list)} | {status_summary}")
                last_print_time = curr_time

            if not headless:
                overlay, _ = draw_visual_overlay(frame, ordered_detections, fps)
                cv2.imshow("DOFBOT Planting Hole Live Detection", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Detection stopped by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("[INFO] Cleaned up video stream and exited.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Planting Hole Detection on DOFBOT.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (console output only)")
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMGSZ,
                        help="Input image size (default: 640; use 320 for Jetson/Pi speedup)")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Confidence threshold (default: 0.50; try 0.25 for angled views)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model if available")
    args = parser.parse_args()

    run_detection(headless=args.headless, imgsz=args.imgsz, use_onnx=args.onnx, conf_thresh=args.conf)
