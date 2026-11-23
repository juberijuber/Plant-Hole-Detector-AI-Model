# ==============================================================================
# DOFBOT Hydroponic Planting Hole Detector - Configuration
# All tunable parameters are defined here. No magic numbers in code.
# ==============================================================================

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATASET_DIR = BASE_DIR / "raw_dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Camera & Video Stream Settings
# ------------------------------------------------------------------------------
# Camera index (0 is typically default USB webcam or Raspberry Pi camera).
# On Jetson Nano with CSI camera, this can be a GStreamer pipeline string.
CAMERA_INDEX = 0

# Video capture resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Optional GStreamer pipeline for Jetson Nano CSI camera (e.g. IMX219)
# Set USE_CSI_CAMERA = True if using Jetson CSI ribbon camera
USE_CSI_CAMERA = False
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

# ------------------------------------------------------------------------------
# 2. YOLOv8 Detection & Inference Settings
# ------------------------------------------------------------------------------
# Path to trained model weights
# Defaults to models/best.pt; falls back to yolov5nu.pt (YOLOv5n)
MODEL_PATH = MODELS_DIR / "best.pt"
FALLBACK_BASE_MODEL = "yolov5nu.pt"
ONNX_MODEL_PATH = MODELS_DIR / "best.onnx"

# Inference parameters
CONFIDENCE_THRESHOLD = 0.50   # Minimum detection confidence
IOU_THRESHOLD = 0.45          # NMS IoU threshold
INFERENCE_IMGSZ = 640         # 640 for standard, 320 or 416 for Jetson Nano / Pi

# Class definitions (YOLO classes)
# 0: empty  (BARE hole with NOTHING inside - vacant slot)
# 1: filled (bucket/cup present in hole, whether empty or with peat plug)
CLASS_NAMES = {
    0: "EMPTY",
    1: "FILLED"
}

# Visualization colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 255),    # Red for EMPTY
    1: (0, 220, 0)     # Green for FILLED
}

# ------------------------------------------------------------------------------
# 3. Spatial Sorting & Hole Numbering
# ------------------------------------------------------------------------------
# Vertical pixel tolerance to group hole detections into the same horizontal row.
# If abs(y1 - y2) <= ROW_TOLERANCE_PX, they belong to the same row.
ROW_TOLERANCE_PX = 45

# Number of past frames to smooth classification (Majority Vote / Stability)
# Helps eliminate frame-to-frame flicker on Jetson Nano / Raspberry Pi feeds.
SMOOTHING_WINDOW_FRAMES = 5

# Centroid distance threshold (pixels) to match a hole across consecutive frames
TRACKING_MAX_DISTANCE_PX = 50

# ------------------------------------------------------------------------------
# 4. Yahboom DOFBOT 6-DOF Robotic Arm Settings
# ------------------------------------------------------------------------------
# Servo angles in degrees: [Servo 1 (Base), Servo 2, Servo 3, Servo 4, Servo 5, Servo 6 (Gripper)]
# Calibrate these exact angles using `python read_servos.py`

# Overhead scanning pose where the camera has an unobstructed view of the tray
SCAN_POSE = [90, 90, 30, 30, 90, 90]

# Pre-calibrated poses to position the arm directly over Hole 1..4
HOLE_POSES = {
    1: [65, 60, 45, 35, 90, 30],
    2: [80, 60, 45, 35, 90, 30],
    3: [95, 60, 45, 35, 90, 30],
    4: [110, 60, 45, 35, 90, 30]
}

# Movement execution parameters
SERVO_MOVE_TIME_MS = 1500     # Time allocated for smooth servo transition (ms)
SERVO_REST_TIME_S = 1.0       # Time to hover/pause over hole before returning (s)

# ------------------------------------------------------------------------------
# 5. Non-AI Fallback Mode (HSV Color Segmentation on Fixed ROIs)
# ------------------------------------------------------------------------------
ROI_CONFIG_FILE = BASE_DIR / "roi_config.json"

# HSV bounds for brown peat/coco soil plug
# In HSV: Hue (0-180), Saturation (0-255), Value (0-255)
HSV_BROWN_LOWER = [8, 40, 30]
HSV_BROWN_UPPER = [28, 255, 180]

# Fraction of pixels in the hole ROI that must match brown to be classified FILLED
FILLED_RATIO_THRESHOLD = 0.12
