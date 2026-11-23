# DOFBOT Planting Hole Detection System

An end-to-end computer vision and robotics project for Yahboom DOFBOT 6-DOF robotic arm to detect, spatially number, and classify planting holes in hydroponic trays into **EMPTY** (Class 0) and **FILLED** (Class 1).

---

## 🎯 System Goal & Features

- **Live Classification**:
  - **EMPTY (Class 0, Red)**: ONLY when there is **NOTHING** in the hole (bare empty hole in the channel).
  - **FILLED (Class 1, Green)**: When a bucket / net cup is present in the hole (whether an empty bucket OR containing a peat plug).
- **Physical Rationale**: A hole with an empty bucket already seated cannot accept a new cup; only completely bare holes are vacant planting targets.
- **Spatial Hole Numbering**:
  - Automatically groups hole detections into horizontal rows using a configurable vertical row tolerance (`ROW_TOLERANCE_PX`).
  - Sorts rows top-to-bottom and holes left-to-right (reading order: Hole 1, Hole 2, Hole 3, Hole 4, ...).
- **Flicker-Free Temporal Smoothing**:
  - Tracks hole centroids across frames.
  - Applies rolling majority-vote over past frames to eliminate class flickering.
- **Edge Optimized (Jetson Nano / Raspberry Pi)**:
  - Supports reduced inference image sizes (`imgsz=320`).
  - Optional ONNX runtime export.
  - Real-time FPS counter.
  - `--headless` flag for remote SSH operation with zero GUI overhead.
- **DOFBOT 6-DOF Action Dispatcher**:
  - Uses `Arm_Lib.Arm_Device` and `Arm_serial_servo_write6_array(angles, time_ms)`.
  - Overhead scanning pose + pre-calibrated hole targeting with automatic servo calibration tool.
- **Rule-Based Fallback Mode**:
  - `6_fallback_roi.py`: Non-AI interactive ROI selection with HSV brown-plug color ratio detection.

---

## 📁 Project Structure

```
hole_detector/
├── dataset/
│   ├── images/ {train, val}
│   └── labels/ {train, val}
├── raw_dataset/
│   ├── images/
│   └── labels/
├── models/                     # Saved best.pt and best.onnx weights
├── data.yaml                   # YOLOv8 dataset configuration
├── config.py                   # Central configuration (camera, thresholds, poses)
├── 1_capture.py                # Dataset image collection tool
├── 2_split_dataset.py          # 80/20 train/val splitter
├── 3_train.py                  # YOLOv8n training pipeline (80 epochs)
├── 4_live_detect.py            # Live detection, numbering & overlay
├── 5_arm_action.py             # DOFBOT robotic arm action dispatcher
├── 6_fallback_roi.py           # Non-AI HSV fallback detector
├── read_servos.py              # Interactive servo calibration utility
├── generate_synthetic_data.py  # Sample dataset generator for immediate testing
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Step-by-Step Quickstart Guide

### Step 1: Set Up Python Virtual Environment

```bash
# Navigate to project directory
cd "d:/Juber B/Planting Hole Detection/hole_detector"

# Create a clean virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux / Jetson / Raspberry Pi:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note for Jetson Nano: PyTorch and torchvision are typically provided via official NVIDIA JetPack wheels).*

---

### Step 2: Calibrate DOFBOT Poses (Optional / Recommended)

Before running robotic arm movements, calibrate your arm angles:

```bash
python read_servos.py
```

1. Move the arm to your overhead view position -> Copy the printed array into `SCAN_POSE` in `config.py`.
2. Move the arm above Hole 1, Hole 2, Hole 3, Hole 4 -> Copy each array into `HOLE_POSES` in `config.py`.

---

### Step 3: Dataset Collection & Labeling

#### How Many Images to Capture?
- **Aim for 100 to 200 images**.
- Keep camera fixed in the overhead `SCAN_POSE`.
- Vary the arrangement across shots:
  - All 4 holes filled
  - 1 empty, 3 filled (test each hole being the empty one)
  - 2 empty, 2 filled
  - All empty
  - Mix of bare empty holes and empty white cups
- Vary ambient lighting (natural light, lamp, slight shadows).

#### Capture Images:
```bash
python 1_capture.py
```
- Press **[SPACE]** to capture an image.
- Press **[Q]** to finish. Images are saved to `raw_dataset/images/`.

#### Labeling Images:
1. Upload `raw_dataset/images/` to **Roboflow** (or use **LabelImg** locally).
2. Create 2 classes:
   - `0`: `empty` (draw bounding box around the net cup or bare hole)
   - `1`: `filled` (draw bounding box around the plug / cup with plant)
3. Export dataset in **YOLOv8 format**.
4. Place exported `.txt` label files into `raw_dataset/labels/`.

---

### Step 4: Split Dataset (80% Train, 20% Val)

```bash
python 2_split_dataset.py
```
*(Tip: To test the pipeline immediately before capturing real photos, run `python generate_synthetic_data.py`, then run `python 2_split_dataset.py`)*.

---

### Step 5: Train YOLOv8n Model

You can train on your laptop or Google Colab:

```bash
python 3_train.py --epochs 80 --imgsz 640
```
- Uses pretrained `yolov8n.pt`.
- Automatically copies the top weights to `models/best.pt`.
- Automatically exports an edge-optimized ONNX model to `models/best.onnx`.

---

### Step 6: Live Detection & Visual Numbering

Run live detection using your camera:

```bash
# Standard desktop / laptop view with visual GUI
python 4_live_detect.py

# High-performance mode on Jetson Nano / Raspberry Pi
python 4_live_detect.py --imgsz 320

# Headless mode for remote SSH execution
python 4_live_detect.py --headless
```

On screen, you will see:
- `Hole 1: FILLED` (Green)
- `Hole 2: FILLED` (Green)
- `Hole 3: FILLED` (Green)
- `Hole 4: EMPTY` (Red)
- Banner: `Empty holes: [4] | Total: 4 | FPS: 29.4`

---

### Step 7: Robotic Arm Automated Action

To have the DOFBOT automatically inspect and move towards the first empty hole:

```bash
# Run arm action dispatcher (with safe interactive confirmation)
python 5_arm_action.py --interactive

# Test movement to a specific hole directly (e.g. Hole 4)
python 5_arm_action.py --hole 4
```

---

### Step 8: Fallback Mode (No-AI HSV Color Detection)

If you haven't trained a YOLO model yet, use the rule-based OpenCV fallback:

1. **One-Time Calibration**:
   ```bash
   python 6_fallback_roi.py --calibrate
   ```
   Drag a bounding box over Hole 1, Hole 2, Hole 3, Hole 4. Press **[SPACE]** after each. Press **[ESC]** when done. Coordinates are saved to `roi_config.json`.

2. **Run Fallback Live Detection**:
   ```bash
   python 6_fallback_roi.py
   ```
