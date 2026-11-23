"""
read_servos.py - Yahboom DOFBOT Servo Angle Calibration Helper

Use this script to calibrate the exact servo angles for:
  1. SCAN_POSE (overhead inspection)
  2. HOLE_POSES (Hole 1, Hole 2, Hole 3, Hole 4, etc.)

Instructions:
  1. Gently move or control your DOFBOT arm to the desired position.
  2. Run this script:
       python read_servos.py
  3. The script will read all 6 servos and output a ready-to-copy line
     that you can paste directly into config.py!
"""

import sys
import time

try:
    from Arm_Lib import Arm_Device
    ARM_LIB_AVAILABLE = True
except ImportError:
    ARM_LIB_AVAILABLE = False


def read_all_servos(arm):
    """Reads angles for servos 1 through 6 from the DOFBOT controller board."""
    angles = []
    for servo_id in range(1, 7):
        try:
            # Arm_Lib API: Arm_serial_servo_read(servo_id) returns integer angle [0-180]
            val = arm.Arm_serial_servo_read(servo_id)
            if val is None or val < 0 or val > 180:
                val = 90
            angles.append(int(val))
        except Exception:
            angles.append(90)
        time.sleep(0.02)
    return angles


def main():
    print("=" * 65)
    print("YAHOOM DOFBOT SERVO ANGLE CALIBRATOR")
    print("=" * 65)

    if not ARM_LIB_AVAILABLE:
        print("[WARNING] Arm_Lib is not installed in this environment.")
        print("          Running in DEMO / SIMULATION mode.")
        print("          On your DOFBOT Jetson / Pi, this will read live hardware angles.")
        print("=" * 65)
        print("\nDemo angles format:")
        print("SCAN_POSE = [90, 90, 30, 30, 90, 90]")
        print("HOLE_POSES = {")
        print("    1: [65, 60, 45, 35, 90, 30],")
        print("    2: [80, 60, 45, 35, 90, 30],")
        print("    3: [95, 60, 45, 35, 90, 30],")
        print("    4: [110, 60, 45, 35, 90, 30]")
        print("}")
        return

    try:
        arm = Arm_Device()
        time.sleep(0.5)
        print("[INFO] Successfully connected to DOFBOT.")
    except Exception as e:
        print(f"[ERROR] Could not connect to DOFBOT serial device: {e}")
        sys.exit(1)

    print("\n[INSTRUCTIONS]")
    print("  - Move DOFBOT to the target position.")
    print("  - Press [ENTER] to sample current angles.")
    print("  - Type 'q' and press [ENTER] to exit.\n")

    pose_names = ["SCAN_POSE", "Hole 1", "Hole 2", "Hole 3", "Hole 4"]
    pose_idx = 0

    while True:
        prompt_name = pose_names[pose_idx] if pose_idx < len(pose_names) else f"Target Pose {pose_idx + 1}"
        user_input = input(f"Position arm for [{prompt_name}] and press [ENTER] (or 'q' to quit): ")
        if user_input.strip().lower() == 'q':
            break

        angles = read_all_servos(arm)
        print(f"\n--> Captured Angles for {prompt_name}:")
        print(f"    {angles}")
        if pose_idx == 0:
            print(f"    Paste into config.py: SCAN_POSE = {angles}\n")
        else:
            print(f"    Paste into config.py: {pose_idx}: {angles},\n")

        pose_idx += 1

    del arm
    print("[INFO] Calibration session ended.")


if __name__ == "__main__":
    main()
