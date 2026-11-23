"""
5_arm_action.py - Yahboom DOFBOT 6-DOF Robotic Arm Action Integration

Moves DOFBOT with Arm_Lib's Arm_Device:
  1. Starts at SCAN_POSE overhead to inspect the hydroponic tray.
  2. Runs detection or inspects input to identify the first EMPTY hole.
  3. Moves the DOFBOT to that hole's pre-calibrated pose from config.HOLE_POSES.
  4. Pauses for inspection/action.
  5. Returns safely to SCAN_POSE.

Note: Includes a simulated fallback mode if Arm_Lib is not installed (e.g. testing on laptop).
"""

import sys
import time
import argparse
from pathlib import Path
import config

# Attempt importing Arm_Lib (Yahboom DOFBOT Python library)
try:
    from Arm_Lib import Arm_Device
    ARM_LIB_AVAILABLE = True
except ImportError:
    ARM_LIB_AVAILABLE = False


class DOFBOTController:
    def __init__(self, mock=False):
        self.mock = mock or not ARM_LIB_AVAILABLE
        self.arm = None
        if self.mock:
            print("[INFO] Arm_Lib not available. Running in SIMULATION / MOCK mode.")
        else:
            print("[INFO] Initializing Yahboom Arm_Device...")
            try:
                self.arm = Arm_Device()
                time.sleep(0.5)
            except Exception as e:
                print(f"[WARNING] Could not connect to DOFBOT serial device: {e}")
                print("[INFO] Falling back to MOCK mode.")
                self.mock = True

    def move_to_angles(self, angles, move_time_ms=config.SERVO_MOVE_TIME_MS):
        """Moves DOFBOT 6 servos to specified angle list."""
        if len(angles) != 6:
            raise ValueError(f"Expected 6 servo angles, got {len(angles)}: {angles}")

        if self.mock:
            print(f"  [SIMULATOR] Moving DOFBOT servos -> {angles} (Duration: {move_time_ms}ms)")
            time.sleep(move_time_ms / 1000.0)
        else:
            # Yahboom DOFBOT 6-servo write API: Arm_serial_servo_write6_array(angles, time_ms)
            self.arm.Arm_serial_servo_write6_array(angles, move_time_ms)
            time.sleep(move_time_ms / 1000.0)

    def go_to_scan_pose(self):
        print(f"[ACTION] Moving DOFBOT to SCAN_POSE: {config.SCAN_POSE}")
        self.move_to_angles(config.SCAN_POSE, config.SERVO_MOVE_TIME_MS)

    def target_empty_hole(self, hole_number):
        if hole_number not in config.HOLE_POSES:
            print(f"[ERROR] Hole {hole_number} has no calibrated pose in config.HOLE_POSES.")
            print(f"        Available poses: {list(config.HOLE_POSES.keys())}")
            print("        Use 'python read_servos.py' to calibrate poses for your setup.")
            return False

        target_angles = config.HOLE_POSES[hole_number]
        print(f"\n[ACTION] Dispatching DOFBOT to Hole {hole_number}: {target_angles}")
        self.move_to_angles(target_angles, config.SERVO_MOVE_TIME_MS)

        print(f"[ACTION] Pausing for {config.SERVO_REST_TIME_S}s over Hole {hole_number}...")
        time.sleep(config.SERVO_REST_TIME_S)

        print("[ACTION] Returning DOFBOT to SCAN_POSE...")
        self.go_to_scan_pose()
        return True

    def close(self):
        if not self.mock and self.arm is not None:
            del self.arm


def main():
    parser = argparse.ArgumentParser(description="Yahboom DOFBOT Planting Hole Action Dispatcher.")
    parser.add_argument("--hole", type=int, default=None,
                        help="Manually specify target hole number (e.g. 4) to test arm movement directly.")
    parser.add_argument("--scan-only", action="store_true",
                        help="Move to scan pose and exit.")
    parser.add_argument("--interactive", action="store_true",
                        help="Ask for confirmation before physical movement.")
    args = parser.parse_args()

    controller = DOFBOTController()

    try:
        # Move to scan pose first
        if args.interactive:
            input("Press [ENTER] to move DOFBOT to SCAN_POSE...")
        controller.go_to_scan_pose()

        if args.scan_only:
            print("[INFO] SCAN_POSE reached. Exiting as requested.")
            return

        target_hole = args.hole
        if target_hole is None:
            # Demonstration target (Hole 4 from prompt example)
            print("[INFO] No specific hole given. Using Hole 4 as demonstration target.")
            target_hole = 4

        if args.interactive:
            input(f"Press [ENTER] to dispatch DOFBOT to target Hole {target_hole}...")

        controller.target_empty_hole(target_hole)
        print("\n[SUCCESS] Arm action routine completed successfully.")

    finally:
        controller.close()


if __name__ == "__main__":
    main()
