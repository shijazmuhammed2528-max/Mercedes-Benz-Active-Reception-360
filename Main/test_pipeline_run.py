"""
Mercedes-Benz Active Reception — End-to-End Real Video Pipeline Test
Runs the CaptureController on the actual 4 video streams:
- Loads YOLO on GPU (or CPU)
- Plays through frames of front, back, left, right
- Verifies detections in all 4 views
- Verifies stationary state
- Triggers automatic capture
- Confirms images and metadata saved in D:\Project Costal star\Data_base
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config, DATABASE_DIR, DATABASE_FILE
from backend.database.db_manager import DatabaseManager
from backend.capture.capture_controller import CaptureController

def run_test():
    print("=" * 70)
    print("LIVE PIPELINE TEST: 4-CAMERA SYNC & AUTOMATIC CAPTURE")
    print("=" * 70)

    cfg = load_config()
    # Speed up processing FPS slightly for test execution
    cfg["yolo_process_fps"] = 12
    cfg["stationary_duration"] = 1.5

    db = DatabaseManager(DATABASE_FILE)
    controller = CaptureController(cfg, db)

    print("\n[1] Starting Capture Controller...")
    start_res = controller.start()
    print("Start result:", start_res)
    assert start_res["success"], f"Start failed: {start_res}"

    print("\n[2] Monitoring Pipeline Execution...")
    captured = False
    start_time = time.time()
    max_wait = 25 # seconds

    while time.time() - start_time < max_wait:
        status = controller.get_status()
        state = status["capture_state"]["state"]
        msg = status["capture_state"]["message"]
        conds = status["conditions"]

        # Format condition summary
        summary = []
        for cam in ["front", "back", "left", "right"]:
            c = conds.get(cam, {})
            det = "DET" if c.get("detected") else "---"
            vq = "VQ_OK" if c.get("view_ok") else "VQ_WAIT"
            stat = "STILL" if c.get("stationary") else c.get("motion_state", "WAIT")
            conf = f"{c.get('confidence', 0)*100:.0f}%"
            summary.append(f"{cam[:1].upper()}:[{det}|{vq}|{stat}|{conf}]")

        print(f"[{time.time()-start_time:4.1f}s] State: {state:16} | {' '.join(summary)} | Msg: {msg}")

        if status["captured"]:
            captured = True
            print("\n*** CAPTURE TRIGGERED SUCCESSFULLY! ***")
            print("Capture Result:", status["capture_result"])
            break

        time.sleep(0.5)

    print("\n[3] Stopping Controller...")
    controller.stop()

    print("\n[4] Verifying Saved Files in Database Directory...")
    assert captured, "Capture was not triggered within time limit!"
    res = status["capture_result"]
    folder = res["folder_path"]
    print(f"Session folder: {folder}")
    assert os.path.isdir(folder), f"Folder does not exist: {folder}"

    for img_name in ["front.jpg", "back.jpg", "left.jpg", "right.jpg", "metadata.json"]:
        p = os.path.join(folder, img_name)
        assert os.path.isfile(p), f"Missing file: {p}"
        size = os.path.getsize(p)
        print(f"  - {img_name}: {size:,} bytes (OK)")

    print("\n[5] Verifying SQLite Record...")
    session_row = db.get_session(res["session_id"])
    assert session_row is not None, "Session not found in SQLite DB!"
    print(f"  - DB Record found: ID={session_row['session_id']}, Vehicle={session_row['vehicle_number']}, Date={session_row['capture_date']}")

    print("\n" + "=" * 70)
    print("SUCCESS: 4-CAMERA LIVE PIPELINE AND CAPTURE FULLY VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
