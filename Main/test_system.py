"""
Mercedes-Benz Active Reception — Automated End-to-End Verification Test
Tests all components:
1. Video readability (all 4 cameras)
2. Database CRUD & search operations
3. View quality validation logic
4. Motion detection logic
5. Storage service & ZIP creation
6. YOLO inference & CUDA/CPU auto-detection
"""
import os
import sys
import json
import time
import shutil
import sqlite3
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import VIDEO_SOURCES, CAMERA_LABELS, DATABASE_DIR, DATABASE_FILE, load_config
from backend.database.db_manager import DatabaseManager
from backend.services.storage_service import check_video_files, get_storage_info
from backend.services.zip_service import create_session_zip
from backend.vision.view_quality import ViewQualityValidator
from backend.vision.motion_detector import MotionDetector
from backend.vision.frame_selector import compute_sharpness, select_best_frame

def test_all():
    print("=" * 70)
    print("MERCEDES-BENZ ACTIVE RECEPTION — SYSTEM TEST SUITE")
    print("=" * 70)
    passed = 0
    total = 0

    # -------------------------------------------------------------
    # TEST 1: Video File Verification
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 1] Checking Video Files...")
    video_status = check_video_files(VIDEO_SOURCES)
    all_videos_exist = all(v["exists"] for v in video_status.values())
    for cam, info in video_status.items():
        print(f"  - {cam.upper()}: {'EXISTS' if info['exists'] else 'MISSING'} ({info['size_mb']} MB) -> {info['path']}")
    if all_videos_exist:
        print("  -> PASS: All four video files exist and are accessible.")
        passed += 1
    else:
        print("  -> FAIL: One or more video files are missing.")

    # -------------------------------------------------------------
    # TEST 2: OpenCV Video Stream Readability
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 2] Testing OpenCV Video Decoding...")
    import cv2
    decoded_all = True
    for cam, path in VIDEO_SOURCES.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  - FAIL: Could not open {cam} ({path})")
            decoded_all = False
            continue
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if ret and frame is not None:
            print(f"  - {cam.upper()}: OK ({w}x{h} @ {fps:.1f} FPS, {frames} frames)")
        else:
            print(f"  - FAIL: Could not read frame from {cam}")
            decoded_all = False
    if decoded_all:
        print("  -> PASS: All 4 video streams decoded successfully.")
        passed += 1
    else:
        print("  -> FAIL: Some video streams failed decoding.")

    # -------------------------------------------------------------
    # TEST 3: Database Initialization, Insertion, & Search
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 3] Testing SQLite Database CRUD & Search...")
    test_db_path = os.path.join(DATABASE_DIR, "test_verification.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = DatabaseManager(test_db_path)
    test_session = {
        "session_id": "test_session_001",
        "vehicle_number": "KL 01 AL 0009",
        "vehicle_number_normalized": "KL01AL0009",
        "capture_date": "2026-11-21",
        "capture_timestamp": "2026-11-21T12:15:30",
        "capture_time_12h": "12:15:30 PM",
        "folder_path": os.path.join(DATABASE_DIR, "KL_01_AL_0009_2026-11-21_12-15-30_PM"),
        "front_image_path": "front.jpg",
        "back_image_path": "back.jpg",
        "left_image_path": "left.jpg",
        "right_image_path": "right.jpg",
        "front_confidence": 0.94,
        "back_confidence": 0.92,
        "left_confidence": 0.91,
        "right_confidence": 0.95,
        "capture_status": "success",
    }
    inserted = db.insert_session(test_session)
    retrieved = db.get_session("test_session_001")
    search_norm = db.search_sessions(vehicle_number="kl01al0009")
    search_space = db.search_sessions(vehicle_number="KL 01 AL 0009")
    search_date = db.search_sessions(date="2026-11-21")
    search_time = db.search_sessions(time_from="12:00", time_to="13:00")
    stats = db.get_stats()

    db_ok = (
        inserted and
        retrieved is not None and
        retrieved["vehicle_number"] == "KL 01 AL 0009" and
        len(search_norm) == 1 and
        len(search_space) == 1 and
        len(search_date) == 1 and
        len(search_time) == 1 and
        stats["total_sessions"] >= 1
    )
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    if db_ok:
        print("  - Insert session: OK")
        print("  - Retrieve session by ID: OK")
        print("  - Search normalized 'kl01al0009': OK (matched 1)")
        print("  - Search formatted 'KL 01 AL 0009': OK (matched 1)")
        print("  - Search by Date '2026-11-21': OK (matched 1)")
        print("  - Search by Time range '12:00' to '13:00': OK (matched 1)")
        print("  - Statistics calculation: OK")
        print("  -> PASS: Database and search verification passed.")
        passed += 1
    else:
        print("  -> FAIL: Database operations failed verification.")

    # -------------------------------------------------------------
    # TEST 4: View Quality Validator
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 4] Testing View Quality Validator...")
    vqv = ViewQualityValidator(margin_ratio=0.03, min_area_ratio=0.05)
    # Valid central detection
    res_valid = vqv.validate([200, 150, 1600, 950], 1920, 1080)
    # Clipped edge detection (touching left boundary)
    res_clipped = vqv.validate([10, 150, 1600, 950], 1920, 1080)
    # Tiny detection (below min area)
    res_tiny = vqv.validate([500, 500, 520, 520], 1920, 1080)

    vq_ok = res_valid["passed"] and (not res_clipped["passed"]) and (not res_tiny["passed"])
    if vq_ok:
        print(f"  - Valid detection check: PASSED (passed={res_valid['passed']})")
        print(f"  - Clipped border check: REJECTED as expected (passed={res_clipped['passed']})")
        print(f"  - Undersized detection: REJECTED as expected (passed={res_tiny['passed']})")
        print("  -> PASS: View quality rules operate correctly.")
        passed += 1
    else:
        print("  -> FAIL: View quality validation gave incorrect results.")

    # -------------------------------------------------------------
    # TEST 5: Motion Detector
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 5] Testing Vehicle Motion Detector...")
    md = MotionDetector(center_threshold=15.0, size_threshold=0.08, stationary_duration=1.0)
    # Simulate moving vehicle
    s1 = md.update([100, 100, 400, 300])
    s2 = md.update([150, 100, 450, 300]) # 50px shift
    # Simulate stationary vehicle (first still frame sets baseline)
    s3 = md.update([200, 100, 500, 300]) # Still frame 1 (displacement from 150 is 50px, moving)
    s3_still = md.update([200, 100, 500, 300]) # Still frame 2 (displacement 0px, starts STOPPING timer)
    time.sleep(1.1)
    s4 = md.update([201, 100, 501, 300]) # ~1px drift after 1.1s (elapsed > 1.0s -> STATIONARY)

    motion_ok = (s2 == "MOVING") and (s4 == "STATIONARY") and md.is_stationary
    if motion_ok:
        print(f"  - Movement displacement detected: OK (state={s2})")
        print(f"  - Stationary duration confirmed: OK (state={s4}, is_stationary={md.is_stationary})")
        print("  -> PASS: Motion detector logic operates correctly.")
        passed += 1
    else:
        print(f"  -> FAIL: Motion detector state mismatch (s2={s2}, s4={s4})")

    # -------------------------------------------------------------
    # TEST 6: Frame Sharpness Selector
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 6] Testing Frame Sharpness Selector...")
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add high-frequency checkerboard pattern
    sharp_img[::20, ::20] = 255
    blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)

    score_sharp = compute_sharpness(sharp_img)
    score_blurry = compute_sharpness(blurry_img)
    best_frame, best_score, idx = select_best_frame([blurry_img, sharp_img])

    sharp_ok = (score_sharp > score_blurry) and (idx == 1)
    if sharp_ok:
        print(f"  - Sharp score: {score_sharp:.1f} vs Blurry score: {score_blurry:.1f}")
        print(f"  - Best frame index selected: {idx}")
        print("  -> PASS: Frame sharpness selection operates correctly.")
        passed += 1
    else:
        print("  -> FAIL: Sharpness calculation failed.")

    # -------------------------------------------------------------
    # TEST 7: ZIP Packaging
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 7] Testing In-Memory ZIP Packaging...")
    test_folder = os.path.join(DATABASE_DIR, "test_zip_folder")
    os.makedirs(test_folder, exist_ok=True)
    with open(os.path.join(test_folder, "test.txt"), "w") as f:
        f.write("Mercedes-Benz Active Reception Test")
    zip_buf = create_session_zip(test_folder, "KL 01 AL 0009")
    shutil.rmtree(test_folder)

    zip_ok = zip_buf is not None and zip_buf.getbuffer().nbytes > 0
    if zip_ok:
        print(f"  - ZIP buffer generated: {zip_buf.getbuffer().nbytes} bytes")
        print("  -> PASS: ZIP service operates correctly.")
        passed += 1
    else:
        print("  -> FAIL: ZIP packaging failed.")

    # -------------------------------------------------------------
    # TEST 8: YOLO Inference & CUDA/CPU Test
    # -------------------------------------------------------------
    total += 1
    print("\n[TEST 8] Checking YOLO Inference Engine...")
    try:
        from ultralytics import YOLO
        import torch
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")

        model_path = os.path.join(PROJECT_ROOT, "yolov8n.pt")
        model = YOLO(model_path)
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_frame, device=device_str, verbose=False)
        print(f"  - Inference test executed on {device_str}: SUCCESS")
        print("  -> PASS: YOLO pipeline verified.")
        passed += 1
    except Exception as e:
        print(f"  - Notice: YOLO/PyTorch still loading or initializing: {e}")
        print("  -> Pending final package installation check.")

    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} / {total} PASSED")
    print("=" * 70)
    return passed == total

if __name__ == "__main__":
    test_all()
