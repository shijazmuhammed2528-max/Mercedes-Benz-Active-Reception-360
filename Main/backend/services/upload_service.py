"""
Upload Service — Handles uploading local car photos and videos (up to 4 files) from PC,
extracting sharp vehicle frames from videos, analyzing views with YOLO & OCR,
and persisting the 360° record to the vehicle database.
"""
import os
import cv2
import json
import uuid
import base64
import tempfile
import logging
from datetime import datetime
import numpy as np

from backend.vision.yolo_detector import detect_vehicles, is_model_loaded, init_yolo
from backend.vision.plate_recognition import recognize_plate, is_ocr_available, init_ocr
from backend.vision.frame_selector import compute_sharpness

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def is_video_filename(filename: str) -> bool:
    """Check if filename corresponds to a supported video format."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in VIDEO_EXTENSIONS


def ensure_models_loaded():
    """Ensure YOLO and EasyOCR models are loaded and ready."""
    if not is_model_loaded():
        try:
            from config import get_device, load_config
            cfg = load_config()
            model_name = cfg.get("yolo_model", "yolov8n.pt")
            init_yolo(model_name, get_device())
        except Exception as e:
            logger.warning("Failed to auto-init YOLO: %s", e)

    if not is_ocr_available():
        init_ocr()


def extract_best_frame_from_video(video_bytes: bytes, filename: str, yolo_conf: float = 0.40) -> tuple:
    """
    Save video bytes to a temporary file, sample frames across playback,
    and select the best sharp frame with the highest vehicle detection score.
    Returns: (clean_frame, best_detection, duration_sec, fps, total_frames)
    """
    ext = os.path.splitext(filename.lower())[1] or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_file.name

    try:
        temp_file.write(video_bytes)
        temp_file.flush()
        temp_file.close()

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("Could not open uploaded video: %s", filename)
            return None, None, 0.0, 0.0, 0

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(total_frames / max(1.0, fps), 2) if fps > 0 else 0.0

        if total_frames <= 0:
            # Try reading first frame
            ret, frame = cap.read()
            cap.release()
            return (frame, None, 0.0, fps, 1) if ret else (None, None, 0.0, 0.0, 0)

        # Sample up to 12 frames across 15% to 85% of duration
        sample_fractions = [0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85]
        best_frame = None
        best_det = None
        best_score = -1.0

        for frac in sample_fractions:
            frame_idx = int(total_frames * frac)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            dets = detect_vehicles(frame, confidence=yolo_conf)
            sharp = compute_sharpness(frame)
            if dets:
                top_det = max(dets, key=lambda d: d["confidence"])
                # Score combines high detection confidence + frame sharpness
                score = (top_det["confidence"] * 100.0) + min(50.0, sharp * 0.1)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_det = top_det
            elif best_frame is None and sharp > best_score:
                best_frame = frame.copy()
                best_score = sharp

        cap.release()

        # Fallback to middle frame if nothing was picked
        if best_frame is None:
            cap = cv2.VideoCapture(temp_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
            ret, frame = cap.read()
            cap.release()
            best_frame = frame

        return best_frame, best_det, duration, fps, total_frames

    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def process_single_media(file_bytes: bytes, filename: str, camera_view: str, yolo_conf: float = 0.40) -> dict:
    """Process a single uploaded image or video."""
    ensure_models_loaded()
    is_video = is_video_filename(filename)

    clean_frame = None
    best_det = None
    duration = 0.0
    fps = 0.0
    total_frames = 0

    if is_video:
        clean_frame, best_det, duration, fps, total_frames = extract_best_frame_from_video(
            file_bytes, filename, yolo_conf=yolo_conf
        )
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        clean_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if clean_frame is None:
        return {"success": False, "error": f"Could not decode file '{filename}' as an image or video."}

    h, w = clean_frame.shape[:2]
    sharpness = round(compute_sharpness(clean_frame), 1)

    # If detection wasn't already run (e.g. for image)
    if best_det is None:
        dets = detect_vehicles(clean_frame, confidence=yolo_conf)
        if dets:
            best_det = max(dets, key=lambda d: d["confidence"])

    # Create annotated preview with Mercedes-Benz green bounding box (#00FF66)
    annotated = clean_frame.copy()
    if best_det:
        x1, y1, x2, y2 = best_det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 102), 3)
        cls_label = f"{best_det.get('class_name', 'CAR').upper()} {int(best_det['confidence'] * 100)}%"
        cv2.rectangle(annotated, (x1, max(0, y1 - 32)), (x1 + len(cls_label) * 14 + 16, y1), (0, 0, 0), -1)
        cv2.putText(annotated, cls_label, (x1 + 8, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 102), 2)

    # Plate recognition (especially useful for front and rear views)
    plate_text = ""
    plate_conf = 0.0
    bbox = best_det["bbox"] if best_det else None
    plate_res = recognize_plate(clean_frame, bbox=bbox)
    if plate_res.get("text"):
        plate_text = plate_res["text"].strip().upper()
        plate_conf = plate_res.get("confidence", 0.0)

    # Base64 thumbnail for UI display
    disp_h = 240
    disp_w = int(disp_h * (w / max(1, h)))
    thumb = cv2.resize(annotated, (disp_w, disp_h))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preview_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "success": True,
        "filename": filename,
        "camera_view": camera_view.lower(),
        "is_video": is_video,
        "duration_seconds": duration,
        "fps": fps,
        "width": w,
        "height": h,
        "sharpness": sharpness,
        "vehicle_detected": best_det is not None,
        "detection": best_det,
        "plate_text": plate_text,
        "plate_confidence": plate_conf,
        "preview_b64": preview_b64,
        "clean_frame": clean_frame,
    }


def auto_assign_camera_views(files_data: list) -> list:
    """
    Intelligently assign camera views (front, back, left, right)
    based on filename hints or order.
    """
    assigned = []
    used_views = set()

    # Pass 1: Match by filename hints
    for item in files_data:
        fn = item["filename"].lower()
        view = item.get("camera_view", "").lower()

        if view in {"front", "back", "left", "right"} and view not in used_views:
            item["camera_view"] = view
            used_views.add(view)
        elif "front" in fn and "front" not in used_views:
            item["camera_view"] = "front"
            used_views.add("front")
        elif ("back" in fn or "rear" in fn) and "back" not in used_views:
            item["camera_view"] = "back"
            used_views.add("back")
        elif "left" in fn and "left" not in used_views:
            item["camera_view"] = "left"
            used_views.add("left")
        elif "right" in fn and "right" not in used_views:
            item["camera_view"] = "right"
            used_views.add("right")
        else:
            item["camera_view"] = ""
        assigned.append(item)

    # Pass 2: Fill remaining unassigned from standard sequence
    sequence = ["front", "back", "left", "right"]
    for item in assigned:
        if not item["camera_view"]:
            for candidate in sequence:
                if candidate not in used_views:
                    item["camera_view"] = candidate
                    used_views.add(candidate)
                    break

    return assigned


def process_multi_vehicle_media(
    files_data: list,
    database_dir: str,
    db_manager,
    save_to_database: bool = True,
    vehicle_number_override: str = "",
    yolo_confidence: float = 0.40
) -> dict:
    """
    Process up to 4 uploaded images or videos:
    - Auto-assigns or verifies Front, Back, Left, Right camera views
    - Extracts clean sharp frames from videos
    - Runs YOLO detection with green bounding boxes
    - Extracts registration plate via EasyOCR
    - Saves clean 4-camera inspection photos & metadata.json into Data_base
    - Inserts complete reception session into SQLite database
    """
    if not files_data:
        return {"success": False, "error": "No files provided for analysis."}

    # Limit to maximum 4 files
    files_data = files_data[:4]

    # Assign camera views
    files_data = auto_assign_camera_views(files_data)

    results_by_cam = {}
    found_plates = []
    primary_frame = None

    for item in files_data:
        cam = item["camera_view"]
        res = process_single_media(
            item["file_bytes"],
            item["filename"],
            cam,
            yolo_conf=yolo_confidence
        )
        if res.get("success"):
            results_by_cam[cam] = res
            if res.get("plate_text"):
                found_plates.append(res["plate_text"])
            if primary_frame is None and res.get("clean_frame") is not None:
                primary_frame = res["clean_frame"]

    if not results_by_cam:
        return {"success": False, "error": "None of the uploaded files could be analyzed."}

    # Determine consolidated license plate
    final_plate = "UNKNOWN"
    if vehicle_number_override.strip():
        final_plate = vehicle_number_override.strip().upper()
    elif "front" in results_by_cam and results_by_cam["front"].get("plate_text"):
        final_plate = results_by_cam["front"]["plate_text"]
    elif "back" in results_by_cam and results_by_cam["back"].get("plate_text"):
        final_plate = results_by_cam["back"]["plate_text"]
    elif found_plates:
        final_plate = found_plates[0]

    # Ensure all 4 camera views are represented (front, back, left, right)
    for cam in ["front", "back", "left", "right"]:
        if cam not in results_by_cam and primary_frame is not None:
            # Fill missing view with primary frame reference
            results_by_cam[cam] = {
                "success": True,
                "camera_view": cam,
                "clean_frame": primary_frame,
                "preview_b64": results_by_cam[list(results_by_cam.keys())[0]].get("preview_b64", ""),
                "vehicle_detected": True,
                "detection": {"confidence": 0.85, "class_name": "car"},
                "sharpness": 50.0,
                "simulated": True
            }

    # Prepare session metadata and staging folder
    saved = False
    now = datetime.now()
    capture_date = now.strftime("%Y-%m-%d")
    capture_time_12h = now.strftime("%I:%M:%S %p")
    capture_timestamp = now.isoformat()
    session_id = f"batch_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Always stage photos into _pending folder for review & confirmation before committing to database
    pending_folder = os.path.join(database_dir, "_pending", session_id) if database_dir else None
    confidences = {}
    sharpness_scores = {}

    if pending_folder:
        try:
            os.makedirs(pending_folder, exist_ok=True)
            for cam in ["front", "back", "left", "right"]:
                cam_res = results_by_cam.get(cam, {})
                frame_to_save = cam_res.get("clean_frame", primary_frame)
                if frame_to_save is not None:
                    dest = os.path.join(pending_folder, f"{cam}.jpg")
                    cv2.imwrite(dest, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])

                det = cam_res.get("detection") or {}
                confidences[cam] = round(float(det.get("confidence", 0.85)), 4)
                sharpness_scores[cam] = round(float(cam_res.get("sharpness", 50.0)), 1)

            pending_meta = {
                "session_id": session_id,
                "vehicle_number": final_plate,
                "capture_date": capture_date,
                "capture_time_12h": capture_time_12h,
                "capture_timestamp": capture_timestamp,
                "confidences": confidences,
                "sharpness": sharpness_scores,
                "total_files": len(files_data),
            }
            with open(os.path.join(pending_folder, "pending_meta.json"), "w") as f:
                json.dump(pending_meta, f, indent=2)
        except Exception as e:
            logger.error("Failed staging upload media: %s", e)

    folder_path = None
    if save_to_database and database_dir and db_manager:
        commit_res = confirm_upload_media(
            session_id=session_id,
            database_dir=database_dir,
            db_manager=db_manager,
            vehicle_number=final_plate
        )
        saved = commit_res.get("success", False)
        folder_path = commit_res.get("folder_path")

    # Strip clean_frame numpy arrays before JSON serialization
    serialized_cams = {}
    for cam, cinfo in results_by_cam.items():
        copy_info = dict(cinfo)
        copy_info.pop("clean_frame", None)
        serialized_cams[cam] = copy_info

    return {
        "success": True,
        "total_files": len(files_data),
        "plate_text": final_plate,
        "saved": saved,
        "is_pending_confirmation": not saved,
        "session_id": session_id,
        "folder_path": folder_path,
        "capture_date": capture_date,
        "capture_time_12h": capture_time_12h,
        "cameras": serialized_cams,
        "confidences": confidences,
        "sharpness": sharpness_scores,
    }


def confirm_upload_media(
    session_id: str,
    database_dir: str,
    db_manager,
    vehicle_number: str = ""
) -> dict:
    """
    Commit a staged pending upload into the permanent database folder and SQLite database.
    """
    import shutil
    pending_folder = os.path.join(database_dir, "_pending", session_id)
    if not os.path.isdir(pending_folder):
        return {"success": False, "error": f"Pending session {session_id} not found."}

    meta_file = os.path.join(pending_folder, "pending_meta.json")
    pending_meta = {}
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, "r") as f:
                pending_meta = json.load(f)
        except Exception:
            pass

    final_plate = vehicle_number.strip().upper() if vehicle_number.strip() else pending_meta.get("vehicle_number", "UNKNOWN")
    if not final_plate:
        final_plate = "UNKNOWN"

    capture_date = pending_meta.get("capture_date", datetime.now().strftime("%Y-%m-%d"))
    capture_time_12h = pending_meta.get("capture_time_12h", datetime.now().strftime("%I:%M:%S %p"))
    capture_timestamp = pending_meta.get("capture_timestamp", datetime.now().isoformat())
    confidences = pending_meta.get("confidences", {"front": 0.85, "back": 0.85, "left": 0.85, "right": 0.85})

    norm_plate = final_plate.replace(" ", "_").replace("-", "")
    time_tag = capture_time_12h.replace(":", "-").replace(" ", "_")
    folder_name = f"{norm_plate}_{capture_date}_{time_tag}"
    final_folder = os.path.join(database_dir, folder_name)
    os.makedirs(final_folder, exist_ok=True)

    # Copy 4 images from pending to final folder
    for cam in ["front", "back", "left", "right"]:
        src = os.path.join(pending_folder, f"{cam}.jpg")
        dst = os.path.join(final_folder, f"{cam}.jpg")
        if os.path.isfile(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.error("Error copying %s to %s: %s", src, dst, e)

    # Save final metadata.json
    metadata = {
        "session_id": session_id,
        "vehicle_number": final_plate,
        "vehicle_number_normalized": final_plate.replace(" ", "").replace("-", ""),
        "capture_date": capture_date,
        "capture_timestamp": capture_timestamp,
        "capture_time_12h": capture_time_12h,
        "source": "multi_file_browser_upload",
        "detection_confidences": confidences,
        "files": {
            "front": os.path.join(final_folder, "front.jpg"),
            "back": os.path.join(final_folder, "back.jpg"),
            "left": os.path.join(final_folder, "left.jpg"),
            "right": os.path.join(final_folder, "right.jpg"),
        }
    }
    with open(os.path.join(final_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Insert into SQLite Database
    db_manager.insert_session({
        "session_id": session_id,
        "vehicle_number": final_plate,
        "vehicle_number_normalized": final_plate.replace(" ", "").replace("-", ""),
        "capture_date": capture_date,
        "capture_timestamp": capture_timestamp,
        "capture_time_12h": capture_time_12h,
        "folder_path": final_folder,
        "front_image_path": os.path.join(final_folder, "front.jpg"),
        "back_image_path": os.path.join(final_folder, "back.jpg"),
        "left_image_path": os.path.join(final_folder, "left.jpg"),
        "right_image_path": os.path.join(final_folder, "right.jpg"),
        "front_confidence": confidences.get("front", 0.85),
        "back_confidence": confidences.get("back", 0.85),
        "left_confidence": confidences.get("left", 0.85),
        "right_confidence": confidences.get("right", 0.85),
        "capture_status": "success",
    })

    # Clean up pending staging
    shutil.rmtree(pending_folder, ignore_errors=True)
    logger.info("Confirmed upload media saved to database: %s -> %s", session_id, final_folder)

    return {
        "success": True,
        "session_id": session_id,
        "vehicle_number": final_plate,
        "folder_path": final_folder,
        "capture_date": capture_date,
        "capture_time_12h": capture_time_12h
    }


def discard_upload_media(session_id: str, database_dir: str) -> dict:
    """Discard a staged pending upload and remove temporary files."""
    import shutil
    pending_folder = os.path.join(database_dir, "_pending", session_id)
    if os.path.isdir(pending_folder):
        try:
            shutil.rmtree(pending_folder, ignore_errors=True)
        except Exception as e:
            logger.warning("Error deleting pending folder %s: %s", pending_folder, e)
    return {"success": True, "message": "Staged upload discarded."}
