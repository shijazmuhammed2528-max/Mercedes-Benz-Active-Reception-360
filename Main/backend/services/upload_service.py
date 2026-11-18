"""
Upload Service — Handles uploading local car photos from PC,
analyzing them with YOLO & OCR, and persisting the record to the vehicle database.
"""
import os
import cv2
import json
import uuid
import base64
import logging
from datetime import datetime
import numpy as np

from backend.vision.yolo_detector import detect_vehicles, is_model_loaded, init_yolo
from backend.vision.plate_recognition import recognize_plate, is_ocr_available, init_ocr
from backend.vision.frame_selector import compute_sharpness

logger = logging.getLogger(__name__)


def process_uploaded_vehicle_image(
    image_bytes: bytes,
    camera_view: str = "front",
    database_dir: str = "",
    db_manager = None,
    save_to_database: bool = True,
    vehicle_number_override: str = "",
    yolo_confidence: float = 0.40,
    device: str = "auto"
) -> dict:
    """
    Process an uploaded image:
    1. Decode image with OpenCV
    2. Detect vehicle with YOLO (draw green bounding box)
    3. Run OCR for license plate recognition
    4. Compute image sharpness & quality
    5. Save inspection photo, metadata.json, and SQLite record if save_to_database=True
    """
    # 1. Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"success": False, "error": "Invalid image file. Could not decode."}

    h, w = frame.shape[:2]
    sharpness = round(compute_sharpness(frame), 1)

    # 2. Ensure YOLO & OCR are loaded
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

    # 3. Run YOLO detection
    detections = detect_vehicles(frame, confidence=yolo_confidence)
    vehicle_detected = len(detections) > 0
    best_det = None
    annotated = frame.copy()

    if vehicle_detected:
        best_det = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best_det["bbox"]
        # Draw Mercedes-Benz standard green bounding box (#00FF66 -> BGR: (102, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 102), 3)

        cls_label = f"{best_det.get('class_name', 'CAR').upper()} {int(best_det['confidence'] * 100)}%"
        # Label banner background
        cv2.rectangle(annotated, (x1, max(0, y1 - 32)), (x1 + len(cls_label) * 14 + 16, y1), (0, 0, 0), -1)
        cv2.putText(annotated, cls_label, (x1 + 8, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 102), 2)

    # 4. Run License Plate OCR
    plate_text = ""
    plate_conf = 0.0
    if vehicle_number_override.strip():
        plate_text = vehicle_number_override.strip().upper()
    else:
        bbox = best_det["bbox"] if best_det else None
        plate_res = recognize_plate(frame, bbox=bbox)
        if plate_res.get("text"):
            plate_text = plate_res["text"].strip().upper()
            plate_conf = plate_res.get("confidence", 0.0)

    display_plate = plate_text if plate_text else "UNKNOWN"

    # Generate base64 thumbnail of annotated image for UI preview
    disp_h = 420
    disp_w = int(disp_h * (w / max(1, h)))
    resized_preview = cv2.resize(annotated, (disp_w, disp_h))
    _, buf = cv2.imencode(".jpg", resized_preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preview_b64 = base64.b64encode(buf).decode("utf-8")

    result = {
        "success": True,
        "width": w,
        "height": h,
        "sharpness": sharpness,
        "vehicle_detected": vehicle_detected,
        "detection": best_det,
        "plate_text": display_plate,
        "plate_confidence": plate_conf,
        "camera_view": camera_view.lower(),
        "preview_b64": preview_b64,
        "saved": False,
        "session_id": None,
        "folder_path": None,
    }

    # 5. Persist to Database if requested
    if save_to_database and database_dir and db_manager:
        try:
            now = datetime.now()
            capture_date = now.strftime("%Y-%m-%d")
            capture_time_12h = now.strftime("%I:%M:%S %p")
            capture_timestamp = now.isoformat()
            sid = f"upload_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            # Clean folder name
            norm_plate = display_plate.replace(" ", "_").replace("-", "")
            time_tag = capture_time_12h.replace(":", "-").replace(" ", "_")
            folder_name = f"{norm_plate}_{capture_date}_{time_tag}"
            folder_path = os.path.join(database_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Save clean source image (without overlays)
            view_filename = f"{camera_view.lower()}.jpg" if camera_view.lower() in {"front", "back", "left", "right"} else "front.jpg"
            img_dest = os.path.join(folder_path, view_filename)
            cv2.imwrite(img_dest, frame)

            # Also create placeholder images for missing views so 360 viewer works
            for cam in ["front", "back", "left", "right"]:
                cam_path = os.path.join(folder_path, f"{cam}.jpg")
                if not os.path.isfile(cam_path):
                    # Copy the uploaded image or create clean blank view
                    cv2.imwrite(cam_path, frame)

            # Metadata JSON
            meta = {
                "session_id": sid,
                "vehicle_number": display_plate,
                "vehicle_number_normalized": display_plate.replace(" ", "").replace("-", ""),
                "capture_date": capture_date,
                "capture_timestamp": capture_timestamp,
                "capture_time_12h": capture_time_12h,
                "source": "manual_upload",
                "primary_view": camera_view.lower(),
                "image_resolution": [w, h],
                "sharpness_score": sharpness,
                "vehicle_detected": vehicle_detected,
                "detection_confidence": best_det["confidence"] if best_det else 0.0,
                "detection_class": best_det.get("class_name", "none") if best_det else "none",
                "files": {
                    "front": os.path.join(folder_path, "front.jpg"),
                    "back": os.path.join(folder_path, "back.jpg"),
                    "left": os.path.join(folder_path, "left.jpg"),
                    "right": os.path.join(folder_path, "right.jpg"),
                }
            }
            with open(os.path.join(folder_path, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Insert into SQLite Database
            db_row = {
                "session_id": sid,
                "vehicle_number": display_plate,
                "vehicle_number_normalized": display_plate.replace(" ", "").replace("-", ""),
                "capture_date": capture_date,
                "capture_timestamp": capture_timestamp,
                "capture_time_12h": capture_time_12h,
                "folder_path": folder_path,
                "front_image_path": os.path.join(folder_path, "front.jpg"),
                "back_image_path": os.path.join(folder_path, "back.jpg"),
                "left_image_path": os.path.join(folder_path, "left.jpg"),
                "right_image_path": os.path.join(folder_path, "right.jpg"),
                "front_confidence": best_det["confidence"] if (best_det and camera_view == "front") else 0.85,
                "back_confidence": best_det["confidence"] if (best_det and camera_view == "back") else 0.85,
                "left_confidence": best_det["confidence"] if (best_det and camera_view == "left") else 0.85,
                "right_confidence": best_det["confidence"] if (best_det and camera_view == "right") else 0.85,
                "capture_status": "success",
            }
            db_manager.insert_session(db_row)

            result["saved"] = True
            result["session_id"] = sid
            result["folder_path"] = folder_path
            logger.info("Uploaded vehicle image stored: %s -> %s", sid, folder_path)
        except Exception as e:
            logger.error("Failed to save uploaded vehicle: %s", e, exc_info=True)
            result["save_error"] = str(e)

    return result
