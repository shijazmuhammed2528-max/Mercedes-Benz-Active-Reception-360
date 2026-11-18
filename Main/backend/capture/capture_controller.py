"""
Capture Controller — Orchestrates the entire capture pipeline.
Manages video playback, YOLO detection, motion tracking, view quality,
and automatic image capture across all four cameras.
"""
import os
import cv2
import json
import time
import uuid
import threading
import logging
import numpy as np
from datetime import datetime

from backend.capture.video_manager import VideoManager
from backend.capture.capture_state import CaptureState
from backend.vision.yolo_detector import detect_vehicles, init_yolo, is_model_loaded
from backend.vision.motion_detector import MotionDetector
from backend.vision.view_quality import ViewQualityValidator
from backend.vision.frame_selector import select_best_frame, compute_sharpness
from backend.vision.plate_recognition import recognize_plate, init_ocr, is_ocr_available

logger = logging.getLogger(__name__)

CAMERA_NAMES = ["front", "back", "left", "right"]


class CaptureController:
    """Main controller that ties everything together."""

    def __init__(self, config: dict, db_manager):
        self.config = config
        self.db = db_manager

        # Video manager
        from config import VIDEO_SOURCES, CAMERA_LABELS
        self.video_manager = VideoManager(VIDEO_SOURCES, CAMERA_LABELS)

        # Motion detectors (one per camera)
        self.motion_detectors = {}
        for name in CAMERA_NAMES:
            self.motion_detectors[name] = MotionDetector(
                center_threshold=config.get("center_move_threshold", 15.0),
                size_threshold=config.get("size_change_threshold", 0.08),
                stationary_duration=config.get("stationary_duration", 2.0),
            )

        # View quality validator
        self.view_validator = ViewQualityValidator(
            margin_ratio=config.get("view_margin_ratio", 0.03),
            min_area_ratio=config.get("min_detection_ratio", 0.05),
        )

        # State
        self.capture_state = CaptureState()
        self.session_id = None
        self.is_running = False
        self.is_paused = False
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._captured = False
        self._capture_result = None
        self._plate_text = ""
        self._last_capture_time = 0
        self._ready_start_time = None

        # Frame streaming
        self._display_frames = {name: None for name in CAMERA_NAMES}

        # YOLO init
        self._yolo_ready = False

    def initialize_yolo(self) -> bool:
        """Load YOLO model."""
        try:
            from config import get_device
            device = get_device()
        except Exception:
            device = "cpu"
        model = self.config.get("yolo_model", "yolov8n.pt")
        success = init_yolo(model, device)
        self._yolo_ready = success
        return success

    def start(self) -> dict:
        """Start a new capture session."""
        if self.is_running:
            return {"success": False, "error": "Session already running"}

        # Initialize YOLO if needed
        if not self._yolo_ready:
            if not self.initialize_yolo():
                return {"success": False, "error": "Failed to initialize YOLO model"}

        # Initialize OCR
        if not is_ocr_available():
            init_ocr()

        # Open all videos
        self.video_manager.reset_all()
        results = self.video_manager.open_all()
        failed = [name for name, ok in results.items() if not ok]
        if failed:
            return {"success": False, "error": f"Failed to open videos: {', '.join(failed)}"}

        # Reset state
        self.session_id = str(uuid.uuid4())[:8]
        self._captured = False
        self._capture_result = None
        self._plate_text = ""
        self._ready_start_time = None
        self._stop_event.clear()
        self.is_paused = False
        for md in self.motion_detectors.values():
            md.reset()

        self.capture_state.set("MONITORING")
        self.is_running = True

        # Start processing thread
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

        logger.info("Capture session started: %s", self.session_id)
        return {"success": True, "session_id": self.session_id}

    def pause(self) -> dict:
        if not self.is_running:
            return {"success": False, "error": "No active session"}
        self.is_paused = True
        self.capture_state.set("PAUSED")
        return {"success": True}

    def resume(self) -> dict:
        if not self.is_running:
            return {"success": False, "error": "No active session"}
        self.is_paused = False
        self.capture_state.set("MONITORING")
        return {"success": True}

    def stop(self) -> dict:
        if not self.is_running:
            return {"success": False, "error": "No active session"}
        self._stop_event.set()
        self.is_running = False
        self.is_paused = False
        self.capture_state.set("STOPPED")
        self.video_manager.release_all()
        logger.info("Capture session stopped")
        return {"success": True}

    def reset(self) -> dict:
        """Reset current session without deleting saved data."""
        if self.is_running:
            self._stop_event.set()
            time.sleep(0.3)
        self.is_running = False
        self.is_paused = False
        self._captured = False
        self._capture_result = None
        self._ready_start_time = None
        self.video_manager.reset_all()
        for md in self.motion_detectors.values():
            md.reset()
        self.capture_state.set("INITIALIZING")
        self._display_frames = {name: None for name in CAMERA_NAMES}
        self.video_manager.release_all()
        logger.info("Session reset")
        return {"success": True}

    def set_plate(self, plate_text: str) -> dict:
        """Manually set/correct plate number."""
        self._plate_text = plate_text.strip().upper()
        # Update saved session if exists
        if self._capture_result and self._capture_result.get("session_id"):
            sid = self._capture_result["session_id"]
            self.db.update_vehicle_number(sid, self._plate_text)
            # Update metadata.json
            folder = self._capture_result.get("folder_path")
            if folder:
                meta_path = os.path.join(folder, "metadata.json")
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        meta["vehicle_number"] = self._plate_text
                        meta["vehicle_number_normalized"] = self._plate_text.replace(" ", "").replace("-", "")
                        with open(meta_path, "w") as f:
                            json.dump(meta, f, indent=2)
                    except Exception as e:
                        logger.error("Failed to update metadata: %s", e)
            # Rename folder
            if folder and os.path.isdir(folder):
                try:
                    parent = os.path.dirname(folder)
                    ts = self._capture_result.get("capture_time_12h", "").replace(":", "-").replace(" ", "_")
                    dt = self._capture_result.get("capture_date", "")
                    norm = self._plate_text.replace(" ", "_")
                    new_name = f"{norm}_{dt}_{ts}"
                    new_path = os.path.join(parent, new_name)
                    if not os.path.exists(new_path):
                        os.rename(folder, new_path)
                        self._capture_result["folder_path"] = new_path
                        # Update DB folder_path and image paths
                        conn = self.db._get_conn()
                        conn.execute("""
                            UPDATE vehicle_sessions 
                            SET folder_path=?, front_image_path=?, back_image_path=?, left_image_path=?, right_image_path=?
                            WHERE session_id=?
                        """, (
                            new_path,
                            os.path.join(new_path, "front.jpg"),
                            os.path.join(new_path, "back.jpg"),
                            os.path.join(new_path, "left.jpg"),
                            os.path.join(new_path, "right.jpg"),
                            sid
                        ))
                        conn.commit()
                        conn.close()
                except Exception as e:
                    logger.error("Failed to rename folder: %s", e)
        return {"success": True, "plate": self._plate_text}

    def get_status(self) -> dict:
        cameras = self.video_manager.get_all_status()
        conditions = {}
        for name in CAMERA_NAMES:
            cam = self.video_manager.get_camera(name)
            conditions[name] = {
                "detected": cam.is_detected if cam else False,
                "view_ok": cam.is_view_ok if cam else False,
                "stationary": cam.is_stationary if cam else False,
                "confidence": cam.confidence if cam else 0.0,
                "motion_state": cam.motion_state if cam else "WAITING",
            }

        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "captured": self._captured,
            "capture_state": self.capture_state.to_dict(),
            "session_id": self.session_id,
            "cameras": cameras,
            "conditions": conditions,
            "plate_text": self._plate_text,
            "capture_result": self._capture_result,
            "yolo_ready": self._yolo_ready,
        }

    def get_display_frame(self, camera: str) -> np.ndarray | None:
        return self._display_frames.get(camera)

    # ─── Processing Loop ──────────────────────────────────────

    def _processing_loop(self):
        """Main processing loop — runs in background thread."""
        process_fps = self.config.get("yolo_process_fps", 8)
        frame_interval = 1.0 / process_fps
        confidence = self.config.get("yolo_confidence", 0.45)
        vehicle_classes = self.config.get("yolo_vehicle_classes", [2, 3, 5, 7])
        inference_size = self.config.get("yolo_inference_size", 640)

        logger.info("Processing loop started (FPS: %d)", process_fps)

        while not self._stop_event.is_set():
            loop_start = time.time()

            if self.is_paused:
                time.sleep(0.1)
                continue

            # Read frames from all cameras
            all_read = True
            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam and cam.is_open:
                    if not cam.read_frame():
                        # Video ended — loop it
                        cam.reset()
                        cam.open()
                        all_read = False

            if not all_read:
                time.sleep(0.05)
                continue

            # Run YOLO on each camera
            any_detected = False
            all_detected = True
            all_view_ok = True
            all_stationary = True

            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam is None or cam.current_frame is None:
                    all_detected = False
                    continue

                # YOLO detection
                detections = detect_vehicles(
                    cam.current_frame,
                    confidence=confidence,
                    vehicle_classes=vehicle_classes,
                    inference_size=inference_size
                )

                if detections:
                    # Take the largest / highest confidence detection
                    best = max(detections, key=lambda d: d["confidence"])
                    cam.detection = best
                    cam.is_detected = True
                    cam.confidence = best["confidence"]
                    any_detected = True

                    # Motion detection
                    motion_state = self.motion_detectors[name].update(best["bbox"])
                    cam.motion_state = motion_state
                    cam.is_stationary = self.motion_detectors[name].is_stationary

                    # View quality
                    vq = self.view_validator.validate(best["bbox"], cam.width, cam.height)
                    cam.view_quality = vq
                    cam.is_view_ok = vq["passed"]

                    if not cam.is_view_ok:
                        all_view_ok = False
                    if not cam.is_stationary:
                        all_stationary = False
                else:
                    cam.detection = None
                    cam.is_detected = False
                    cam.confidence = 0.0
                    cam.motion_state = "WAITING"
                    cam.is_view_ok = False
                    cam.is_stationary = False
                    self.motion_detectors[name].update(None)
                    all_detected = False

                # Update display frame
                self._display_frames[name] = cam.get_display_frame()

            # Update capture state
            if not self._captured:
                if not any_detected:
                    self._ready_start_time = None
                    self.capture_state.set("MONITORING", "Waiting for vehicle...")
                elif not all_detected:
                    self._ready_start_time = None
                    self.capture_state.set("VEHICLE_DETECTED", "Vehicle partially visible")
                elif not all_stationary:
                    self._ready_start_time = None
                    # Check if any stopping
                    stopping = any(
                        self.video_manager.get_camera(n).motion_state == "STOPPING"
                        for n in CAMERA_NAMES
                    )
                    if stopping:
                        self.capture_state.set("VEHICLE_STOPPING", "Vehicle is slowing down...")
                    else:
                        self.capture_state.set("VEHICLE_MOVING", "Vehicle is moving")
                elif not all_view_ok:
                    self._ready_start_time = None
                    self.capture_state.set("CHECKING_VIEWS", "Checking all four views...")
                else:
                    # ALL CONDITIONS MET (Full vehicle detected, view quality verified, stationary in all 4 views)
                    now_ts = time.time()
                    if self._ready_start_time is None:
                        self._ready_start_time = now_ts

                    delay_needed = float(self.config.get("capture_delay_seconds", 1.0))
                    elapsed_ready = now_ts - self._ready_start_time

                    if elapsed_ready >= delay_needed:
                        cooldown = self.config.get("capture_cooldown", 5.0)
                        if now_ts - self._last_capture_time > cooldown:
                            self._do_capture()
                            self._ready_start_time = None
                    else:
                        remaining = max(0.1, delay_needed - elapsed_ready)
                        self.capture_state.set("CAPTURE_READY", f"Full view verified — capturing in {remaining:.1f}s...")

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Processing loop ended")

    def _do_capture(self):
        """Capture images from all four cameras."""
        self.capture_state.set("CAPTURING", "Capturing images...")
        logger.info("Starting capture...")

        try:
            now = datetime.now()
            capture_date = now.strftime("%Y-%m-%d")
            capture_time_12h = now.strftime("%I:%M:%S %p")
            capture_timestamp = now.isoformat()
            session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            # Attempt OCR on front camera
            front_cam = self.video_manager.get_camera("front")
            plate_result = {"text": "", "success": False}
            if front_cam and front_cam.original_frame is not None:
                bbox = front_cam.detection["bbox"] if front_cam.detection else None
                plate_result = recognize_plate(front_cam.original_frame, bbox)

            plate_text = plate_result.get("text", "") if plate_result.get("success") else ""
            if not plate_text:
                plate_text = "UNKNOWN"
            self._plate_text = plate_text

            # Create folder
            from config import DATABASE_DIR
            plate_norm = plate_text.replace(" ", "_").replace("-", "_")
            time_part = now.strftime("%I-%M-%S_%p")
            folder_name = f"{plate_norm}_{capture_date}_{time_part}"
            folder_path = os.path.join(DATABASE_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Capture and save images
            confidences = {}
            min_sharpness = self.config.get("min_sharpness", 30.0)

            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam is None:
                    continue

                # Select best frame from buffer
                frames = list(cam.frame_buffer)
                if frames:
                    best_frame, sharpness, _ = select_best_frame(frames, min_sharpness)
                else:
                    best_frame = cam.original_frame
                    sharpness = compute_sharpness(best_frame) if best_frame is not None else 0

                if best_frame is not None:
                    img_path = os.path.join(folder_path, f"{name}.jpg")
                    cv2.imwrite(img_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logger.info("Saved %s (sharpness: %.1f)", img_path, sharpness)

                confidences[name] = cam.confidence

            # Save metadata
            metadata = {
                "vehicle_number": plate_text,
                "vehicle_number_normalized": plate_text.replace(" ", "").replace("-", ""),
                "capture_date": capture_date,
                "capture_time_12h": capture_time_12h,
                "capture_timestamp": capture_timestamp,
                "session_id": session_id,
                "camera_count": 4,
                "capture_status": "success",
                "images": {name: f"{name}.jpg" for name in CAMERA_NAMES},
                "camera_source_videos": {
                    "front": "front.mp4", "back": "back.mp4",
                    "left": "left.mp4", "right": "right.mp4"
                },
                "detection_confidence": confidences,
            }
            meta_path = os.path.join(folder_path, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Insert into database
            db_data = {
                "session_id": session_id,
                "vehicle_number": plate_text,
                "vehicle_number_normalized": plate_text.replace(" ", "").replace("-", ""),
                "capture_date": capture_date,
                "capture_timestamp": capture_timestamp,
                "capture_time_12h": capture_time_12h,
                "folder_path": folder_path,
                "front_image_path": os.path.join(folder_path, "front.jpg"),
                "back_image_path": os.path.join(folder_path, "back.jpg"),
                "left_image_path": os.path.join(folder_path, "left.jpg"),
                "right_image_path": os.path.join(folder_path, "right.jpg"),
                "front_confidence": confidences.get("front", 0),
                "back_confidence": confidences.get("back", 0),
                "left_confidence": confidences.get("left", 0),
                "right_confidence": confidences.get("right", 0),
                "capture_status": "success",
            }
            self.db.insert_session(db_data)

            self._captured = True
            self._last_capture_time = time.time()
            self._capture_result = {
                "session_id": session_id,
                "vehicle_number": plate_text,
                "capture_date": capture_date,
                "capture_time_12h": capture_time_12h,
                "folder_path": folder_path,
                "confidences": confidences,
            }
            self.capture_state.set("SAVED", "Capture saved successfully!")
            logger.info("Capture completed: %s -> %s", session_id, folder_path)

        except Exception as e:
            logger.error("Capture failed: %s", e, exc_info=True)
            self.capture_state.set("ERROR", f"Capture failed: {str(e)}")
