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
        from config import VIDEO_SOURCES, CAMERA_LABELS, load_config
        cfg = load_config()
        sources = dict(VIDEO_SOURCES)
        saved_sources = cfg.get("camera_sources")
        if saved_sources and isinstance(saved_sources, dict):
            for k, v in saved_sources.items():
                if k in sources:
                    if str(v).isdigit():
                        sources[k] = int(v)
                    elif str(v).startswith("video"):
                        sources[k] = VIDEO_SOURCES.get(k, v)
                    else:
                        sources[k] = v
        self.video_manager = VideoManager(sources, CAMERA_LABELS)

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
        self._lock = threading.RLock()
        self._captured = False
        self._pending_capture = None
        self._capture_result = None
        self._plate_text = ""
        self._last_capture_time = 0
        self._ready_start_time = None

        # Frame streaming
        self._display_frames = {name: None for name in CAMERA_NAMES}

        # YOLO init
        self._yolo_ready = False
        self.auto_capture_enabled = True

    def set_auto_capture(self, enabled: bool) -> bool:
        """Enable or disable automatic capture upon validity check."""
        with self._lock:
            self.auto_capture_enabled = bool(enabled)
            logger.info("AutoCapture enabled set to: %s", self.auto_capture_enabled)
            return self.auto_capture_enabled

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

    def update_config(self, new_config: dict):
        """Update runtime configuration dynamically."""
        with self._lock:
            self.config.update(new_config)
            for md in self.motion_detectors.values():
                md.center_threshold = float(self.config.get("center_move_threshold", 15.0))
                md.size_threshold = float(self.config.get("size_change_threshold", 0.08))
                md.stationary_duration = float(self.config.get("stationary_duration", 2.0))
            self.view_validator.margin_ratio = float(self.config.get("view_margin_ratio", 0.03))
            self.view_validator.min_area_ratio = float(self.config.get("min_detection_ratio", 0.05))
            logger.info("Runtime config updated: capture_delay=%.1fs, stationary=%.1fs",
                        float(self.config.get("capture_delay_seconds", 1.0)),
                        float(self.config.get("stationary_duration", 2.0)))

    def set_camera_sources(self, mode_or_sources) -> dict:
        """Switch camera sources between live camera device indices (0,1,2,3) or video files."""
        from config import VIDEO_SOURCES, CAMERA_DEVICE_MAP
        with self._lock:
            if mode_or_sources == "live":
                sources = dict(CAMERA_DEVICE_MAP)
            elif mode_or_sources == "video":
                sources = dict(VIDEO_SOURCES)
            elif isinstance(mode_or_sources, dict):
                sources = mode_or_sources
            else:
                return {"success": False, "error": f"Unknown source mode: {mode_or_sources}"}

            self.video_manager.set_sources(sources)
            logger.info("Camera sources switched to %s: %s", mode_or_sources, sources)
            return {"success": True, "sources": {k: str(v) for k, v in sources.items()}, "mode": mode_or_sources}

    def assign_screen_camera(self, screen_name: str, source: str | int) -> dict:
        """Assign a specific physical camera (0,1,2...) or video file to one of the 4 reception screens."""
        from config import VIDEO_SOURCES, save_config, load_config
        with self._lock:
            # Resolve actual source
            actual_source = source
            if str(source).startswith("video"):
                actual_source = VIDEO_SOURCES.get(screen_name, str(source))
            elif isinstance(source, str) and source.strip().isdigit():
                actual_source = int(source.strip())

            success = self.video_manager.set_screen_source(screen_name, actual_source)

            # Persist assignment in app config
            cfg = load_config()
            sources_map = cfg.get("camera_sources", {})
            sources_map[screen_name] = str(source)
            cfg["camera_sources"] = sources_map
            save_config(cfg)
            self.config["camera_sources"] = sources_map

            logger.info("Screen '%s' assigned to camera source '%s' (success=%s, is_open=%s)",
                        screen_name, source, success,
                        self.video_manager.cameras.get(screen_name, None).is_open if screen_name in self.video_manager.cameras else False)

            return {
                "success": True,
                "screen": screen_name,
                "source": str(source),
                "is_open": self.video_manager.cameras[screen_name].is_open if screen_name in self.video_manager.cameras else False,
                "mapping": self.get_camera_mapping()
            }

    def get_camera_mapping(self) -> dict:
        """Get current screen-to-camera assignments and status."""
        mapping = {}
        for name, cam in self.video_manager.cameras.items():
            src_str = str(cam.video_path)
            is_dev = cam._is_device_source()
            mapping[name] = {
                "source": src_str,
                "is_device": is_dev,
                "device_index": cam._get_device_index() if is_dev else None,
                "is_open": cam.is_open,
                "label": cam.label,
            }
        return mapping

    def start(self) -> dict:
        """Start a new capture session."""
        with self._lock:
            if self.is_running:
                self._captured = False
                self.capture_state.set("MONITORING")
                return {"success": True, "session_id": self.session_id, "already_running": True}

            self.is_running = True  # Mark immediately to prevent race conditions from concurrent stream requests

            # Initialize YOLO if needed
            if not self._yolo_ready:
                if not self.initialize_yolo():
                    self.is_running = False
                    return {"success": False, "error": "Failed to initialize YOLO model"}

            # Initialize OCR in background so camera startup is immediate
            if not is_ocr_available():
                threading.Thread(target=init_ocr, daemon=True).start()

            # Open all cameras/videos
            self.video_manager.reset_all()
            self.video_manager.open_all()

            # Seed display frames immediately so every stream gets valid visuals on frame 0
            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam and cam.current_frame is not None:
                    disp = cam.get_display_frame()
                    self._display_frames[name] = disp if disp is not None else cam.current_frame

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

            # Start single processing thread
            if self._thread is None or not self._thread.is_alive():
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
        self._stop_event.set()
        self.is_running = False
        self.is_paused = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self.capture_state.set("STOPPED")
        self.video_manager.release_all()
        self._display_frames = {name: None for name in CAMERA_NAMES}
        logger.info("Capture session stopped and physical cameras turned OFF")
        return {"success": True}

    def reset(self) -> dict:
        """Reset current session without deleting saved data."""
        if self.is_running:
            self._stop_event.set()
            time.sleep(0.3)
        self.is_running = False
        self.is_paused = False
        self._captured = False
        if self._pending_capture:
            p_folder = self._pending_capture.get("pending_folder")
            if p_folder and os.path.isdir(p_folder):
                try:
                    import shutil
                    shutil.rmtree(p_folder, ignore_errors=True)
                except Exception:
                    pass
        self._pending_capture = None
        self._capture_result = None
        self._ready_start_time = None
        self.video_manager.release_all()
        for md in self.motion_detectors.values():
            md.reset()
        self.capture_state.set("INITIALIZING")
        self._display_frames = {name: None for name in CAMERA_NAMES}
        logger.info("Session reset and physical cameras turned OFF")
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

        delay_needed = float(self.config.get("capture_delay_seconds", 1.0))
        ready_countdown = None
        if self._ready_start_time is not None and not self._captured:
            elapsed = time.time() - self._ready_start_time
            ready_countdown = max(0.0, round(delay_needed - elapsed, 1))

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
            "pending_capture": self._pending_capture,
            "yolo_ready": self._yolo_ready,
            "auto_capture": self.auto_capture_enabled,
            "capture_delay_seconds": delay_needed,
            "ready_countdown": ready_countdown,
        }

    def get_display_frame(self, camera: str) -> np.ndarray | None:
        frame = self._display_frames.get(camera)
        if frame is not None:
            return frame
        cam = self.video_manager.get_camera(camera)
        if cam and cam.current_frame is not None:
            disp = cam.get_display_frame()
            return disp if disp is not None else cam.current_frame
        return None

    # ─── Processing Loop ──────────────────────────────────────

    def _processing_loop(self):
        """Main processing loop — runs in background thread."""
        process_fps = self.config.get("yolo_process_fps", 30)
        frame_interval = 1.0 / process_fps
        confidence = self.config.get("yolo_confidence", 0.45)
        vehicle_classes = self.config.get("yolo_vehicle_classes", [2, 3, 5, 7])
        inference_size = self.config.get("yolo_inference_size", 640)

        logger.info("Processing loop started on RTX 4050 GPU (Target FPS: %d)", process_fps)

        while not self._stop_event.is_set():
            loop_start = time.time()
            process_fps = max(1, int(self.config.get("yolo_process_fps", 30)))
            frame_interval = 1.0 / process_fps
            confidence = float(self.config.get("yolo_confidence", 0.45))
            vehicle_classes = self.config.get("yolo_vehicle_classes", [2, 3, 5, 7])
            inference_size = int(self.config.get("yolo_inference_size", 640))

            if self.is_paused:
                time.sleep(0.1)
                continue

            # Read frames from all cameras
            all_read = True
            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam and cam.is_open:
                    if not cam.read_frame():
                        all_read = False
                    else:
                        disp = cam.get_display_frame()
                        if disp is not None:
                            self._display_frames[name] = disp
                else:
                    all_read = False

            # Run YOLO and update visual display frames for every camera
            # Group cameras by unique source to avoid duplicate YOLO inferences if multiple screens share the same camera
            any_detected = False
            all_detected = True
            all_view_ok = True
            all_stationary = True

            unique_detections = {}
            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam is None or cam.current_frame is None or not cam.is_open or not cam.is_connected:
                    continue
                source_key = cam._get_source_key()
                if source_key not in unique_detections:
                    unique_detections[source_key] = detect_vehicles(
                        cam.current_frame,
                        confidence=confidence,
                        vehicle_classes=vehicle_classes,
                        inference_size=inference_size
                    )

            for name in CAMERA_NAMES:
                cam = self.video_manager.get_camera(name)
                if cam is None or cam.current_frame is None or not cam.is_open:
                    all_detected = False
                    continue

                # If camera device is not connected, skip YOLO and show clean NO SIGNAL frame
                if not cam.is_connected:
                    cam.detection = None
                    cam.is_detected = False
                    cam.confidence = 0.0
                    cam.motion_state = "DISCONNECTED"
                    cam.is_view_ok = False
                    cam.is_stationary = False
                    self.motion_detectors[name].update(None)
                    all_detected = False
                    self._display_frames[name] = cam.get_display_frame()
                    continue

                # Retrieve YOLO detection for this camera's hardware/stream source
                source_key = cam._get_source_key()
                detections = unique_detections.get(source_key, [])

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
            if self.auto_capture_enabled and not self._captured and self._pending_capture is None:
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
            elif not self.auto_capture_enabled and self._pending_capture is None and not self._captured:
                self._ready_start_time = None
                self.capture_state.set("MONITORING", "AutoCapture paused (Manual trigger available)")

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Processing loop ended")

    def _do_capture(self):
        """
        Stage captured images from all four cameras and prepare for user confirmation.
        Does NOT insert into database until the user explicitly confirms!
        """
        self.capture_state.set("CAPTURING", "Capturing vehicle images...")
        logger.info("Starting vehicle capture for user confirmation...")

        try:
            now = datetime.now()
            capture_date = now.strftime("%Y-%m-%d")
            capture_time_12h = now.strftime("%I:%M:%S %p")
            capture_timestamp = now.isoformat()
            session_id = f"session_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            # Create pending staging folder
            from config import DATABASE_DIR
            pending_folder = os.path.join(DATABASE_DIR, "_pending", session_id)
            os.makedirs(pending_folder, exist_ok=True)

            # Ensure OCR is initialized
            if not is_ocr_available():
                init_ocr()

            # Capture and save sharpest frames to pending folder
            confidences = {}
            sharpness_scores = {}
            min_sharpness = self.config.get("min_sharpness", 30.0)
            captured_frames = {}

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
                    captured_frames[name] = best_frame
                    img_path = os.path.join(pending_folder, f"{name}.jpg")
                    cv2.imwrite(img_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logger.info("Pending image saved: %s (sharpness: %.1f)", img_path, sharpness)

                confidences[name] = round(float(cam.confidence or 0.0), 3)
                sharpness_scores[name] = round(float(sharpness), 1)

            # Automatic Number Plate Identification across all cameras
            detected_plate = ""
            best_ocr_conf = 0.0

            # Scan front and back cameras first (standard license plate locations), then left/right
            for name in ["front", "back", "left", "right"]:
                frame_to_ocr = captured_frames.get(name)
                if frame_to_ocr is None:
                    continue

                cam = self.video_manager.get_camera(name)
                bbox = cam.detection["bbox"] if (cam and cam.detection and "bbox" in cam.detection) else None

                plate_res = recognize_plate(frame_to_ocr, bbox)
                if plate_res.get("success") and plate_res.get("text"):
                    text = plate_res["text"].strip().upper()
                    conf = float(plate_res.get("confidence", 0.0))
                    # Prioritize Indian pattern or highest OCR confidence
                    if conf > best_ocr_conf or not detected_plate:
                        detected_plate = text
                        best_ocr_conf = conf
                        logger.info("Auto-identified license plate from %s: %s (conf: %.2f)", name, text, conf)
                        if conf >= 0.55:
                            break

            plate_text = detected_plate if detected_plate else "UNKNOWN"
            self._plate_text = plate_text

            # Store pending capture metadata awaiting confirmation
            self._pending_capture = {
                "session_id": session_id,
                "vehicle_number": plate_text,
                "capture_date": capture_date,
                "capture_time_12h": capture_time_12h,
                "capture_timestamp": capture_timestamp,
                "confidences": confidences,
                "sharpness": sharpness_scores,
                "pending_folder": pending_folder,
                "images": {
                    name: f"/api/capture/pending/image/{name}?sid={session_id}&t={int(time.time()*1000)}"
                    for name in CAMERA_NAMES
                }
            }

            self.capture_state.set("AWAITING_CONFIRMATION", f"Vehicle captured ({plate_text}) — Please review and confirm to save...")
            logger.info("Vehicle captured. Awaiting confirmation for session %s (Plate: %s)", session_id, plate_text)

        except Exception as e:
            logger.error("Capture staging failed: %s", e, exc_info=True)
            self.capture_state.set("ERROR", f"Capture failed: {str(e)}")

    def confirm_capture(self, session_id: str = None, confirmed_vehicle_number: str = None) -> dict:
        """
        User confirmed vehicle capture!
        Writes record into SQLite database, saves images to permanent folder, and records metadata.
        """
        with self._lock:
            if not self._pending_capture:
                return {"success": False, "error": "No pending capture awaiting confirmation"}

            p = self._pending_capture
            if session_id and p.get("session_id") != session_id:
                return {"success": False, "error": "Session ID mismatch for pending capture"}

            from config import DATABASE_DIR
            pending_folder = p["pending_folder"]
            plate_text = confirmed_vehicle_number.strip().upper() if confirmed_vehicle_number else p["vehicle_number"]
            if not plate_text:
                plate_text = "UNKNOWN"

            now = datetime.now()
            time_part = now.strftime("%I-%M-%S_%p")
            plate_norm = plate_text.replace(" ", "_").replace("-", "_")
            folder_name = f"{plate_norm}_{p['capture_date']}_{time_part}"
            final_folder_path = os.path.join(DATABASE_DIR, folder_name)
            os.makedirs(final_folder_path, exist_ok=True)

            # Name files with: {numberplate}_{date}_{time}_{camera}.jpg
            # (e.g. KL_01_AL_0009_2026-11-20_12-06-31_AM_front.jpg)
            # and ALSO keep {camera}.jpg for fast API lookup & backward compatibility
            import shutil
            image_paths = {}
            saved_filenames = {}
            for name in CAMERA_NAMES:
                src_img = os.path.join(pending_folder, f"{name}.jpg")
                named_filename = f"{plate_norm}_{p['capture_date']}_{time_part}_{name}.jpg"
                dst_named = os.path.join(final_folder_path, named_filename)
                dst_standard = os.path.join(final_folder_path, f"{name}.jpg")

                if os.path.isfile(src_img):
                    try:
                        shutil.copy2(src_img, dst_named)
                        shutil.copy2(src_img, dst_standard)
                        image_paths[name] = dst_named
                        saved_filenames[name] = named_filename
                    except Exception as e:
                        logger.error("Failed copying %s to %s: %s", src_img, dst_named, e)

            # Save metadata.json
            metadata = {
                "vehicle_number": plate_text,
                "vehicle_number_normalized": plate_text.replace(" ", "").replace("-", ""),
                "capture_date": p["capture_date"],
                "capture_time_12h": p["capture_time_12h"],
                "capture_timestamp": p["capture_timestamp"],
                "session_id": p["session_id"],
                "camera_count": 4,
                "capture_status": "success",
                "folder_name": folder_name,
                "images": saved_filenames,
                "standard_images": {name: f"{name}.jpg" for name in CAMERA_NAMES},
                "detection_confidence": p["confidences"],
            }
            meta_path = os.path.join(final_folder_path, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Insert into database with the descriptive file paths
            db_data = {
                "session_id": p["session_id"],
                "vehicle_number": plate_text,
                "vehicle_number_normalized": plate_text.replace(" ", "").replace("-", ""),
                "capture_date": p["capture_date"],
                "capture_timestamp": p["capture_timestamp"],
                "capture_time_12h": p["capture_time_12h"],
                "folder_path": final_folder_path,
                "front_image_path": image_paths.get("front", os.path.join(final_folder_path, "front.jpg")),
                "back_image_path": image_paths.get("back", os.path.join(final_folder_path, "back.jpg")),
                "left_image_path": image_paths.get("left", os.path.join(final_folder_path, "left.jpg")),
                "right_image_path": image_paths.get("right", os.path.join(final_folder_path, "right.jpg")),
                "front_confidence": p["confidences"].get("front", 0),
                "back_confidence": p["confidences"].get("back", 0),
                "left_confidence": p["confidences"].get("left", 0),
                "right_confidence": p["confidences"].get("right", 0),
                "capture_status": "success",
            }
            self.db.insert_session(db_data)

            # Clean up pending folder
            try:
                shutil.rmtree(pending_folder, ignore_errors=True)
            except Exception:
                pass

            self._captured = True
            self._last_capture_time = time.time()
            self._capture_result = {
                "session_id": p["session_id"],
                "vehicle_number": plate_text,
                "capture_date": p["capture_date"],
                "capture_time_12h": p["capture_time_12h"],
                "folder_path": final_folder_path,
                "confidences": p["confidences"],
            }
            self._plate_text = plate_text
            self._pending_capture = None
            self.capture_state.set("SAVED", f"Vehicle {plate_text} saved to database successfully!")
            logger.info("Vehicle confirmed & saved to database: %s -> %s", p["session_id"], final_folder_path)

            return {
                "success": True,
                "session_id": p["session_id"],
                "vehicle_number": plate_text,
                "folder_path": final_folder_path,
                "capture_date": p["capture_date"],
                "capture_time_12h": p["capture_time_12h"]
            }

    def discard_capture(self, session_id: str = None) -> dict:
        """
        User discarded the pending capture.
        Deletes temporary pending files, does NOT touch database, and resumes monitoring.
        """
        with self._lock:
            if not self._pending_capture:
                return {"success": True, "message": "No pending capture to discard"}

            p = self._pending_capture
            pending_folder = p.get("pending_folder")
            if pending_folder and os.path.isdir(pending_folder):
                try:
                    import shutil
                    shutil.rmtree(pending_folder, ignore_errors=True)
                except Exception as e:
                    logger.warning("Error deleting pending folder %s: %s", pending_folder, e)

            self._pending_capture = None
            self._captured = False
            self._ready_start_time = None
            self._last_capture_time = time.time()
            self.capture_state.set("MONITORING", "Capture discarded. Resuming vehicle monitoring...")
            logger.info("Pending capture %s discarded. Resumed monitoring.", p.get("session_id"))
            return {"success": True, "message": "Capture discarded"}

    def get_pending_image(self, camera_name: str) -> tuple[bool, bytes | None]:
        """Return image bytes of a camera from the pending capture."""
        with self._lock:
            if not self._pending_capture:
                return False, None
            pending_folder = self._pending_capture.get("pending_folder")
            if not pending_folder:
                return False, None
            img_path = os.path.join(pending_folder, f"{camera_name}.jpg")
            if os.path.isfile(img_path):
                try:
                    with open(img_path, "rb") as f:
                        return True, f.read()
                except Exception:
                    return False, None
            return False, None
