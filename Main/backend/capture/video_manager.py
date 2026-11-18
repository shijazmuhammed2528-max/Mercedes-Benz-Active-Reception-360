"""
Video Manager — Manages four camera video streams with OpenCV.
Provides thread-safe frame access, synchronized playback, and MJPEG streaming.
"""
import os
import cv2
import time
import threading
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages a single camera video stream."""

    def __init__(self, name: str, video_path: str, label: str):
        self.name = name
        self.video_path = video_path
        self.label = label
        self.cap = None
        self.lock = threading.Lock()
        self.current_frame = None
        self.original_frame = None  # Unmodified frame for capture
        self.frame_number = 0
        self.total_frames = 0
        self.fps = 24.0
        self.width = 0
        self.height = 0
        self.is_open = False
        self.is_playing = False
        self.frame_buffer = deque(maxlen=30)  # Buffer for best-frame selection

        # Detection state (updated by capture controller)
        self.detection = None
        self.motion_state = "WAITING"
        self.view_quality = None
        self.is_detected = False
        self.is_view_ok = False
        self.is_stationary = False
        self.confidence = 0.0

    def _is_device_source(self) -> bool:
        if isinstance(self.video_path, int):
            return True
        if isinstance(self.video_path, str) and self.video_path.strip().isdigit():
            return True
        return False

    def _get_device_index(self) -> int:
        if isinstance(self.video_path, int):
            return self.video_path
        return int(str(self.video_path).strip())

    def open(self) -> bool:
        with self.lock:
            is_device = self._is_device_source()
            if not is_device and not os.path.isfile(str(self.video_path)):
                logger.error("Video file not found: %s", self.video_path)
                return False
            try:
                if is_device:
                    dev_idx = self._get_device_index()
                    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
                    self.cap = cv2.VideoCapture(dev_idx, backend)
                else:
                    self.cap = cv2.VideoCapture(self.video_path)

                if not self.cap.isOpened():
                    logger.error("Cannot open stream: %s", self.video_path)
                    return False
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 24.0
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_device else 0
                self.is_open = True
                self.frame_number = 0
                logger.info("Opened %s (source: %s): %dx%d, %.1f fps",
                            self.name, str(self.video_path), self.width, self.height, self.fps)
                return True
            except Exception as e:
                logger.error("Error opening %s: %s", self.video_path, e)
                return False

    def read_frame(self) -> bool:
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return False
            ret, frame = self.cap.read()
            if not ret:
                # Loop video file or mark as ended for device
                if not self._is_device_source():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_number = 0
                return False
            self.original_frame = frame.copy()
            self.current_frame = frame
            self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if not self._is_device_source() else self.frame_number + 1
            # Add to buffer for best-frame selection
            self.frame_buffer.append(self.original_frame.copy())
            return True

    def get_display_frame(self) -> np.ndarray | None:
        """Get current frame with detection overlays drawn."""
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()

        # Draw detection overlay
        if self.detection and self.is_detected:
            bbox = self.detection["bbox"]
            conf = self.detection["confidence"]
            cls_name = self.detection.get("class_name", "vehicle")
            x1, y1, x2, y2 = bbox

            # Green bounding box
            color = (0, 255, 102)  # #00FF66 in BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Detection info
            label = f"{cls_name.upper()} {conf:.1%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Status bar at bottom of bbox
            status_text = f"{self.motion_state}"
            if self.is_view_ok:
                status_text += " | VIEW OK"
            cv2.putText(frame, status_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Camera label overlay
        cv2.putText(frame, self.label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Status indicator
        status_color = (0, 255, 102) if self.is_detected else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, status_color, -1)

        return frame

    def reset(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_number = 0
            self.current_frame = None
            self.original_frame = None
            self.detection = None
            self.motion_state = "WAITING"
            self.view_quality = None
            self.is_detected = False
            self.is_view_ok = False
            self.is_stationary = False
            self.confidence = 0.0
            self.frame_buffer.clear()

    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_open = False
            self.is_playing = False
            self.current_frame = None
            self.original_frame = None

    def get_status(self) -> dict:
        is_dev = self._is_device_source()
        return {
            "name": self.name,
            "label": self.label,
            "is_open": self.is_open,
            "is_playing": self.is_playing,
            "is_device": is_dev,
            "file_exists": True if is_dev else os.path.isfile(str(self.video_path)),
            "source": str(self.video_path),
            "frame_number": self.frame_number,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "is_detected": self.is_detected,
            "confidence": self.confidence,
            "motion_state": self.motion_state,
            "is_view_ok": self.is_view_ok,
            "is_stationary": self.is_stationary,
        }


class VideoManager:
    """Manages all four camera streams."""

    def __init__(self, video_sources: dict, camera_labels: dict):
        self.cameras = {}
        for name, path in video_sources.items():
            label = camera_labels.get(name, name.upper())
            self.cameras[name] = CameraStream(name, path, label)

    def set_sources(self, video_sources: dict):
        """Update sources dynamically."""
        for name, path in video_sources.items():
            if name in self.cameras:
                cam = self.cameras[name]
                was_open = cam.is_open
                if was_open:
                    cam.release()
                cam.video_path = path
                if was_open:
                    cam.open()

    def open_all(self) -> dict:
        results = {}
        for name, cam in self.cameras.items():
            results[name] = cam.open()
        return results

    def reset_all(self):
        for cam in self.cameras.values():
            cam.reset()

    def release_all(self):
        for cam in self.cameras.values():
            cam.release()

    def get_camera(self, name: str) -> CameraStream | None:
        return self.cameras.get(name)

    def get_all_status(self) -> dict:
        return {name: cam.get_status() for name, cam in self.cameras.items()}

    def all_open(self) -> bool:
        return all(cam.is_open for cam in self.cameras.values())
