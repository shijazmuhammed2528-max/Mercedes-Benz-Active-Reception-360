"""
Video Manager — Manages four camera video streams with OpenCV.
Provides thread-safe hardware camera capture sharing, dynamic device assignment,
and clean live camera streaming without silent fallback to recorded videos.
"""
import os
import cv2
import time
import threading
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class HardwareDeviceManager:
    """
    Centralized, thread-safe hardware capture pool.
    Allows multiple visual screens to share any physical webcam device (e.g. Camera 0),
    preventing Windows DirectShow exclusive device lock conflicts.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._devices = {}  # dev_idx -> cv2.VideoCapture
        self._latest_frames = {}  # dev_idx -> np.ndarray
        self._dev_locks = {}
        self._resolutions = {}

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = HardwareDeviceManager()
            return cls._instance

    def open_device(self, dev_idx: int) -> bool:
        with self._lock:
            if dev_idx in self._devices and self._devices[dev_idx] is not None and self._devices[dev_idx].isOpened():
                return True

            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            try:
                cap = cv2.VideoCapture(dev_idx, backend)
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self._devices[dev_idx] = cap
                        self._latest_frames[dev_idx] = frame.copy()
                        self._dev_locks[dev_idx] = threading.Lock()
                        self._resolutions[dev_idx] = (frame.shape[1], frame.shape[0])
                        logger.info("HardwareDeviceManager: Successfully opened Camera %d (%dx%d)",
                                    dev_idx, frame.shape[1], frame.shape[0])
                        return True
                    else:
                        cap.release()
            except Exception as e:
                logger.warning("HardwareDeviceManager: Error probing Camera %d: %s", dev_idx, e)

            logger.info("HardwareDeviceManager: Camera %d is not connected", dev_idx)
            return False

    def is_device_open(self, dev_idx: int) -> bool:
        with self._lock:
            return dev_idx in self._devices and self._devices[dev_idx].isOpened()

    def read_frame(self, dev_idx: int) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            cap = self._devices.get(dev_idx)
            if cap is None or not cap.isOpened():
                return False, None
            d_lock = self._dev_locks.get(dev_idx)

        if d_lock:
            with d_lock:
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._latest_frames[dev_idx] = frame
                    return True, frame
                return False, self._latest_frames.get(dev_idx)
        return False, None

    def get_resolution(self, dev_idx: int) -> tuple[int, int]:
        with self._lock:
            return self._resolutions.get(dev_idx, (640, 480))

    def release_device(self, dev_idx: int):
        with self._lock:
            cap = self._devices.pop(dev_idx, None)
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._latest_frames.pop(dev_idx, None)
            self._dev_locks.pop(dev_idx, None)

    def release_all(self):
        with self._lock:
            for dev_idx, cap in list(self._devices.items()):
                try:
                    cap.release()
                except Exception:
                    pass
            self._devices.clear()
            self._latest_frames.clear()
            self._dev_locks.clear()
            logger.info("HardwareDeviceManager: Released all physical cameras")


class CameraStream:
    """Manages a single camera view screen."""

    def __init__(self, name: str, video_path: str | int, label: str):
        self.name = name
        self.video_path = video_path
        self.label = label
        self.cap = None  # Used only if explicitly streaming a video file
        self.lock = threading.Lock()
        self.current_frame = None
        self.original_frame = None  # Unmodified frame for capture
        self.frame_number = 0
        self.total_frames = 0
        self.fps = 24.0
        self.width = 640
        self.height = 480
        self.is_open = False
        self.is_connected = False
        self.is_playing = False
        self.frame_buffer = deque(maxlen=30)  # Buffer for best-frame selection
        self.hw_manager = HardwareDeviceManager.get_instance()

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
            if is_device:
                dev_idx = self._get_device_index()
                # Open via shared hardware device manager (no silent fallback to video files!)
                opened = self.hw_manager.open_device(dev_idx)
                self.is_connected = opened
                self.is_open = True
                if opened:
                    w, h = self.hw_manager.get_resolution(dev_idx)
                    self.width = w
                    self.height = h
                    logger.info("Screen '%s' connected to physical Camera %d (%dx%d)", self.name, dev_idx, w, h)
                else:
                    self.width = 640
                    self.height = 360
                    logger.info("Screen '%s' assigned to Camera %d (Device not plugged in)", self.name, dev_idx)
                return True
            else:
                # Explicit video file (only if user explicitly chose it)
                if not os.path.isfile(str(self.video_path)):
                    logger.error("Video file not found: %s", self.video_path)
                    return False
                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap is None or not self.cap.isOpened():
                    logger.error("Cannot open stream: %s", self.video_path)
                    return False
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.is_open = True
                self.is_connected = True
                return True

    def set_source(self, new_source: str | int) -> bool:
        """Dynamically update source for this camera screen."""
        with self.lock:
            was_open = self.is_open
            if was_open:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.is_open = False
            self.video_path = new_source
            if was_open:
                return self.open()
            return True

    def _create_no_signal_frame(self) -> np.ndarray:
        """Generate clean, high-tech 'NO CAMERA SIGNAL' visual frame."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (15, 15, 15)
        # Background grid
        for y in range(0, 360, 45):
            cv2.line(frame, (0, y), (640, y), (25, 25, 25), 1)
        for x in range(0, 640, 64):
            cv2.line(frame, (x, 0), (x, 360), (25, 25, 25), 1)

        dev_idx = self._get_device_index() if self._is_device_source() else self.video_path
        cv2.putText(frame, self.label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
        cv2.putText(frame, f"CAMERA {dev_idx}: NO VIDEO SIGNAL", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 140, 255), 2)
        cv2.putText(frame, f"Connect USB Camera {dev_idx} or select Camera 0 from list", (115, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
        # Red disconnected status indicator
        cv2.circle(frame, (640 - 24, 24), 8, (0, 0, 255), -1)
        return frame

    def read_frame(self) -> bool:
        with self.lock:
            if not self.is_open:
                return False

            if self._is_device_source():
                dev_idx = self._get_device_index()
                if self.is_connected:
                    ret, frame = self.hw_manager.read_frame(dev_idx)
                    if ret and frame is not None:
                        self.original_frame = frame.copy()
                        self.current_frame = frame
                        self.frame_number += 1
                        self.frame_buffer.append(self.original_frame.copy())
                        return True
                    else:
                        self.current_frame = self._create_no_signal_frame()
                        return True
                else:
                    self.current_frame = self._create_no_signal_frame()
                    return True
            else:
                if self.cap is None or not self.cap.isOpened():
                    return False
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_number = 0
                    ret, frame = self.cap.read()
                    if not ret:
                        return False
                self.original_frame = frame.copy()
                self.current_frame = frame
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.frame_buffer.append(self.original_frame.copy())
                return True

    def get_display_frame(self) -> np.ndarray | None:
        """Get current frame with detection overlays drawn."""
        with self.lock:
            if self.current_frame is None:
                if self._is_device_source() and not self.is_connected:
                    return self._create_no_signal_frame()
                return None
            frame = self.current_frame.copy()

        # If device is not connected, return the clean no-signal frame
        if self._is_device_source() and not self.is_connected:
            return frame

        # Draw detection overlay
        if self.detection and self.is_detected:
            bbox = self.detection["bbox"]
            conf = self.detection["confidence"]
            cls_name = self.detection.get("class_name", "vehicle")
            x1, y1, x2, y2 = bbox

            # Bright Green bounding box with 3px border
            color = (0, 255, 102)  # #00FF66 in BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Detection info badge above box
            label = f"{cls_name.upper()} {conf:.1%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, max(th + 12, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 6, max(th + 4, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # Status bar at bottom of bbox
            status_text = f"{self.motion_state}"
            if self.is_view_ok:
                status_text += " | VIEW OK"
            cv2.putText(frame, status_text, (x1, min(frame.shape[0] - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Camera label overlay
        cv2.putText(frame, self.label, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Status indicator dot
        if self.is_detected:
            status_color = (0, 255, 102)  # Green
        elif self.is_connected:
            status_color = (0, 140, 255)  # Amber / Blue
        else:
            status_color = (0, 0, 255)    # Red

        cv2.circle(frame, (frame.shape[1] - 20, 22), 9, status_color, -1)
        cv2.circle(frame, (frame.shape[1] - 20, 22), 9, (255, 255, 255), 1)

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
            self.is_connected = False
            self.current_frame = None
            self.original_frame = None

    def get_status(self) -> dict:
        is_dev = self._is_device_source()
        return {
            "name": self.name,
            "label": self.label,
            "is_open": self.is_open,
            "is_connected": self.is_connected,
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
    """Manages all four camera screens and hardware device access."""

    def __init__(self, video_sources: dict, camera_labels: dict):
        self.cameras = {}
        for name, path in video_sources.items():
            label = camera_labels.get(name, name.upper())
            self.cameras[name] = CameraStream(name, path, label)

    def set_screen_source(self, screen_name: str, new_source: str | int) -> bool:
        """Update a single screen's assigned camera source."""
        if screen_name in self.cameras:
            return self.cameras[screen_name].set_source(new_source)
        return False

    def get_sources(self) -> dict:
        """Return dict of screen to source assignments."""
        return {name: str(cam.video_path) for name, cam in self.cameras.items()}

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
        HardwareDeviceManager.get_instance().release_all()

    def get_camera(self, name: str) -> CameraStream | None:
        return self.cameras.get(name)

    def get_all_status(self) -> dict:
        return {name: cam.get_status() for name, cam in self.cameras.items()}

    def all_open(self) -> bool:
        return all(cam.is_open for cam in self.cameras.values())
