"""
Camera Discovery Service — Enumerates connected hardware video devices
and maps physical USB/integrated webcams for each reception screen.
"""
import os
import cv2
import time
import logging
import threading
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Cache for probed cameras
_cached_cameras: List[Dict[str, Any]] = []
_cache_timestamp: float = 0.0
_cache_lock = threading.Lock()
CACHE_TTL = 10.0  # seconds (quick auto-detection of USB cameras)

def probe_hardware_cameras(max_devices: int = 6, currently_open_indices: set = None) -> List[Dict[str, Any]]:
    """
    Probe video device indices via DirectShow on Windows.
    Quickly discovers connected webcams without blocking or conflicting with active streams.
    """
    discovered = []
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    if currently_open_indices is None:
        currently_open_indices = set()

    from backend.capture.video_manager import HardwareDeviceManager
    hw_mgr = HardwareDeviceManager.get_instance()

    for idx in range(max_devices):
        is_connected = False
        width, height = 0, 0

        # If already open and active in HardwareDeviceManager, don't re-open
        if idx in currently_open_indices or hw_mgr.is_device_open(idx):
            is_connected = True
            width, height = hw_mgr.get_resolution(idx)
        else:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        is_connected = True
                        height, width = frame.shape[:2]
                    cap.release()
            except Exception:
                is_connected = False

        if idx == 0:
            friendly = "Camera 0 (Laptop Main / Integrated)"
        elif idx == 1:
            friendly = "Camera 1 (USB / External Camera)"
        else:
            friendly = f"Camera {idx} (USB Camera {idx})"

        discovered.append({
            "id": str(idx),
            "index": idx,
            "label": friendly,
            "connected": is_connected,
            "resolution": f"{width}x{height}" if is_connected else None,
            "is_device": True,
        })

    return discovered


def get_available_cameras(video_sources: dict = None, force_refresh: bool = False, currently_open_indices: set = None) -> Dict[str, Any]:
    """
    Get available hardware cameras and fallback prerecorded video sources.
    Uses cached probe results when within CACHE_TTL.
    """
    global _cached_cameras, _cache_timestamp

    with _cache_lock:
        now = time.time()
        if force_refresh or (now - _cache_timestamp > CACHE_TTL) or not _cached_cameras:
            try:
                _cached_cameras = probe_hardware_cameras(max_devices=6, currently_open_indices=currently_open_indices)
                _cache_timestamp = now
                logger.info("Probed hardware cameras: %d connected of %d checked",
                            sum(1 for c in _cached_cameras if c["connected"]),
                            len(_cached_cameras))
            except Exception as e:
                logger.error("Error probing cameras: %s", e)
                if not _cached_cameras:
                    # Fallback list if probe threw error
                    _cached_cameras = [
                        {"id": str(i), "index": i, "label": f"Camera {i}" + (" (Laptop Main)" if i == 0 else f" (USB {i})"), "connected": (i < 2), "is_device": True}
                        for i in range(4)
                    ]

        devices = list(_cached_cameras)

    # Prerecorded video options
    videos = []
    if video_sources:
        for cam_name, vpath in video_sources.items():
            exists = os.path.isfile(vpath) if isinstance(vpath, str) else False
            videos.append({
                "id": f"video_{cam_name}",
                "screen": cam_name,
                "path": str(vpath),
                "label": f"Demo Video ({os.path.basename(str(vpath))})",
                "connected": exists,
                "is_device": False,
            })

    # WiFi & IP Camera options
    try:
        from config import load_config
        cfg = load_config()
        wifi_cameras = cfg.get("wifi_cameras", [])
    except Exception:
        wifi_cameras = []

    return {
        "devices": devices,
        "wifi_cameras": wifi_cameras,
        "videos": videos,
        "cached": not force_refresh,
    }
