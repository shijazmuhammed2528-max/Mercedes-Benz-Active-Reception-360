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
CACHE_TTL = 10.0  # seconds

def probe_hardware_cameras(max_devices: int = 6) -> List[Dict[str, Any]]:
    """
    Probe video device indices (0 to max_devices-1) via DirectShow on Windows.
    Returns list of camera info dictionaries.
    """
    discovered = []
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY

    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx, backend)
        is_connected = False
        width, height = 0, 0
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                is_connected = True
                height, width = frame.shape[:2]
            cap.release()

        # Generate user-friendly label
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


def get_available_cameras(video_sources: dict = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get available hardware cameras and fallback prerecorded video sources.
    Uses cached probe results when within CACHE_TTL.
    """
    global _cached_cameras, _cache_timestamp

    with _cache_lock:
        now = time.time()
        if force_refresh or (now - _cache_timestamp > CACHE_TTL) or not _cached_cameras:
            try:
                _cached_cameras = probe_hardware_cameras(max_devices=6)
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

    return {
        "devices": devices,
        "videos": videos,
        "cached": not force_refresh,
    }
