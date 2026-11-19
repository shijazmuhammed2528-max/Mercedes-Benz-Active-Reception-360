"""
Mercedes-Benz Active Reception — Configuration
Central configuration for the entire application.
"""
import os
import json

# ─── Base Paths ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = r"D:\Project Costal star\Videos"
DATABASE_DIR = r"D:\Project Costal star\Data_base"
DATABASE_FILE = os.path.join(DATABASE_DIR, "vehicle_database.db")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "app_config.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ─── Video Source Mapping ──────────────────────────────────────
VIDEO_SOURCES = {
    "front": os.path.join(VIDEO_DIR, "front.mp4"),
    "back":  os.path.join(VIDEO_DIR, "back.mp4"),
    "left":  os.path.join(VIDEO_DIR, "left.mp4"),
    "right": os.path.join(VIDEO_DIR, "right.mp4"),
}

CAMERA_LABELS = {
    "front": "CAMERA 0 [FRONT]",
    "back":  "CAMERA 1 [BACK]",
    "left":  "CAMERA 2 [LEFT]",
    "right": "CAMERA 3 [RIGHT]",
}

CAMERA_DEVICE_MAP = {
    "front": 0,
    "back":  1,
    "left":  2,
    "right": 3,
}

# ─── YOLO Configuration ───────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"           # lightweight; change to yolov8s.pt for better accuracy
YOLO_CONFIDENCE = 0.35
YOLO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO IDs)
YOLO_INFERENCE_SIZE = 640
YOLO_PROCESS_FPS = 8                 # frames analysed per second per camera

# ─── Capture Configuration ─────────────────────────────────────
STATIONARY_DURATION = 2.0            # seconds vehicle must be still
CAPTURE_DELAY_SECONDS = 1.0          # delay (seconds) after full detection before taking picture
CENTER_MOVE_THRESHOLD = 15.0         # px – max centre drift per frame
SIZE_CHANGE_THRESHOLD = 0.08         # fractional bbox size change
VIEW_MARGIN_RATIO = 0.03            # min margin from frame edges (fraction)
MIN_DETECTION_RATIO = 0.05          # min bbox area / frame area
MIN_SHARPNESS = 30.0                # Laplacian variance threshold
CAPTURE_COOLDOWN = 5.0              # seconds between captures

# ─── Device ────────────────────────────────────────────────────
DEVICE = "auto"  # "auto" | "cuda" | "cpu"

# ─── Web Server ────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False

# ─── Defaults dict for JSON persistence ────────────────────────
_DEFAULTS = {
    "yolo_model": YOLO_MODEL,
    "yolo_confidence": YOLO_CONFIDENCE,
    "yolo_process_fps": YOLO_PROCESS_FPS,
    "yolo_inference_size": YOLO_INFERENCE_SIZE,
    "stationary_duration": STATIONARY_DURATION,
    "capture_delay_seconds": CAPTURE_DELAY_SECONDS,
    "center_move_threshold": CENTER_MOVE_THRESHOLD,
    "size_change_threshold": SIZE_CHANGE_THRESHOLD,
    "view_margin_ratio": VIEW_MARGIN_RATIO,
    "min_detection_ratio": MIN_DETECTION_RATIO,
    "min_sharpness": MIN_SHARPNESS,
    "capture_cooldown": CAPTURE_COOLDOWN,
    "device": DEVICE,
    "video_dir": VIDEO_DIR,
    "database_dir": DATABASE_DIR,
}


def load_config() -> dict:
    """Load persisted config, falling back to defaults."""
    cfg = dict(_DEFAULTS)
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg.update(saved)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    """Persist configuration to JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def get_device():
    """Determine compute device."""
    import torch
    cfg = load_config()
    dev = cfg.get("device", "auto")
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev
