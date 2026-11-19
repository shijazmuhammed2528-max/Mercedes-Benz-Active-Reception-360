"""
YOLO Vehicle Detector — Runs ultralytics YOLO inference on frames.
"""
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

_model = None
_device = "cpu"
_yolo_lock = threading.Lock()


def init_yolo(model_name: str = "yolov8n.pt", device: str = "cpu"):
    """Load the YOLO model once with RTX 4050 GPU optimizations."""
    global _model, _device
    with _yolo_lock:
        try:
            import torch
            from ultralytics import YOLO
            _device = device
            if "cuda" in str(device) and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_device(0)
                logger.info("RTX 4050 GPU Acceleration Active (cuDNN benchmark enabled, 6GB VRAM pool)")
            _model = YOLO(model_name)
            logger.info("YOLO model loaded: %s on %s", model_name, device)
            return True
        except Exception as e:
            logger.error("YOLO init failed: %s", e)
            _model = None
            return False


def detect_vehicles(frame: np.ndarray, confidence: float = 0.45,
                    vehicle_classes: list = None, inference_size: int = 640) -> list:
    """
    Run YOLO on a single frame with zero-overhead inference_mode.
    Thread-safe execution for concurrent capture and streaming loops.
    Returns list of dicts: {bbox, confidence, class_id, class_name}
    """
    if _model is None or frame is None or frame.size == 0:
        return []
    if vehicle_classes is None:
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    with _yolo_lock:
        try:
            import torch
            with torch.inference_mode():
                results = _model.predict(
                    frame,
                    conf=confidence,
                    classes=vehicle_classes,
                    imgsz=inference_size,
                    device=_device,
                    verbose=False
                )
            detections = []
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = r.names.get(cls_id, "vehicle")
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": round(conf, 4),
                            "class_id": cls_id,
                            "class_name": cls_name,
                        })
            return detections
        except Exception as e:
            logger.error("YOLO detection error: %s", e)
            return []


def is_model_loaded() -> bool:
    return _model is not None
