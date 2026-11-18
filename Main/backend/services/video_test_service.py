"""
Video Test Service — Automated diagnostics for prerecorded reception camera videos.
Tests file integrity, OpenCV video decoding, FPS, resolution, frame accessibility,
and YOLO vehicle detection confidence.
"""
import os
import cv2
import base64
import logging
from backend.vision.yolo_detector import detect_vehicles, is_model_loaded, init_yolo

logger = logging.getLogger(__name__)


def ensure_yolo_loaded():
    """Ensure YOLO model is initialized for video testing."""
    if not is_model_loaded():
        try:
            from config import get_device, load_config
            cfg = load_config()
            model_name = cfg.get("yolo_model", "yolov8n.pt")
            device = get_device()
            init_yolo(model_name, device)
        except Exception as e:
            logger.warning("Failed to auto-init YOLO for video test: %s", e)


def test_single_video(camera_name: str, video_path: str, test_yolo: bool = True) -> dict:
    """Run full diagnostic test on one video file."""
    result = {
        "camera": camera_name,
        "path": video_path,
        "exists": False,
        "size_mb": 0.0,
        "readable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "yolo_tested": False,
        "vehicle_detected": False,
        "detection_confidence": 0.0,
        "detection_class": "",
        "preview_jpeg_b64": "",
        "status": "FAILED",
        "message": ""
    }

    if not os.path.isfile(video_path):
        result["message"] = f"File not found: {video_path}"
        return result

    result["exists"] = True
    result["size_mb"] = round(os.path.getsize(video_path) / (1024 * 1024), 2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["message"] = "OpenCV failed to open video file (unsupported codec or corrupted file)"
        return result

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(total_frames / max(1.0, fps), 2) if fps > 0 else 0.0

        result["width"] = width
        result["height"] = height
        result["fps"] = round(fps, 1)
        result["frame_count"] = total_frames
        result["duration_seconds"] = duration

        # Test reading frame at ~45% position where vehicle is centered in bay
        test_frame_idx = max(0, min(total_frames - 1, int(total_frames * 0.45)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            # Fallback to frame 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if not ret or frame is None:
            result["message"] = "Failed to decode video frames"
            return result

        result["readable"] = True

        # YOLO detection test
        if test_yolo:
            ensure_yolo_loaded()
            result["yolo_tested"] = True
            try:
                detections = detect_vehicles(frame, confidence=0.35)
                if detections:
                    best = max(detections, key=lambda d: d["confidence"])
                    result["vehicle_detected"] = True
                    result["detection_confidence"] = round(best["confidence"] * 100, 1)
                    result["detection_class"] = best.get("class_name", "vehicle")

                    # Draw green bounding box on preview
                    x1, y1, x2, y2 = best["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 102), 3)
                    label = f"{best.get('class_name', 'CAR').upper()} {result['detection_confidence']}%"
                    cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 102), 2)
            except Exception as e:
                logger.warning("YOLO test failed on %s: %s", camera_name, e)

        # Generate lightweight base64 preview thumbnail
        thumb_h = 240
        thumb_w = int(thumb_h * (width / max(1, height)))
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            result["preview_jpeg_b64"] = base64.b64encode(buf).decode("utf-8")

        result["status"] = "PASSED"
        result["message"] = f"OK ({width}x{height} @ {fps:.0f}fps, {duration}s)"
        if result["vehicle_detected"]:
            result["message"] += f" — {result['detection_class'].upper()} detected ({result['detection_confidence']}%)"

    except Exception as e:
        result["message"] = f"Video test exception: {str(e)}"
    finally:
        cap.release()

    return result


def test_all_videos(video_sources: dict, test_yolo: bool = True) -> dict:
    """Run full diagnostic test on all 4 camera videos."""
    report = {}
    all_passed = True

    for cam_name, path in video_sources.items():
        res = test_single_video(cam_name, path, test_yolo=test_yolo)
        report[cam_name] = res
        if res["status"] != "PASSED":
            all_passed = False

    return {
        "all_passed": all_passed,
        "cameras": report
    }
