"""
Plate Recognition — OCR for vehicle registration plates.
Uses EasyOCR when available, otherwise allows manual entry.
"""
import logging
import re
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ocr_reader = None
_ocr_available = False


def init_ocr():
    """Initialize EasyOCR reader. Non-fatal if unavailable."""
    global _ocr_reader, _ocr_available
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        _ocr_available = True
        logger.info("EasyOCR initialized (GPU: %s)", use_gpu)
    except ImportError:
        logger.warning("EasyOCR not installed — plate recognition will require manual entry")
        _ocr_available = False
    except Exception as e:
        logger.warning("EasyOCR init failed: %s — manual entry required", e)
        _ocr_available = False


def recognize_plate(frame: np.ndarray, bbox: list = None) -> dict:
    """
    Attempt plate recognition on a frame.
    Returns: {text, confidence, raw_results, success}
    """
    if not _ocr_available or _ocr_reader is None:
        return {"text": "", "confidence": 0.0, "success": False, "reason": "OCR not available"}

    try:
        # Crop region of interest if bbox provided (lower half of vehicle for plate)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # License plates are typically in the lower third of the vehicle
            plate_y1 = y1 + int((y2 - y1) * 0.5)
            plate_y2 = y2
            roi = frame[plate_y1:plate_y2, x1:x2]
        else:
            roi = frame

        if roi.size == 0:
            return {"text": "", "confidence": 0.0, "success": False, "reason": "Empty ROI"}

        # Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = _ocr_reader.readtext(gray)

        if not results:
            return {"text": "", "confidence": 0.0, "success": False, "reason": "No text found"}

        # Combine results and find plate-like patterns
        all_text = " ".join([r[1] for r in results])
        avg_conf = float(sum([r[2] for r in results]) / len(results))

        # Try to find plate pattern (Indian format: XX 00 XX 0000)
        plate_text = normalize_plate(all_text)

        return {
            "text": plate_text if plate_text else all_text.upper(),
            "confidence": round(avg_conf, 3),
            "success": bool(plate_text),
            "raw": all_text,
        }
    except Exception as e:
        logger.error("OCR error: %s", e)
        return {"text": "", "confidence": 0.0, "success": False, "reason": str(e)}


def normalize_plate(text: str) -> str:
    """Normalize plate text — remove noise, uppercase."""
    if not text:
        return ""
    # Remove non-alphanumeric except spaces
    clean = re.sub(r'[^A-Za-z0-9 ]', '', text.upper().strip())
    # Try to match Indian plate pattern
    match = re.search(r'([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{1,4})', clean)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
    return clean


def is_ocr_available() -> bool:
    return _ocr_available
