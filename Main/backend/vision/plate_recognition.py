"""
Plate Recognition — OCR for vehicle registration plates.
Uses EasyOCR with multi-region scanning and adaptive contrast enhancement.
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


def _run_ocr_on_roi(roi: np.ndarray) -> tuple[str, float, str]:
    """Helper to preprocess and run OCR on a candidate crop."""
    if roi is None or roi.size == 0 or _ocr_reader is None:
        return "", 0.0, ""

    try:
        h, w = roi.shape[:2]
        if h < 25 or w < 50:
            return "", 0.0, ""

        # Contrast enhancement with CLAHE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        results = _ocr_reader.readtext(enhanced)
        if not results:
            # Fallback to standard grayscale
            results = _ocr_reader.readtext(gray)

        if not results:
            return "", 0.0, ""

        all_text = " ".join([r[1] for r in results])
        avg_conf = float(sum([r[2] for r in results]) / len(results))
        plate = normalize_plate(all_text)
        return plate, avg_conf, all_text
    except Exception as e:
        logger.debug("OCR error on ROI: %s", e)
        return "", 0.0, ""


def recognize_plate(frame: np.ndarray, bbox: list = None) -> dict:
    """
    Attempt plate recognition on a frame.
    Scans candidate regions:
    1. Lower half of vehicle bbox (standard license plate placement)
    2. Full vehicle bbox
    3. Full frame fallback
    Returns: {text, confidence, raw, success}
    """
    if not _ocr_available or _ocr_reader is None:
        if not _ocr_available:
            init_ocr()
        if not _ocr_available or _ocr_reader is None:
            return {"text": "", "confidence": 0.0, "success": False, "reason": "OCR not available"}

    if frame is None or frame.size == 0:
        return {"text": "", "confidence": 0.0, "success": False, "reason": "Empty frame"}

    try:
        candidates = []

        # Candidate 1: Lower half of vehicle bbox
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 > x1 and y2 > y1:
                # Lower 55% of the vehicle bbox
                plate_y1 = y1 + int((y2 - y1) * 0.45)
                roi_lower = frame[plate_y1:y2, x1:x2]
                plate1, conf1, raw1 = _run_ocr_on_roi(roi_lower)
                if plate1 and conf1 > 0.4:
                    return {"text": plate1, "confidence": round(conf1, 3), "success": True, "raw": raw1}
                if plate1:
                    candidates.append((conf1, plate1, raw1))

                # Candidate 2: Full vehicle bbox
                roi_full = frame[y1:y2, x1:x2]
                plate2, conf2, raw2 = _run_ocr_on_roi(roi_full)
                if plate2 and conf2 > 0.4:
                    return {"text": plate2, "confidence": round(conf2, 3), "success": True, "raw": raw2}
                if plate2:
                    candidates.append((conf2, plate2, raw2))

        # Candidate 3: Full frame
        plate3, conf3, raw3 = _run_ocr_on_roi(frame)
        if plate3:
            candidates.append((conf3, plate3, raw3))

        if candidates:
            # Pick highest confidence result
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_conf, best_plate, best_raw = candidates[0]
            return {
                "text": best_plate,
                "confidence": round(best_conf, 3),
                "success": bool(best_plate and len(best_plate) >= 4),
                "raw": best_raw
            }

        return {"text": "", "confidence": 0.0, "success": False, "reason": "No plate found"}

    except Exception as e:
        logger.error("OCR error: %s", e)
        return {"text": "", "confidence": 0.0, "success": False, "reason": str(e)}


def normalize_plate(text: str) -> str:
    """Normalize plate text — remove noise, format registration."""
    if not text:
        return ""
    clean = re.sub(r'[^A-Za-z0-9 ]', '', text.upper().strip())

    # 1. Standard Indian Registration Format (e.g. KL 01 AL 0009, MH 12 DE 1432)
    m1 = re.search(r'([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{1,4})', clean)
    if m1:
        return f"{m1.group(1)} {m1.group(2)} {m1.group(3)} {m1.group(4)}"

    # 2. Bharat Series Format (e.g. 22 BH 1234 AA)
    m2 = re.search(r'(\d{2})\s*(BH)\s*(\d{4})\s*([A-Z]{1,2})', clean)
    if m2:
        return f"{m2.group(1)} {m2.group(2)} {m2.group(3)} {m2.group(4)}"

    # 3. Two letters + digits (e.g. DL 1C 1234)
    m3 = re.search(r'([A-Z]{2})\s*(\d{1,2}[A-Z]{1,2})\s*(\d{1,4})', clean)
    if m3:
        return f"{m3.group(1)} {m3.group(2)} {m3.group(3)}"

    # 4. General Alphanumeric Plate: keep words with total length >= 4
    words = [w for w in clean.split() if len(w) >= 2]
    if len("".join(words)) >= 4:
        return " ".join(words)

    return clean if len(clean) >= 4 else ""


def is_ocr_available() -> bool:
    return _ocr_available
