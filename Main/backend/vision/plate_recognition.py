"""
Plate Recognition — High-precision OCR for vehicle registration plates.
Optimized for Indian registration formats (e.g., KL 01 AB 1111, KL 01 AL 0009),
Bharat Series (e.g., 22 BH 1234 AA), and two-line license plates.
Uses EasyOCR with sensitive character stroke detection, multi-scale cropping,
spatial box sorting, and positional character error correction.
"""
import logging
import re
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ocr_reader = None
_ocr_available = False

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', 'T': '1', '|': '1', '!': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'B': '8',
    'g': '9', 'q': '9'
}

CHAR_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}


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


def is_ocr_available() -> bool:
    return _ocr_available


def is_full_plate(text: str) -> bool:
    """
    Check if a registration string is a full vehicle registration.
    A full plate must have state/series letters and 1-4 digits, with total alphanumeric length >= 7.
    Single 4-digit numbers (e.g. '1111') are NOT full plates!
    """
    if not text:
        return False
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean) < 7:
        return False

    # Check for Indian standard: State (2 letters) + digits (at least 2) + number
    has_state = bool(re.search(r'([A-Z]{2})\s*(\d{1,2})', text.upper())) or clean[:2] in INDIAN_STATES
    has_letters = bool(re.search(r'[A-Z]', clean))
    has_digits = bool(re.search(r'\d', clean))

    # Check if ends with 1 to 4 digits (the unique registration number)
    ends_with_digits = bool(re.search(r'\d{1,4}$', clean))

    return has_letters and has_digits and ends_with_digits and (has_state or len(clean) >= 8)


def _to_digits(s: str) -> str:
    res = []
    for c in s:
        res.append(CHAR_TO_DIGIT.get(c, CHAR_TO_DIGIT.get(c.upper(), c)))
    return "".join(res)


def _to_letters(s: str) -> str:
    res = []
    for c in s:
        res.append(CHAR_TO_LETTER.get(c, CHAR_TO_LETTER.get(c.upper(), c)))
    return "".join(res)


def parse_indian_plate(raw_text: str) -> tuple[str, bool]:
    """
    Robust Indian License Plate Parser with positional OCR error correction.
    Returns: (formatted_plate_string, is_full_plate)
    """
    if not raw_text:
        return "", False

    # Strip noise, keep alphanumeric
    clean = re.sub(r'[^A-Za-z0-9]', '', raw_text.upper().strip())

    # Remove HSRP (High Security Registration Plate) 'IND' watermark if prepended
    if (clean.startswith("IND") or clean.startswith("1ND")) and len(clean) >= 9:
        clean = clean[3:]

    # Positional substitution for known first 2 letters if state code was misread (e.g. K1 -> KL, KI -> KL)
    if len(clean) >= 8:
        prefix2 = clean[:2]
        if prefix2 in ["K1", "KI", "RL", "HL", "IL"]:
            clean = "KL" + clean[2:]
        elif prefix2 in ["M1", "M|"]:
            clean = "MH" + clean[2:]
        elif prefix2 in ["D1"]:
            clean = "DL" + clean[2:]
        elif prefix2 in ["T1"]:
            clean = "TN" + clean[2:]

    # 1. Exact match starting from beginning of clean string
    # Standard format: [State 2L] [Dist 1-2D] [Series 0-3L] [Num 1-4D]
    m_exact = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$', clean)
    if m_exact:
        st, dist, ser, num = m_exact.groups()
        if st in INDIAN_STATES or len(clean) >= 8:
            if len(dist) == 1:
                dist = "0" + dist
            num = num.zfill(4) if len(num) <= 4 else num
            ser_part = f" {ser}" if ser else ""
            return f"{st} {dist}{ser_part} {num}".strip(), True

    # 2. Regex search for standard Indian pattern with VALID Indian state code
    for m in re.finditer(r'([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{0,3})\s*([0-9]{1,4})', clean):
        st, dist, ser, num = m.groups()
        if st in INDIAN_STATES:
            if len(dist) == 1:
                dist = "0" + dist
            num = num.zfill(4) if len(num) <= 4 else num
            ser_part = f" {ser}" if ser else ""
            return f"{st} {dist}{ser_part} {num}".strip(), True

    # 3. Bharat Series format: [2 digits year] BH [4 digits] [1-2 letters] (e.g. 22 BH 1234 AA)
    m_bh = re.search(r'(\d{2})\s*(BH)\s*(\d{4})\s*([A-Z]{1,2})', clean)
    if m_bh:
        yr, bh, num, ser = m_bh.groups()
        return f"{yr} {bh} {num} {ser}", True

    # 4. Clean string positional reconstruction for common OCR misreads (e.g. 'KLOLALooog' -> 'KL 01 AL 0009')
    if 8 <= len(clean) <= 12:
        st_candidate = _to_letters(clean[:2])
        if clean[:2] in ["K1", "KI", "RL", "HL", "IL"] or st_candidate == "KL":
            st_candidate = "KL"

        num_candidate = _to_digits(clean[-4:])
        middle = clean[2:-4]
        m_dist = re.match(r'^([0-9ODQLIT!]{1,2})(.*)$', middle)
        if m_dist:
            dist_raw, ser_raw = m_dist.groups()
            dist_candidate = _to_digits(dist_raw)
            if len(dist_candidate) == 1:
                dist_candidate = "0" + dist_candidate
            ser_candidate = _to_letters(ser_raw)
            ser_part = f" {ser_candidate}" if ser_candidate else ""
            formatted = f"{st_candidate} {dist_candidate}{ser_part} {num_candidate}".strip()
            if st_candidate in INDIAN_STATES:
                return formatted, True

    # 5. Two letters + digits (e.g. DL 1C 1234)
    m3 = re.search(r'([A-Z]{2})\s*(\d{1,2}[A-Z]{1,2})\s*(\d{1,4})', raw_text.upper())
    if m3:
        st, dist_ser, num = m3.groups()
        return f"{st} {dist_ser} {num.zfill(4)}", True

    return "", False


def normalize_plate(text: str) -> str:
    """Normalize plate text — returns formatted full registration or cleaned string."""
    if not text:
        return ""

    parsed, is_full = parse_indian_plate(text)
    if is_full and parsed:
        return parsed

    # Fallback: clean alphanumeric
    clean = re.sub(r'[^A-Za-z0-9 ]', '', text.upper().strip())
    # Try Indian regex on cleaned text
    m = re.search(r'([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{0,3})\s*(\d{1,4})', clean)
    if m:
        st, dist, ser, num = m.groups()
        if len(dist) == 1:
            dist = "0" + dist
        ser_part = f" {ser}" if ser else ""
        return f"{st} {dist}{ser_part} {num.zfill(4)}".strip()

    words = [w for w in clean.split() if len(w) >= 2]
    if len("".join(words)) >= 4:
        return " ".join(words)

    return clean if len(clean) >= 4 else ""


def _sort_and_merge_ocr_boxes(results: list) -> tuple[str, float]:
    """
    Sorts OCR bounding boxes in natural reading order (top-to-bottom, left-to-right)
    and merges lines correctly for two-line and multi-block plates.
    """
    if not results:
        return "", 0.0

    items = []
    for r in results:
        pts = np.array(r[0])
        x_min = float(np.min(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        x_max = float(np.max(pts[:, 0]))
        y_max = float(np.max(pts[:, 1]))
        h = max(1.0, y_max - y_min)
        txt = str(r[1]).strip()
        conf = float(r[2])
        if txt:
            items.append({
                "text": txt,
                "conf": conf,
                "x": x_min,
                "y": y_min,
                "h": h
            })

    if not items:
        return "", 0.0

    avg_h = sum(it["h"] for it in items) / len(items)
    line_threshold = avg_h * 0.65

    # Group into reading lines
    items.sort(key=lambda it: it["y"])
    lines = []
    current_line = [items[0]]

    for it in items[1:]:
        if abs(it["y"] - current_line[0]["y"]) < line_threshold:
            current_line.append(it)
        else:
            current_line.sort(key=lambda x: x["x"])
            lines.append(current_line)
            current_line = [it]

    if current_line:
        current_line.sort(key=lambda x: x["x"])
        lines.append(current_line)

    ordered_texts = []
    confs = []
    for line in lines:
        for it in line:
            ordered_texts.append(it["text"])
            confs.append(it["conf"])

    merged_text = " ".join(ordered_texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return merged_text, avg_conf


def _run_ocr_on_roi(roi: np.ndarray) -> tuple[str, float, str, bool]:
    """
    Run EasyOCR on a candidate crop with dynamic super-resolution / upscaling,
    contrast enhancement, and reading order assembly.
    Returns: (plate_text, confidence, raw_text, is_full)
    """
    if roi is None or roi.size == 0 or _ocr_reader is None:
        return "", 0.0, "", False

    try:
        h, w = roi.shape[:2]
        if h < 18 or w < 30:
            return "", 0.0, "", False

        # Dynamic Upscaling: license plate characters must be at least 45-60px tall for accurate OCR
        crop = roi.copy()
        if h < 80 or w < 280:
            scale_y = max(1.5, 90.0 / h)
            scale_x = max(1.5, 320.0 / w)
            scale = max(scale_x, scale_y)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # EasyOCR with sensitive parameters (detects both small state letters and large digits)
        ocr_kwargs = {
            "detail": 1,
            "paragraph": False,
            "text_threshold": 0.20,
            "low_text": 0.15,
            "min_size": 6,
            "link_threshold": 0.35,
            "mag_ratio": 1.5,
            "allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -/"
        }

        results = _ocr_reader.readtext(enhanced, **ocr_kwargs)
        if not results:
            results = _ocr_reader.readtext(gray, **ocr_kwargs)

        if not results:
            return "", 0.0, "", False

        # Sort and merge lines in spatial reading order
        raw_text, avg_conf = _sort_and_merge_ocr_boxes(results)
        if not raw_text:
            return "", 0.0, "", False

        # Parse plate
        parsed, is_full = parse_indian_plate(raw_text)
        if is_full and parsed:
            return parsed, avg_conf, raw_text, True

        normalized = normalize_plate(raw_text)
        is_full_norm = is_full_plate(normalized)
        return normalized, avg_conf, raw_text, is_full_norm

    except Exception as e:
        logger.debug("OCR error on ROI: %s", e)
        return "", 0.0, "", False


def find_plate_crops(frame: np.ndarray, bbox: list = None) -> list[tuple[np.ndarray, str]]:
    """
    Extract focused license plate candidate crops using edge detection and bumper aspect ratios.
    Returns list of tuples: (crop_image, label)
    """
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    crops = []

    if bbox is not None and len(bbox) == 4:
        bx1, by1, bx2, by2 = bbox
        x1 = max(0, min(w - 1, int(bx1)))
        y1 = max(0, min(h - 1, int(by1)))
        x2 = max(0, min(w, int(bx2)))
        y2 = max(0, min(h, int(by2)))
        bw = x2 - x1
        bh = y2 - y1

        if bw > 50 and bh > 40:
            # Crop 1: Lower Bumper Center (primary location of front & rear license plates)
            # x: center 65% of vehicle, y: lower 45% of vehicle
            cb_x1 = max(0, x1 + int(bw * 0.18))
            cb_x2 = min(w, x1 + int(bw * 0.82))
            cb_y1 = max(0, y1 + int(bh * 0.45))
            cb_y2 = min(h, y2)
            crop_cb = frame[cb_y1:cb_y2, cb_x1:cb_x2]
            if crop_cb.size > 0:
                crops.append((crop_cb, "bumper_center"))

            # Crop 2: Lower 55% of full vehicle bbox
            low_y1 = max(0, y1 + int(bh * 0.45))
            crop_low = frame[low_y1:y2, x1:x2]
            if crop_low.size > 0:
                crops.append((crop_low, "bumper_full_width"))

            # Crop 3: Exact contour detection for rectangular plate inside lower bumper
            if crop_cb.size > 0:
                try:
                    gray_b = cv2.cvtColor(crop_cb, cv2.COLOR_BGR2GRAY)
                    blur_b = cv2.bilateralFilter(gray_b, 9, 75, 75)
                    sobelx = cv2.Sobel(blur_b, cv2.CV_8U, 1, 0, ksize=3)
                    _, thresh_b = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
                    morph_b = cv2.morphologyEx(thresh_b, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(morph_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cx, cy, cw_box, ch_box = cv2.boundingRect(cnt)
                        if ch_box == 0:
                            continue
                        aspect = cw_box / float(ch_box)
                        area = cw_box * ch_box
                        if (1.5 <= aspect <= 6.5) and (cw_box >= 60) and (ch_box >= 18) and (area >= 1200):
                            pad_x = int(cw_box * 0.12)
                            pad_y = int(ch_box * 0.20)
                            px1 = max(0, cx - pad_x)
                            py1 = max(0, cy - pad_y)
                            px2 = min(crop_cb.shape[1], cx + cw_box + pad_x)
                            py2 = min(crop_cb.shape[0], cy + ch_box + pad_y)
                            plate_rect = crop_cb[py1:py2, px1:px2]
                            if plate_rect.size > 0:
                                crops.append((plate_rect, "contour_plate"))
                                break
                except Exception:
                    pass

            # Crop 4: Full vehicle bbox
            crop_full = frame[y1:y2, x1:x2]
            if crop_full.size > 0:
                crops.append((crop_full, "vehicle_full"))

    # Crop 5: Center-lower frame fallback
    fy1 = int(h * 0.35)
    fx1 = int(w * 0.15)
    fx2 = int(w * 0.85)
    crop_frame_center = frame[fy1:h, fx1:fx2]
    if crop_frame_center.size > 0:
        crops.append((crop_frame_center, "frame_center"))

    # Crop 6: Full frame
    crops.append((frame, "full_frame"))
    return crops


def recognize_plate(frame: np.ndarray, bbox: list = None) -> dict:
    """
    Attempt plate recognition on a frame.
    Scans candidate crops and guarantees preference for full registration numbers
    (e.g., KL 01 AB 1111) over partial digits (e.g., 1111).
    Returns: {text, confidence, raw, success, is_full}
    """
    if not _ocr_available or _ocr_reader is None:
        if not _ocr_available:
            init_ocr()
        if not _ocr_available or _ocr_reader is None:
            return {"text": "", "confidence": 0.0, "success": False, "is_full": False, "reason": "OCR not available"}

    if frame is None or frame.size == 0:
        return {"text": "", "confidence": 0.0, "success": False, "is_full": False, "reason": "Empty frame"}

    try:
        candidate_crops = find_plate_crops(frame, bbox)

        full_plate_candidates = []
        partial_candidates = []

        for crop, label in candidate_crops:
            plate_text, conf, raw, is_full = _run_ocr_on_roi(crop)
            if not plate_text:
                continue

            if is_full:
                full_plate_candidates.append({
                    "text": plate_text,
                    "confidence": conf,
                    "raw": raw,
                    "label": label,
                    "score": 1000.0 + conf * 100.0 + len(plate_text)
                })
                # If we found a full plate with good confidence, we can stop early
                if conf >= 0.65:
                    break
            elif len(plate_text) >= 4:
                partial_candidates.append({
                    "text": plate_text,
                    "confidence": conf,
                    "raw": raw,
                    "label": label,
                    "score": conf * 100.0 + len(plate_text)
                })

        # Check if we can assemble a full plate from partial pieces (e.g. state code 'KL 01 AB' + digits '1111')
        if not full_plate_candidates and len(partial_candidates) >= 2:
            combined_raw = " ".join([c["raw"] for c in partial_candidates])
            comb_parsed, comb_is_full = parse_indian_plate(combined_raw)
            if comb_is_full and comb_parsed:
                return {
                    "text": comb_parsed,
                    "confidence": round(sum(c["confidence"] for c in partial_candidates) / len(partial_candidates), 3),
                    "success": True,
                    "is_full": True,
                    "raw": combined_raw
                }

        # 1. Prioritize full plate candidates
        if full_plate_candidates:
            full_plate_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = full_plate_candidates[0]
            return {
                "text": best["text"],
                "confidence": round(best["confidence"], 3),
                "success": True,
                "is_full": True,
                "raw": best["raw"]
            }

        # 2. Fallback to partial candidate only if NO full plate exists
        if partial_candidates:
            partial_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = partial_candidates[0]
            return {
                "text": best["text"],
                "confidence": round(best["confidence"], 3),
                "success": True,
                "is_full": False,
                "raw": best["raw"]
            }

        return {"text": "", "confidence": 0.0, "success": False, "is_full": False, "reason": "No plate found"}

    except Exception as e:
        logger.error("OCR error: %s", e)
        return {"text": "", "confidence": 0.0, "success": False, "is_full": False, "reason": str(e)}
