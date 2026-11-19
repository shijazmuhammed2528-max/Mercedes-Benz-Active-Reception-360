"""
Plate Recognition — High-precision OCR for vehicle registration plates.
Optimized for Indian registration formats (e.g., KL 52 V 7552, KL 01 AB 1111, KL 01 AL 0009),
Bharat Series (e.g., 22 BH 1234 AA), and two-line license plates.
Uses EasyOCR with sensitive character stroke detection, morphological Blackhat isolation,
multi-scale super-resolution, spatial box sorting, and positional character error correction.
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
    'O': '0', 'D': '0', 'Q': '0', 'o': '0',
    'I': '1', 'L': '1', 'l': '1', 'T': '1', '|': '1', '!': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'E': '3',
    'A': '4',
    'S': '5', 's': '5',
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

STATE_CORRECTIONS = {
    "LL": "KL", "0L": "KL", "OL": "KL", "CL": "KL", "EL": "KL",
    "K1": "KL", "KI": "KL", "RL": "KL", "HL": "KL", "IL": "KL", "1L": "KL",
    "4L": "KL", "@L": "KL", "L": "KL", "K": "KL",
    "4": "KL", "@": "KL", "0": "KL", "U": "KL", "D": "KL",
    "M1": "MH", "M|": "MH", "D1": "DL", "T1": "TN",
}


def init_ocr():
    """Initialize EasyOCR reader with FP32 unquantized weights for maximum precision."""
    global _ocr_reader, _ocr_available
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        # quantize=False preserves full 32-bit floating point model weights
        _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, quantize=False, verbose=False)
        _ocr_available = True
        logger.info("EasyOCR initialized (GPU: %s, FP32 unquantized)", use_gpu)
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
    A full plate must have state/series letters and digits, with total alphanumeric length >= 7.
    Single 4-digit numbers (e.g. '1111' or '7552') are NOT full plates!
    """
    if not text:
        return False
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean) < 7:
        return False

    has_state = bool(re.search(r'([A-Z]{2})\s*(\d{1,2})', text.upper())) or clean[:2] in INDIAN_STATES
    has_letters = bool(re.search(r'[A-Z]', clean))
    has_digits = bool(re.search(r'\d', clean))
    ends_with_digits = bool(re.search(r'\d{1,4}$', clean))

    return has_letters and has_digits and ends_with_digits and (has_state or len(clean) >= 8)


def clean_ocr_tokens(raw: str) -> str:
    """Clean brackets, quotes, braces and OCR noise symbols."""
    s = re.sub(r'[{}\[\]\(\)"\'=~_\-,|:;`]', ' ', raw).strip()
    s = re.sub(r'^(IND|1ND)\s*', '', s, flags=re.I).strip()
    return s


def _to_digits(s: str) -> str:
    return "".join([CHAR_TO_DIGIT.get(c, c) for c in s])


def _to_letters(s: str) -> str:
    return "".join([CHAR_TO_LETTER.get(c, c.upper()) for c in s])


def parse_indian_plate(raw_text: str) -> tuple[str, bool]:
    """
    Robust Indian License Plate Parser with positional error correction.
    Handles HSRP emblem distortion, two-line plates, and character confusions.
    Returns: (formatted_plate_string, is_full_plate)
    """
    if not raw_text:
        return "", False

    cleaned = clean_ocr_tokens(raw_text)

    # 1. Exact match for standard Indian registration: [State 2L] [Dist 1-2D] [Series 0-3L] [Num 1-4D]
    m1 = re.search(r'([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{0,3})\s*([0-9]{1,4})', cleaned.upper())
    if m1:
        st, dist, ser, num = m1.groups()
        if st in INDIAN_STATES:
            if len(dist) == 1: dist = "0" + dist
            num = num.zfill(4) if len(num) <= 4 else num
            ser_part = f" {ser}" if ser else ""
            return f"{st} {dist}{ser_part} {num}".strip(), True

    # 2. Pattern with flexible spacing & prefix/suffix noise (e.g. '3 0 52 v7552) 70o 3', 'L 52 v 7552')
    # Look for [pfx] [dist 2 digits] [series 1-2 letters] [number 3-4 digits]
    m2 = re.search(r'(?:^|\s)([A-Za-z0-9@]{1,3})\s*([0-9]{2})\s*([A-Za-z]{1,2})\s*([0-9]{3,4})(?:$|\s)', cleaned, flags=re.I)
    if not m2:
        m2 = re.search(r'(?:^|\s)([A-Za-z0-9@]{1,3})\s*([0-9]{2})\s*([A-Za-z]{1,2})([0-9]{3,4})(?:$|\s)', cleaned, flags=re.I)
    if not m2:
        m2 = re.search(r'(?:^|\s)([A-Za-z0-9@]{1,3})([0-9]{2})\s*([A-Za-z]{1,2})\s*([0-9]{3,4})(?:$|\s)', cleaned, flags=re.I)
    if not m2:
        m2 = re.search(r'(?:^|\s)([A-Za-z0-9@]{1,3})([0-9]{2})\s*([A-Za-z]{1,2})([0-9]{3,4})(?:$|\s)', cleaned, flags=re.I)

    if m2:
        pfx, dist, ser, num = m2.groups()
        pfx_up = pfx.upper()
        st = STATE_CORRECTIONS.get(pfx_up, pfx_up)
        if st in ["4", "@", "0", "U", "D", "L", "K", "CL", "EL", "0L", "LL"] or len(st) == 1:
            st = "KL"
        if (st in INDIAN_STATES or len(st) == 2) and dist.isdigit() and num.isdigit():
            num_fmt = num.zfill(4) if len(num) <= 4 else num
            return f"{st} {dist.zfill(2)} {ser.upper()} {num_fmt}", True

    # 3. Compact string positional matching (e.g. 'KLOALooog' -> 'KL 01 AL 0009', 'KL52V7552')
    clean_alpha = re.sub(r'[^A-Za-z0-9]', '', cleaned)
    for st_cand in ["KL", "MH", "DL", "TN", "KA", "AP", "TS", "UP", "HR", "GJ", "RJ", "WB"]:
        idx = clean_alpha.upper().find(st_cand)
        if idx != -1:
            sub = clean_alpha[idx:]
            if 7 <= len(sub) <= 12:
                st = st_cand
                num_raw = sub[-4:]
                num = _to_digits(num_raw)
                middle = sub[len(st_cand):-4]
                m_mid = re.match(r'^([0-9ODQLITsS]{1,2})(.*)$', middle)
                if m_mid:
                    dist_raw, ser_raw = m_mid.groups()
                    dist = _to_digits(dist_raw)
                    if len(dist) == 1: dist = "0" + dist
                    if dist == "00": dist = "01"
                    ser = _to_letters(ser_raw)
                    ser_part = f" {ser}" if ser else ""
                    return f"{st} {dist}{ser_part} {num}", True

    # 4. Bharat Series format: [2 digits year] BH [4 digits] [1-2 letters] (e.g. 22 BH 1234 AA)
    m_bh = re.search(r'(\d{2})\s*(BH)\s*(\d{4})\s*([A-Z]{1,2})', cleaned.upper())
    if m_bh:
        yr, bh, num, ser = m_bh.groups()
        return f"{yr} {bh} {num} {ser}", True

    # 5. Fallback token scan
    tokens = cleaned.split()
    if tokens:
        num_str = ""
        ser_str = ""
        dist_str = ""
        state_str = ""

        for idx in range(len(tokens) - 1, -1, -1):
            tok = tokens[idx]
            tok_clean = re.sub(r'[^A-Za-z0-9]', '', tok)
            if not tok_clean:
                continue

            if not num_str:
                tok_digits = "".join([CHAR_TO_DIGIT.get(c, c) for c in tok_clean if c in CHAR_TO_DIGIT or c.isdigit()])
                if 3 <= len(tok_digits) <= 4:
                    num_str = tok_digits.zfill(4)
                    num_prefix = tok_clean[:len(tok_clean) - len(tok_digits)]
                    letters = re.findall(r'[A-Za-z]', num_prefix)
                    if letters and not ser_str:
                        ser_str = "".join(letters).upper()
                    continue

            if not ser_str and re.match(r'^[A-Za-z]{1,2}$', tok_clean):
                ser_str = tok_clean.upper()
                continue

            if not dist_str:
                m_dist = re.search(r'([0-9]{1,2})$', tok_clean)
                if m_dist:
                    dist_str = m_dist.group(1).zfill(2)
                    pfx = tok_clean[:m_dist.start()].upper()
                    if pfx:
                        state_str = STATE_CORRECTIONS.get(pfx, pfx)
                    continue

            if not state_str:
                pfx = tok_clean.upper()
                state_str = STATE_CORRECTIONS.get(pfx, pfx)

        if not state_str or state_str not in INDIAN_STATES:
            if state_str in STATE_CORRECTIONS or state_str in ["L", "K", "4", "@", "0", "U", "D", "CL", "EL", ""]:
                state_str = "KL"

        if num_str and dist_str:
            ser_part = f" {ser_str}" if ser_str else ""
            return f"{state_str} {dist_str}{ser_part} {num_str}", True

    return "", False


def normalize_plate(text: str) -> str:
    """Normalize plate text — returns formatted full registration or cleaned string."""
    if not text:
        return ""

    parsed, is_full = parse_indian_plate(text)
    if is_full and parsed:
        return parsed

    clean = clean_ocr_tokens(text).upper()
    m = re.search(r'([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{0,3})\s*([0-9]{1,4})', clean)
    if m:
        st, dist, ser, num = m.groups()
        if len(dist) == 1: dist = "0" + dist
        ser_part = f" {ser}" if ser else ""
        return f"{st} {dist}{ser_part} {num.zfill(4)}".strip()

    words = [w for w in clean.split() if len(w) >= 2]
    if len("".join(words)) >= 4:
        return " ".join(words)

    return clean if len(clean) >= 4 else ""


def _sort_and_merge_ocr_boxes(results: list) -> tuple[str, float]:
    """Sorts OCR bounding boxes in reading order and merges lines."""
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
    Run multi-pass EasyOCR on a candidate crop using:
    1. Morphological Blackhat transform (isolates characters on bright plates)
    2. CLAHE contrast enhancement
    3. Unsharp masking
    Returns: (plate_text, confidence, raw_text, is_full)
    """
    if roi is None or roi.size == 0 or _ocr_reader is None:
        return "", 0.0, "", False

    try:
        h, w = roi.shape[:2]
        if h < 16 or w < 25:
            return "", 0.0, "", False

        # Super-resolution / upscaling
        scale = max(2.0, 90.0 / max(1, h))
        target_w = int(w * scale)
        target_h = int(h * scale)
        crop = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Prepare multi-pass enhanced versions
        versions = []

        # Pass 1: Morphological Blackhat (isolates dark character ink from reflective plate surface)
        k_size = max(9, int(target_h * 0.25) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        inv_bh = cv2.bitwise_not(blackhat)
        versions.append(("blackhat", inv_bh))

        # Pass 2: High-contrast CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        versions.append(("clahe", enhanced))

        # Pass 3: Unsharp mask (sharpens fuzzy character edges)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 1.5)
        unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
        versions.append(("unsharp", unsharp))

        ocr_kwargs = {
            "detail": 1,
            "paragraph": False,
            "text_threshold": 0.12,
            "low_text": 0.08,
            "min_size": 4,
            "link_threshold": 0.25,
            "mag_ratio": 1.2,
        }

        best_result = ("", 0.0, "", False)

        for vname, vimg in versions:
            results = _ocr_reader.readtext(vimg, **ocr_kwargs)
            if not results:
                continue

            raw_text, avg_conf = _sort_and_merge_ocr_boxes(results)
            if not raw_text:
                continue

            parsed, is_full = parse_indian_plate(raw_text)
            if is_full and parsed:
                return parsed, avg_conf, raw_text, True

            normalized = normalize_plate(raw_text)
            is_full_norm = is_full_plate(normalized)
            if is_full_norm:
                return normalized, avg_conf, raw_text, True

            if not best_result[0] or avg_conf > best_result[1]:
                best_result = (normalized, avg_conf, raw_text, is_full_norm)

        return best_result

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

        if bw > 40 and bh > 30:
            # Crop 1: Lower Bumper Center (primary location of front & rear license plates)
            cb_x1 = max(0, x1 + int(bw * 0.15))
            cb_x2 = min(w, x1 + int(bw * 0.85))
            cb_y1 = max(0, y1 + int(bh * 0.35))
            cb_y2 = min(h, y2)
            crop_cb = frame[cb_y1:cb_y2, cb_x1:cb_x2]
            if crop_cb.size > 0:
                crops.append((crop_cb, "bumper_center"))

            # Crop 2: Lower 55% of full vehicle bbox
            low_y1 = max(0, y1 + int(bh * 0.40))
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
                        if (1.5 <= aspect <= 7.0) and (cw_box >= 50) and (ch_box >= 14) and (area >= 800):
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
    fy1 = int(h * 0.30)
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
    (e.g., KL 52 V 7552, KL 01 AB 1111) over partial digits.
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
                if conf >= 0.60:
                    break
            elif len(plate_text) >= 4:
                partial_candidates.append({
                    "text": plate_text,
                    "confidence": conf,
                    "raw": raw,
                    "label": label,
                    "score": conf * 100.0 + len(plate_text)
                })

        # Check if we can assemble a full plate from partial pieces
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
