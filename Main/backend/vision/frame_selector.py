"""
Frame Selector — Selects the best (sharpest) frame from a buffer for capture.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_sharpness(frame: np.ndarray) -> float:
    """Compute Laplacian variance as sharpness measure."""
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def select_best_frame(frames: list, min_sharpness: float = 30.0) -> tuple:
    """
    From a list of frames, select the sharpest one.
    Returns (best_frame, sharpness_score, index).
    """
    if not frames:
        return None, 0.0, -1

    best_frame = None
    best_score = -1.0
    best_idx = -1

    for i, frame in enumerate(frames):
        if frame is None:
            continue
        score = compute_sharpness(frame)
        if score > best_score:
            best_score = score
            best_frame = frame
            best_idx = i

    if best_score < min_sharpness:
        logger.warning("Best frame sharpness %.1f below threshold %.1f", best_score, min_sharpness)

    return best_frame, best_score, best_idx
