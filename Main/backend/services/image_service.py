"""
Image Service — Serves images from the database directory.
"""
import os
import logging

logger = logging.getLogger(__name__)

ALLOWED_CAMERAS = {"front", "back", "left", "right"}


def get_image_path(database_dir: str, session_id: str, camera: str,
                   db_manager) -> str | None:
    """Safely resolve an image path for a session and camera."""
    if camera not in ALLOWED_CAMERAS:
        return None

    session = db_manager.get_session(session_id)
    if not session:
        return None

    folder = session.get("folder_path", "")
    if not folder or not os.path.isdir(folder):
        return None

    # Ensure folder is under database_dir (path traversal protection)
    abs_folder = os.path.abspath(folder)
    abs_db_dir = os.path.abspath(database_dir)
    if not abs_folder.startswith(abs_db_dir):
        logger.warning("Path traversal attempt: %s", folder)
        return None

    img_path = os.path.join(folder, f"{camera}.jpg")
    if os.path.isfile(img_path):
        return img_path
    return None


def get_image_bytes(database_dir: str, session_id: str, camera: str, db_manager) -> bytes | None:
    """
    Get image bytes: first checks disk file; if missing or on another computer,
    retrieves directly from the SQL database BLOB storage.
    """
    if camera not in ALLOWED_CAMERAS:
        return None

    # 1. Try file path on disk
    path = get_image_path(database_dir, session_id, camera, db_manager)
    if path and os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            pass

    # 2. Fallback to SQL database BLOB storage
    if db_manager and hasattr(db_manager, "get_session_image_blob"):
        blob = db_manager.get_session_image_blob(session_id, camera)
        if blob:
            return blob

    return None


def get_thumbnail(image_path: str = None, image_bytes: bytes = None, max_width: int = 300) -> bytes | None:
    """Generate a thumbnail as JPEG bytes from either path or raw bytes."""
    import cv2
    import numpy as np

    img = None
    if image_bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_path and os.path.isfile(image_path):
        img = cv2.imread(image_path)

    if img is None:
        return None

    try:
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
    except Exception as e:
        logger.error("Thumbnail error: %s", e)
        return None
