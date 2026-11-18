"""
Mercedes-Benz Active Reception — Main Flask Application
"""
import os
import sys
import cv2
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response, send_file
from flask_cors import CORS

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import config
from config import (
    DATABASE_DIR, DATABASE_FILE, VIDEO_SOURCES, CAMERA_LABELS,
    HOST, PORT, DEBUG, load_config, save_config
)

# Ensure directories
os.makedirs(DATABASE_DIR, exist_ok=True)

# Initialize database
from backend.database.db_manager import DatabaseManager
db = DatabaseManager(DATABASE_FILE)
db.recover_from_filesystem(DATABASE_DIR)

# Initialize capture controller
from backend.capture.capture_controller import CaptureController
app_config = load_config()
controller = CaptureController(app_config, db)

# Flask app
app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/capture")
def capture_page():
    return render_template("live_capture.html")

@app.route("/database")
def database_page():
    return render_template("database.html")

@app.route("/vehicle/<session_id>")
def vehicle_detail_page(session_id):
    return render_template("vehicle_details.html", session_id=session_id)

@app.route("/settings")
def settings_page():
    return render_template("settings.html")

# ═══════════════════════════════════════════════════════════════
# API — Dashboard
# ═══════════════════════════════════════════════════════════════

@app.route("/api/dashboard/stats")
def api_dashboard_stats():
    stats = db.get_stats()
    return jsonify(stats)

@app.route("/api/dashboard/recent")
def api_dashboard_recent():
    limit = request.args.get("limit", 10, type=int)
    sessions = db.get_recent(limit)
    return jsonify(sessions)

@app.route("/api/status")
def api_status():
    from backend.services.storage_service import check_video_files, get_storage_info
    return jsonify({
        "videos": check_video_files(VIDEO_SOURCES),
        "storage": get_storage_info(DATABASE_DIR),
        "yolo_ready": controller._yolo_ready,
        "database": os.path.isfile(DATABASE_FILE),
    })

# ═══════════════════════════════════════════════════════════════
# API — Capture Control
# ═══════════════════════════════════════════════════════════════

@app.route("/api/capture/start", methods=["POST"])
def api_capture_start():
    result = controller.start()
    return jsonify(result)

@app.route("/api/capture/pause", methods=["POST"])
def api_capture_pause():
    result = controller.pause()
    return jsonify(result)

@app.route("/api/capture/resume", methods=["POST"])
def api_capture_resume():
    result = controller.resume()
    return jsonify(result)

@app.route("/api/capture/stop", methods=["POST"])
def api_capture_stop():
    result = controller.stop()
    return jsonify(result)

@app.route("/api/capture/reset", methods=["POST"])
def api_capture_reset():
    result = controller.reset()
    return jsonify(result)

@app.route("/api/capture/status")
def api_capture_status():
    return jsonify(controller.get_status())

@app.route("/api/capture/set-plate", methods=["POST"])
def api_set_plate():
    data = request.get_json() or {}
    plate = data.get("plate", "")
    if not plate:
        return jsonify({"success": False, "error": "No plate text provided"}), 400
    result = controller.set_plate(plate)
    return jsonify(result)

# ═══════════════════════════════════════════════════════════════
# API — Video Streaming (MJPEG)
# ═══════════════════════════════════════════════════════════════

def generate_mjpeg(camera_name: str):
    """Generate MJPEG stream for a camera."""
    while True:
        frame = controller.get_display_frame(camera_name)
        if frame is not None:
            # Resize for streaming (reduce bandwidth)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Send a blank frame placeholder
            blank = create_blank_frame(camera_name)
            ret, buffer = cv2.imencode('.jpg', blank)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        import time
        time.sleep(1.0 / 15)  # ~15fps stream


def create_blank_frame(camera_name: str):
    """Create a blank placeholder frame."""
    import numpy as np
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    label = CAMERA_LABELS.get(camera_name, camera_name.upper())
    cv2.putText(frame, label, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    cv2.putText(frame, "NO SIGNAL", (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    return frame


@app.route("/api/capture/stream/<camera>")
def api_camera_stream(camera):
    if camera not in VIDEO_SOURCES:
        return "Invalid camera", 404
    return Response(
        generate_mjpeg(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ═══════════════════════════════════════════════════════════════
# API — Database Search
# ═══════════════════════════════════════════════════════════════

@app.route("/api/database/search", methods=["POST"])
def api_database_search():
    data = request.get_json() or {}
    vehicle_number = data.get("vehicle_number", "").strip()
    date = data.get("date", "").strip()
    time_from = data.get("time_from", "").strip()
    time_to = data.get("time_to", "").strip()

    results = db.search_sessions(
        vehicle_number=vehicle_number or None,
        date=date or None,
        time_from=time_from or None,
        time_to=time_to or None,
    )
    return jsonify(results)

# ═══════════════════════════════════════════════════════════════
# API — Vehicle Session
# ═══════════════════════════════════════════════════════════════

@app.route("/api/vehicle/session/<session_id>")
def api_vehicle_session(session_id):
    session = db.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session)

@app.route("/api/vehicle/image/<session_id>/<camera>")
def api_vehicle_image(session_id, camera):
    if camera not in {"front", "back", "left", "right"}:
        return "Invalid camera", 400
    from backend.services.image_service import get_image_path
    path = get_image_path(DATABASE_DIR, session_id, camera, db)
    if not path:
        return "Image not found", 404
    return send_file(path, mimetype='image/jpeg')

@app.route("/api/vehicle/thumbnail/<session_id>/<camera>")
def api_vehicle_thumbnail(session_id, camera):
    if camera not in {"front", "back", "left", "right"}:
        return "Invalid camera", 400
    from backend.services.image_service import get_image_path, get_thumbnail
    path = get_image_path(DATABASE_DIR, session_id, camera, db)
    if not path:
        return "Image not found", 404
    thumb = get_thumbnail(path, max_width=300)
    if not thumb:
        return send_file(path, mimetype='image/jpeg')
    return Response(thumb, mimetype='image/jpeg')

@app.route("/api/vehicle/download/<session_id>/<camera>")
def api_vehicle_download(session_id, camera):
    if camera not in {"front", "back", "left", "right"}:
        return "Invalid camera", 400
    from backend.services.image_service import get_image_path
    path = get_image_path(DATABASE_DIR, session_id, camera, db)
    if not path:
        return "Image not found", 404
    return send_file(path, as_attachment=True, download_name=f"{camera}.jpg")

@app.route("/api/vehicle/download-zip/<session_id>")
def api_vehicle_download_zip(session_id):
    session = db.get_session(session_id)
    if not session:
        return "Session not found", 404

    folder = session.get("folder_path", "")
    if not folder or not os.path.isdir(folder):
        return "Session folder not found", 404

    # Security: ensure folder is under DATABASE_DIR
    if not os.path.abspath(folder).startswith(os.path.abspath(DATABASE_DIR)):
        return "Invalid path", 403

    from backend.services.zip_service import create_session_zip
    vn = session.get("vehicle_number", "VEHICLE")
    zip_buffer = create_session_zip(folder, vn)
    if not zip_buffer:
        return "ZIP creation failed", 500

    dt = session.get("capture_date", "")
    t = session.get("capture_time_12h", "").replace(":", "-").replace(" ", "_")
    filename = f"{vn.replace(' ', '_')}_{dt}_{t}.zip"

    return send_file(
        zip_buffer, mimetype='application/zip',
        as_attachment=True, download_name=filename
    )

# ═══════════════════════════════════════════════════════════════
# API — Settings
# ═══════════════════════════════════════════════════════════════

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    cfg = load_config()
    from backend.services.storage_service import check_video_files, get_storage_info
    cfg["video_status"] = check_video_files(VIDEO_SOURCES)
    cfg["storage_info"] = get_storage_info(DATABASE_DIR)
    # GPU check
    try:
        import torch
        cfg["cuda_available"] = torch.cuda.is_available()
        cfg["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    except Exception:
        cfg["cuda_available"] = False
        cfg["gpu_name"] = "None"
    return jsonify(cfg)

@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    data = request.get_json() or {}
    cfg = load_config()
    # Only update known keys
    allowed = [
        "yolo_model", "yolo_confidence", "yolo_process_fps", "yolo_inference_size",
        "stationary_duration", "capture_delay_seconds", "center_move_threshold", "size_change_threshold",
        "view_margin_ratio", "min_detection_ratio", "min_sharpness",
        "capture_cooldown", "device",
    ]
    for key in allowed:
        if key in data:
            cfg[key] = data[key]
    save_config(cfg)
    return jsonify({"success": True, "config": cfg})

# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Mercedes-Benz Active Reception — Starting")
    logger.info("=" * 60)
    logger.info("Dashboard: http://%s:%d", HOST, PORT)
    logger.info("Database dir: %s", DATABASE_DIR)

    # Pre-initialize YOLO in background
    def _init():
        controller.initialize_yolo()
    threading.Thread(target=_init, daemon=True).start()

    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, use_reloader=False)
