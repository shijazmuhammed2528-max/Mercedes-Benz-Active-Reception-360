"""
ZIP Service — Creates ZIP downloads for vehicle sessions.
"""
import os
import io
import zipfile
import logging

logger = logging.getLogger(__name__)


def create_session_zip(folder_path: str, vehicle_number: str = "VEHICLE") -> io.BytesIO | None:
    """Create a ZIP archive of a vehicle session folder."""
    if not os.path.isdir(folder_path):
        logger.error("Folder not found: %s", folder_path)
        return None

    try:
        buffer = io.BytesIO()
        zip_prefix = vehicle_number.replace(" ", "_")

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                if os.path.isfile(filepath):
                    # Store with relative path inside ZIP
                    arcname = f"{zip_prefix}/{filename}"
                    zf.write(filepath, arcname)

        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error("ZIP creation failed: %s", e)
        return None
