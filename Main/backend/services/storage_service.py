"""
Storage Service — Filesystem utilities for the database directory.
"""
import os
import logging

logger = logging.getLogger(__name__)


def get_storage_info(database_dir: str) -> dict:
    """Get storage usage info."""
    if not os.path.isdir(database_dir):
        return {"exists": False, "total_size": 0, "folder_count": 0}

    total_size = 0
    folder_count = 0
    file_count = 0

    for entry in os.listdir(database_dir):
        path = os.path.join(database_dir, entry)
        if os.path.isdir(path):
            folder_count += 1
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
                    file_count += 1

    return {
        "exists": True,
        "path": database_dir,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "folder_count": folder_count,
        "file_count": file_count,
    }


def check_video_files(video_sources: dict) -> dict:
    """Check existence and size of video files."""
    result = {}
    for name, path in video_sources.items():
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        result[name] = {
            "path": path,
            "exists": exists,
            "size_mb": round(size / (1024 * 1024), 2) if exists else 0,
        }
    return result
