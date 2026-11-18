"""
Database Manager – SQLite operations for vehicle session storage.
"""
import os
import sqlite3
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_COLUMNS_NO_BLOB = """
    id, session_id, vehicle_number, vehicle_number_normalized,
    capture_date, capture_timestamp, capture_time_12h, folder_path,
    front_image_path, back_image_path, left_image_path, right_image_path,
    front_confidence, back_confidence, left_confidence, right_confidence,
    capture_status, created_at
"""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS vehicle_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    vehicle_number TEXT NOT NULL DEFAULT 'UNKNOWN',
    vehicle_number_normalized TEXT NOT NULL DEFAULT 'UNKNOWN',
    capture_date TEXT NOT NULL,
    capture_timestamp TEXT NOT NULL,
    capture_time_12h TEXT NOT NULL,
    folder_path TEXT NOT NULL,
    front_image_path TEXT,
    back_image_path TEXT,
    left_image_path TEXT,
    right_image_path TEXT,
    front_image_blob BLOB,
    back_image_blob BLOB,
    left_image_blob BLOB,
    right_image_blob BLOB,
    front_confidence REAL DEFAULT 0,
    back_confidence REAL DEFAULT 0,
    left_confidence REAL DEFAULT 0,
    right_confidence REAL DEFAULT 0,
    capture_status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_vehicle_number ON vehicle_sessions(vehicle_number_normalized);
CREATE INDEX IF NOT EXISTS idx_capture_date ON vehicle_sessions(capture_date);
CREATE INDEX IF NOT EXISTS idx_capture_timestamp ON vehicle_sessions(capture_timestamp);
CREATE INDEX IF NOT EXISTS idx_session_id ON vehicle_sessions(session_id);
"""


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        try:
            conn = self._get_conn()
            conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
            # Safe schema migration for existing SQLite databases
            existing_cols = [c[1] for c in conn.execute("PRAGMA table_info(vehicle_sessions)").fetchall()]
            for blob_col in ["front_image_blob", "back_image_blob", "left_image_blob", "right_image_blob"]:
                if blob_col not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE vehicle_sessions ADD COLUMN {blob_col} BLOB")
                        logger.info("Migrated SQL schema: added %s BLOB column", blob_col)
                    except Exception as e:
                        logger.warning("Column migration %s: %s", blob_col, e)
            conn.commit()
            conn.close()
            logger.info("Database initialized: %s", self.db_path)
        except Exception as e:
            logger.error("Database init failed: %s", e)
            raise

    def insert_session(self, data: dict) -> bool:
        # Extract binary image data for SQL BLOB storage (from dict or by reading from path)
        blobs = {}
        for cam in ["front", "back", "left", "right"]:
            b = data.get(f"{cam}_image_blob")
            if b is None:
                p = data.get(f"{cam}_image_path")
                if p and os.path.isfile(p):
                    try:
                        with open(p, "rb") as f:
                            b = f.read()
                    except Exception:
                        b = None
            blobs[cam] = sqlite3.Binary(b) if b else None

        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO vehicle_sessions
                (session_id, vehicle_number, vehicle_number_normalized,
                 capture_date, capture_timestamp, capture_time_12h,
                 folder_path, front_image_path, back_image_path,
                 left_image_path, right_image_path,
                 front_image_blob, back_image_blob, left_image_blob, right_image_blob,
                 front_confidence, back_confidence, left_confidence, right_confidence,
                 capture_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["session_id"],
                data.get("vehicle_number", "UNKNOWN"),
                data.get("vehicle_number_normalized", "UNKNOWN"),
                data["capture_date"],
                data["capture_timestamp"],
                data["capture_time_12h"],
                data["folder_path"],
                data.get("front_image_path", ""),
                data.get("back_image_path", ""),
                data.get("left_image_path", ""),
                data.get("right_image_path", ""),
                blobs.get("front"),
                blobs.get("back"),
                blobs.get("left"),
                blobs.get("right"),
                data.get("front_confidence", 0),
                data.get("back_confidence", 0),
                data.get("left_confidence", 0),
                data.get("right_confidence", 0),
                data.get("capture_status", "success"),
                datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()
            logger.info("Session inserted into SQL (with image BLOBs): %s", data["session_id"])
            return True
        except sqlite3.IntegrityError:
            logger.warning("Duplicate session: %s", data.get("session_id"))
            return False
        except Exception as e:
            logger.error("Insert failed: %s", e)
            return False

    def update_vehicle_number(self, session_id: str, vehicle_number: str) -> bool:
        normalized = vehicle_number.upper().replace(" ", "").replace("-", "")
        try:
            conn = self._get_conn()
            conn.execute("""
                UPDATE vehicle_sessions
                SET vehicle_number = ?, vehicle_number_normalized = ?
                WHERE session_id = ?
            """, (vehicle_number.upper(), normalized, session_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error("Update vehicle number failed: %s", e)
            return False

    def search_sessions(self, vehicle_number: str = None, date: str = None,
                        time_from: str = None, time_to: str = None,
                        limit: int = 50, offset: int = 0) -> list:
        query = f"SELECT {_COLUMNS_NO_BLOB} FROM vehicle_sessions WHERE 1=1"
        params = []

        if vehicle_number:
            normalized = vehicle_number.upper().replace(" ", "").replace("-", "")
            query += " AND vehicle_number_normalized LIKE ?"
            params.append(f"%{normalized}%")

        if date:
            query += " AND capture_date = ?"
            params.append(date)

        if time_from and time_to:
            # HTML time input sends HH:MM (24-hour). Compare against 24-hour time from capture_timestamp
            query += " AND time(capture_timestamp) >= time(?) AND time(capture_timestamp) <= time(?)"
            params.append(time_from)
            params.append(time_to)
        elif time_from:
            query += " AND time(capture_timestamp) >= time(?)"
            params.append(time_from)
        elif time_to:
            query += " AND time(capture_timestamp) <= time(?)"
            params.append(time_to)

        query += " ORDER BY capture_timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            conn = self._get_conn()
            rows = conn.execute(query, params).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def get_session(self, session_id: str) -> dict | None:
        try:
            conn = self._get_conn()
            row = conn.execute(
                f"SELECT {_COLUMNS_NO_BLOB} FROM vehicle_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error("Get session failed: %s", e)
            return None

    def get_session_image_blob(self, session_id: str, camera: str) -> bytes | None:
        """Fetch raw binary image data directly from SQL database BLOB storage."""
        col = f"{camera.lower()}_image_blob"
        try:
            conn = self._get_conn()
            row = conn.execute(
                f"SELECT {col} FROM vehicle_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            conn.close()
            if row and row[0]:
                return bytes(row[0])
            return None
        except Exception as e:
            logger.error("Failed to fetch image blob for %s/%s: %s", session_id, camera, e)
            return None

    def get_stats(self) -> dict:
        try:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM vehicle_sessions").fetchone()[0]
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = conn.execute(
                "SELECT COUNT(*) FROM vehicle_sessions WHERE capture_date = ?",
                (today,)
            ).fetchone()[0]
            success = conn.execute(
                "SELECT COUNT(*) FROM vehicle_sessions WHERE capture_status = 'success'"
            ).fetchone()[0]
            conn.close()
            return {
                "total_sessions": total,
                "today_sessions": today_count,
                "success_sessions": success,
                "total_vehicles": total,
            }
        except Exception as e:
            logger.error("Stats failed: %s", e)
            return {"total_sessions": 0, "today_sessions": 0, "success_sessions": 0, "total_vehicles": 0}

    def get_recent(self, limit: int = 10) -> list:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                f"SELECT {_COLUMNS_NO_BLOB} FROM vehicle_sessions ORDER BY capture_timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Get recent failed: %s", e)
            return []

    def delete_session(self, session_id: str, database_dir: str = None) -> bool:
        """Delete a single session from DB and its image folder from disk."""
        session = self.get_session(session_id)
        if not session:
            return False

        # 1. Delete image folder from disk
        folder = session.get("folder_path")
        if folder and os.path.isdir(folder):
            try:
                import shutil
                if database_dir:
                    abs_db = os.path.abspath(database_dir)
                    abs_folder = os.path.abspath(folder)
                    # Protect against deleting root or database_dir itself
                    if abs_folder.startswith(abs_db) and abs_folder != abs_db:
                        shutil.rmtree(abs_folder, ignore_errors=True)
                        logger.info("Deleted session folder: %s", folder)
                else:
                    shutil.rmtree(folder, ignore_errors=True)
                    logger.info("Deleted session folder: %s", folder)
            except Exception as e:
                logger.error("Failed to delete folder %s: %s", folder, e)

        # 2. Delete record from database
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM vehicle_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            conn.close()
            logger.info("Deleted session %s from database", session_id)
            return True
        except Exception as e:
            logger.error("Failed to delete session %s from DB: %s", session_id, e)
            return False

    def clear_all_sessions(self, database_dir: str = None) -> int:
        """Delete all vehicle sessions from DB and all session folders from disk."""
        try:
            conn = self._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM vehicle_sessions").fetchone()[0]
            conn.execute("DELETE FROM vehicle_sessions")
            conn.commit()
            conn.close()
            logger.info("Cleared %d sessions from database", count)
        except Exception as e:
            logger.error("Failed to clear sessions: %s", e)
            return 0

        # Clean all session folders from database_dir
        if database_dir and os.path.isdir(database_dir):
            import shutil
            abs_db = os.path.abspath(database_dir)
            for item in os.listdir(database_dir):
                item_path = os.path.join(database_dir, item)
                # Only delete session subfolders; keep database file, .gitkeep, etc.
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path, ignore_errors=True)
                        logger.info("Deleted session folder: %s", item_path)
                    except Exception as e:
                        logger.error("Failed to delete %s: %s", item_path, e)

        return count

    def recover_from_filesystem(self, database_dir: str):
        """Scan database directory and recover sessions from metadata.json files."""
        if not os.path.isdir(database_dir):
            return
        for folder_name in os.listdir(database_dir):
            folder_path = os.path.join(database_dir, folder_name)
            meta_path = os.path.join(folder_path, "metadata.json")
            if os.path.isdir(folder_path) and os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    sid = meta.get("session_id", folder_name)
                    existing = self.get_session(sid)
                    if existing:
                        continue
                    self.insert_session({
                        "session_id": sid,
                        "vehicle_number": meta.get("vehicle_number", "UNKNOWN"),
                        "vehicle_number_normalized": meta.get("vehicle_number_normalized", "UNKNOWN"),
                        "capture_date": meta.get("capture_date", ""),
                        "capture_timestamp": meta.get("capture_timestamp", ""),
                        "capture_time_12h": meta.get("capture_time_12h", ""),
                        "folder_path": folder_path,
                        "front_image_path": os.path.join(folder_path, "front.jpg"),
                        "back_image_path": os.path.join(folder_path, "back.jpg"),
                        "left_image_path": os.path.join(folder_path, "left.jpg"),
                        "right_image_path": os.path.join(folder_path, "right.jpg"),
                        "front_confidence": meta.get("detection_confidence", {}).get("front", 0),
                        "back_confidence": meta.get("detection_confidence", {}).get("back", 0),
                        "left_confidence": meta.get("detection_confidence", {}).get("left", 0),
                        "right_confidence": meta.get("detection_confidence", {}).get("right", 0),
                        "capture_status": meta.get("capture_status", "success"),
                    })
                    logger.info("Recovered session from filesystem: %s", sid)
                except Exception as e:
                    logger.warning("Failed to recover %s: %s", folder_name, e)
