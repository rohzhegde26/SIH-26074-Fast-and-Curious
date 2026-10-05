"""
src/api/feedback_store.py

SQLite-backed persistence store for KMF Dairy Secretary ground validations.
Operates with WAL journal mode, parameterized queries, and thread-safe connections.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "serving" / "nandini_feedback.db"
DEFAULT_JSON_PATH = ROOT / "data" / "serving" / "nandini_feedback.json"

_initialized_dbs = set()
_init_lock = threading.Lock()


def init_db(db_path: Optional[Path] = None, json_migration_path: Optional[Path] = None) -> None:
    """
    Initializes the SQLite schema and automatically migrates legacy JSON seed fixtures
    on first run if the table is empty. Thread-safe and runs once per database path.
    """
    target_db = (db_path or DEFAULT_DB_PATH).resolve()
    db_key = str(target_db)

    with _init_lock:
        if db_key in _initialized_dbs and target_db.exists():
            return
        target_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(target_db), timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nandini_feedback (
                    id TEXT PRIMARY KEY,
                    lgd_code TEXT NOT NULL,
                    panchayat_name TEXT,
                    rained_bool INTEGER NOT NULL,
                    mm_reported REAL,
                    observer_role TEXT,
                    milk_center_id TEXT,
                    observation_period TEXT,
                    submitted_at TEXT NOT NULL,
                    recalibration_flagged INTEGER DEFAULT 0,
                    source TEXT NOT NULL CHECK(source IN ('seed_fixture', 'field_submission')),
                    device_info TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nandini_lgd ON nandini_feedback(lgd_code);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nandini_source ON nandini_feedback(source);")
            conn.commit()

            # Check if table is empty; if so, perform one-time migration of JSON seed fixtures
            cursor = conn.execute("SELECT COUNT(*) FROM nandini_feedback;")
            row = cursor.fetchone()
            count = row[0] if row else 0

            json_path = json_migration_path or DEFAULT_JSON_PATH
            should_migrate = (count == 0) and (
                (json_migration_path is not None and json_migration_path.exists())
                or (target_db == DEFAULT_DB_PATH.resolve() and json_path.exists())
            )
            if should_migrate:
                try:
                    with open(json_path, encoding="utf-8") as f:
                        entries = json.load(f)
                    if isinstance(entries, list) and len(entries) > 0:
                        insert_sql = """
                            INSERT OR IGNORE INTO nandini_feedback (
                                id, lgd_code, panchayat_name, rained_bool, mm_reported,
                                observer_role, milk_center_id, observation_period,
                                submitted_at, recalibration_flagged, source, device_info
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'seed_fixture', ?);
                        """
                        records = [
                            (
                                entry.get("validation_id") or f"seed_{idx:04d}",
                                str(entry.get("lgd_code", "")),
                                entry.get("panchayat_name", ""),
                                1 if entry.get("rained_bool") else 0,
                                entry.get("mm_reported"),
                                entry.get("observer_role", "DAIRY_SECRETARY"),
                                entry.get("milk_center_id", ""),
                                entry.get("observation_period", "LAST_12_HOURS"),
                                entry.get("recorded_at_utc", datetime.now(timezone.utc).isoformat()),
                                1 if entry.get("recalibration_flagged") else 0,
                                entry.get("device_info"),
                            )
                            for idx, entry in enumerate(entries)
                        ]
                        conn.executemany(insert_sql, records)
                        conn.commit()
                except Exception:
                    pass
        _initialized_dbs.add(db_key)


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a SQLite connection with timeout and busy_timeout configured."""
    target_path = (db_path or DEFAULT_DB_PATH).resolve()
    init_db(target_path)
    conn = sqlite3.connect(str(target_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=10000;")
    conn.row_factory = sqlite3.Row
    return conn


def insert_feedback(entry: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """
    Inserts a single feedback entry with parameterized query.
    Defaults source to 'field_submission' if not explicitly provided.
    """
    target_db = (db_path or DEFAULT_DB_PATH).resolve()

    val_id = entry.get("validation_id") or entry.get("id")
    lgd = str(entry.get("lgd_code", ""))
    p_name = entry.get("panchayat_name", "")
    rained = 1 if entry.get("rained_bool") else 0
    mm = entry.get("mm_reported")
    role = entry.get("observer_role", "DAIRY_SECRETARY")
    center = entry.get("milk_center_id", "")
    period = entry.get("observation_period", "LAST_12_HOURS")
    submitted = entry.get("recorded_at_utc") or entry.get("submitted_at") or datetime.now(timezone.utc).isoformat()
    recal = 1 if entry.get("recalibration_flagged") else 0
    source = entry.get("source", "field_submission")
    device = entry.get("device_info")

    sql = """
        INSERT INTO nandini_feedback (
            id, lgd_code, panchayat_name, rained_bool, mm_reported,
            observer_role, milk_center_id, observation_period,
            submitted_at, recalibration_flagged, source, device_info
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    with get_connection(target_db) as conn:
        conn.execute(
            sql,
            (val_id, lgd, p_name, rained, mm, role, center, period, submitted, recal, source, device),
        )
        conn.commit()


def get_feedback_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Calculates aggregate validation network metrics, reporting real vs seed submissions separately.
    """
    target_db = (db_path or DEFAULT_DB_PATH).resolve()

    with get_connection(target_db) as conn:
        cur = conn.execute("SELECT COUNT(*) as total FROM nandini_feedback;")
        total = cur.fetchone()["total"]

        if total == 0:
            return {
                "total_validations": 0,
                "rain_reported_count": 0,
                "no_rain_reported_count": 0,
                "model_agreement_rate_pct": 100.0,
                "active_dairy_centers": 0,
                "real_count": 0,
                "seed_count": 0,
            }

        cur = conn.execute("SELECT COUNT(*) as cnt FROM nandini_feedback WHERE rained_bool = 1;")
        rain_cnt = cur.fetchone()["cnt"]
        no_rain_cnt = total - rain_cnt

        cur = conn.execute("SELECT COUNT(*) as cnt FROM nandini_feedback WHERE recalibration_flagged = 0;")
        agreed_cnt = cur.fetchone()["cnt"]
        agreement_rate = round((agreed_cnt / total) * 100.0, 1)

        cur = conn.execute(
            "SELECT COUNT(DISTINCT milk_center_id) as cnt FROM nandini_feedback WHERE milk_center_id IS NOT NULL AND milk_center_id != '';"
        )
        centers = cur.fetchone()["cnt"]

        cur = conn.execute("SELECT COUNT(*) as cnt FROM nandini_feedback WHERE source = 'field_submission';")
        real_count = cur.fetchone()["cnt"]

        cur = conn.execute("SELECT COUNT(*) as cnt FROM nandini_feedback WHERE source = 'seed_fixture';")
        seed_count = cur.fetchone()["cnt"]

    return {
        "total_validations": total,
        "rain_reported_count": rain_cnt,
        "no_rain_reported_count": no_rain_cnt,
        "model_agreement_rate_pct": agreement_rate,
        "active_dairy_centers": max(centers, 1),
        "real_count": real_count,
        "seed_count": seed_count,
    }


def list_feedbacks(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns all feedback entries ordered by submission timestamp descending."""
    target_db = (db_path or DEFAULT_DB_PATH).resolve()

    with get_connection(target_db) as conn:
        cur = conn.execute("SELECT * FROM nandini_feedback ORDER BY submitted_at DESC;")
        rows = cur.fetchall()
        return [
            {
                "validation_id": row["id"],
                "lgd_code": row["lgd_code"],
                "panchayat_name": row["panchayat_name"],
                "rained_bool": bool(row["rained_bool"]),
                "mm_reported": row["mm_reported"],
                "observer_role": row["observer_role"],
                "milk_center_id": row["milk_center_id"],
                "observation_period": row["observation_period"],
                "recorded_at_utc": row["submitted_at"],
                "recalibration_flagged": bool(row["recalibration_flagged"]),
                "source": row["source"],
                "device_info": row["device_info"],
            }
            for row in rows
        ]
