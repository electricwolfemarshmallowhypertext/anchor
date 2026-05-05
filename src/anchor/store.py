from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import IdentityCapsule, IdentityPatch

DEFAULT_DB_PATH = Path(".anchor") / "anchor.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                agent_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                capsule_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, version)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                patch_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_patches_agent_to_version
            ON patches(agent_id, to_version)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    init_db(path)
    conn = sqlite3.connect(Path(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_version(conn: sqlite3.Connection, agent_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) AS version FROM identities WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if row is None or row["version"] is None:
        return 0
    return int(row["version"])


def load_identity(agent_id: str, path: str | Path = DEFAULT_DB_PATH) -> IdentityCapsule | None:
    conn = _connect(path)
    try:
        row = conn.execute(
            """
            SELECT capsule_json
            FROM identities
            WHERE agent_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (agent_id,),
        ).fetchone()
        if row is None:
            return None
        return IdentityCapsule.model_validate_json(row["capsule_json"])
    finally:
        conn.close()


def save_identity(capsule: IdentityCapsule, path: str | Path = DEFAULT_DB_PATH) -> IdentityCapsule:
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        latest = _latest_version(conn, capsule.agent_id)
        next_version = latest + 1
        now_iso = utc_now_iso()
        stored = capsule.model_copy(deep=True)
        stored.version = next_version
        stored.rollback_version = max(1, stored.rollback_version)
        stored.updated_at = datetime.fromisoformat(now_iso)
        if latest == 0:
            stored.created_at = stored.updated_at
        payload = stored.model_dump_json()
        conn.execute(
            """
            INSERT INTO identities(agent_id, version, capsule_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (stored.agent_id, next_version, payload, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO events(agent_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                stored.agent_id,
                "identity_saved",
                json.dumps({"version": next_version}),
                now_iso,
            ),
        )
        conn.commit()
        return stored
    finally:
        conn.close()


def append_patch(agent_id: str, patch: IdentityPatch, path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        to_version = patch.from_version + 1
        identity_row = conn.execute(
            """
            SELECT 1
            FROM identities
            WHERE agent_id = ? AND version = ?
            LIMIT 1
            """,
            (agent_id, to_version),
        ).fetchone()
        if identity_row is None:
            raise ValueError(
                f"cannot append patch: identity version {to_version} does not exist for agent '{agent_id}'"
            )
        now_iso = utc_now_iso()
        conn.execute(
            """
            INSERT INTO patches(agent_id, from_version, to_version, patch_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                patch.from_version,
                to_version,
                patch.model_dump_json(),
                now_iso,
            ),
        )
        conn.execute(
            """
            INSERT INTO events(agent_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                agent_id,
                "patch_appended",
                json.dumps({"from_version": patch.from_version, "to_version": to_version}),
                now_iso,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def rollback(agent_id: str, version: int, path: str | Path = DEFAULT_DB_PATH) -> IdentityCapsule:
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        target_row = conn.execute(
            """
            SELECT capsule_json
            FROM identities
            WHERE agent_id = ? AND version = ?
            """,
            (agent_id, version),
        ).fetchone()
        if target_row is None:
            raise ValueError(f"version {version} not found for agent '{agent_id}'")

        latest = _latest_version(conn, agent_id)
        next_version = latest + 1
        now_iso = utc_now_iso()
        capsule = IdentityCapsule.model_validate_json(target_row["capsule_json"])
        capsule.version = next_version
        capsule.rollback_version = version
        capsule.updated_at = datetime.fromisoformat(now_iso)
        payload = capsule.model_dump_json()

        conn.execute(
            """
            INSERT INTO identities(agent_id, version, capsule_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agent_id, next_version, payload, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO events(agent_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                agent_id,
                "rollback",
                json.dumps({"from_version": latest, "target_version": version, "to_version": next_version}),
                now_iso,
            ),
        )
        conn.commit()
        return capsule
    finally:
        conn.close()


def export_json(agent_id: str, path: str | Path, db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    capsule = load_identity(agent_id, db_path)
    if capsule is None:
        raise ValueError(f"agent '{agent_id}' not found")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(capsule.model_dump_json(indent=2), encoding="utf-8")
    return out_path
