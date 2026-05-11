from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import IdentityCapsule, IdentityPatch

DEFAULT_DB_PATH = Path(".anchor") / "anchor.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                agent_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                capsule_json TEXT NOT NULL,
                capsule_hash TEXT,
                previous_hash TEXT,
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
                requested_by TEXT,
                approved_by TEXT,
                applied_by TEXT,
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
        _ensure_column(conn, "identities", "capsule_hash", "TEXT")
        _ensure_column(conn, "identities", "previous_hash", "TEXT")
        _ensure_column(conn, "patches", "requested_by", "TEXT")
        _ensure_column(conn, "patches", "approved_by", "TEXT")
        _ensure_column(conn, "patches", "applied_by", "TEXT")
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


def _canonical_capsule_payload(capsule: IdentityCapsule) -> str:
    payload = capsule.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_capsule_hash(capsule: IdentityCapsule) -> str:
    canonical = _canonical_capsule_payload(capsule)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coalesce_hash(row: sqlite3.Row) -> str:
    stored = row["capsule_hash"]
    if isinstance(stored, str) and stored.strip():
        return stored
    capsule = IdentityCapsule.model_validate_json(row["capsule_json"])
    return compute_capsule_hash(capsule)


def _record_event(
    conn: sqlite3.Connection,
    agent_id: str,
    event_type: str,
    payload: dict[str, Any],
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO events(agent_id, event_type, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (agent_id, event_type, json.dumps(payload), created_at),
    )


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


def load_identity_integrity(agent_id: str, path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    conn = _connect(path)
    try:
        row = conn.execute(
            """
            SELECT agent_id, version, capsule_json, capsule_hash, previous_hash
            FROM identities
            WHERE agent_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (agent_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "agent_id": row["agent_id"],
            "version": int(row["version"]),
            "capsule_hash": _coalesce_hash(row),
            "previous_hash": row["previous_hash"],
        }
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

        previous_hash: str | None = None
        if latest > 0:
            previous_row = conn.execute(
                """
                SELECT capsule_json, capsule_hash
                FROM identities
                WHERE agent_id = ? AND version = ?
                """,
                (stored.agent_id, latest),
            ).fetchone()
            if previous_row is not None:
                previous_hash = _coalesce_hash(previous_row)

        capsule_hash = compute_capsule_hash(stored)
        payload = stored.model_dump_json()
        conn.execute(
            """
            INSERT INTO identities(agent_id, version, capsule_json, capsule_hash, previous_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (stored.agent_id, next_version, payload, capsule_hash, previous_hash, now_iso, now_iso),
        )
        _record_event(
            conn,
            stored.agent_id,
            "identity_saved",
            {"version": next_version, "capsule_hash": capsule_hash, "previous_hash": previous_hash},
            now_iso,
        )
        conn.commit()
        return stored
    finally:
        conn.close()


def append_patch(
    agent_id: str,
    patch: IdentityPatch,
    path: str | Path = DEFAULT_DB_PATH,
    requested_by: str | None = None,
    approved_by: str | None = None,
    applied_by: str | None = None,
) -> None:
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
            INSERT INTO patches(
                agent_id, from_version, to_version, patch_json, requested_by, approved_by, applied_by, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                patch.from_version,
                to_version,
                patch.model_dump_json(),
                requested_by,
                approved_by,
                applied_by,
                now_iso,
            ),
        )
        _record_event(
            conn,
            agent_id,
            "patch_appended",
            {
                "from_version": patch.from_version,
                "to_version": to_version,
                "requested_by": requested_by,
                "approved_by": approved_by,
                "applied_by": applied_by,
            },
            now_iso,
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

        previous_row = conn.execute(
            """
            SELECT capsule_json, capsule_hash
            FROM identities
            WHERE agent_id = ? AND version = ?
            """,
            (agent_id, latest),
        ).fetchone()
        previous_hash = _coalesce_hash(previous_row) if previous_row is not None else None
        capsule_hash = compute_capsule_hash(capsule)
        conn.execute(
            """
            INSERT INTO identities(agent_id, version, capsule_json, capsule_hash, previous_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (agent_id, next_version, payload, capsule_hash, previous_hash, now_iso, now_iso),
        )
        _record_event(
            conn,
            agent_id,
            "rollback",
            {
                "from_version": latest,
                "target_version": version,
                "to_version": next_version,
                "capsule_hash": capsule_hash,
                "previous_hash": previous_hash,
            },
            now_iso,
        )
        conn.commit()
        return capsule
    finally:
        conn.close()


def export_json(agent_id: str, path: str | Path, db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    capsule = load_identity(agent_id, db_path)
    if capsule is None:
        raise ValueError(f"agent '{agent_id}' not found")
    integrity = load_identity_integrity(agent_id, db_path)
    if integrity is None:
        raise ValueError(f"integrity metadata not found for agent '{agent_id}'")
    envelope = {
        "agent_id": capsule.agent_id,
        "version": capsule.version,
        "capsule_hash": integrity["capsule_hash"],
        "previous_hash": integrity["previous_hash"],
        "exported_at": utc_now_iso(),
        "capsule": capsule.model_dump(mode="json"),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return out_path


def import_json(path: str | Path, db_path: str | Path = DEFAULT_DB_PATH, force: bool = False) -> IdentityCapsule:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"agent_id", "version", "capsule_hash", "previous_hash", "exported_at", "capsule"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"import envelope missing required keys: {', '.join(missing)}")

    capsule = IdentityCapsule.model_validate(payload["capsule"])
    if capsule.agent_id != payload["agent_id"]:
        raise ValueError("import envelope agent_id does not match capsule.agent_id")

    computed_hash = compute_capsule_hash(capsule)
    if payload["capsule_hash"] != computed_hash and not force:
        raise ValueError("import rejected: capsule_hash mismatch (use --force to override)")

    local = load_identity(capsule.agent_id, db_path)
    local_integrity = load_identity_integrity(capsule.agent_id, db_path)
    incoming_version = int(payload["version"])
    incoming_previous_hash = payload["previous_hash"]

    if local is None:
        if incoming_version != 1 and not force:
            raise ValueError("import rejected: version lineage mismatch (expected version 1 for new agent)")
        if incoming_previous_hash not in (None, "") and not force:
            raise ValueError("import rejected: previous_hash must be null for new agent")
    else:
        assert local_integrity is not None
        expected_next = local.version + 1
        if incoming_version != expected_next and not force:
            raise ValueError(
                f"import rejected: version lineage mismatch (expected {expected_next}, got {incoming_version})"
            )
        if incoming_previous_hash != local_integrity["capsule_hash"] and not force:
            raise ValueError("import rejected: previous_hash lineage mismatch (use --force to override)")

    saved = save_identity(capsule, db_path)
    conn = _connect(db_path)
    try:
        now_iso = utc_now_iso()
        _record_event(
            conn,
            saved.agent_id,
            "import",
            {
                "path": str(Path(path)),
                "force": force,
                "imported_version": incoming_version,
                "imported_capsule_hash": payload["capsule_hash"],
                "saved_version": saved.version,
            },
            now_iso,
        )
        conn.commit()
    finally:
        conn.close()
    return saved
