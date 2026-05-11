from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .models import IdentityCapsule, IdentityPatch

DEFAULT_DB_PATH = Path(".anchor") / "anchor.db"
CHECKPOINT_DIR_NAME = "checkpoints"
CHECKPOINT_PRIVATE_KEY_NAME = "checkpoint_ed25519_private.pem"
CHECKPOINT_PUBLIC_KEY_NAME = "checkpoint_ed25519_public.pem"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_root(path: str | Path) -> Path:
    return Path(path).parent


def _checkpoint_dir(path: str | Path) -> Path:
    return _db_root(path) / CHECKPOINT_DIR_NAME


def _checkpoint_path(agent_id: str, path: str | Path) -> Path:
    return _checkpoint_dir(path) / f"{agent_id}.checkpoint.json"


def _private_key_path(path: str | Path) -> Path:
    return _db_root(path) / CHECKPOINT_PRIVATE_KEY_NAME


def _public_key_path(path: str | Path) -> Path:
    return _db_root(path) / CHECKPOINT_PUBLIC_KEY_NAME


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key_bytes = _private_key_path(path).read_bytes()
    return serialization.load_pem_private_key(key_bytes, password=None)


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    key_bytes = _public_key_path(path).read_bytes()
    return serialization.load_pem_public_key(key_bytes)


def ensure_checkpoint_keypair(path: str | Path = DEFAULT_DB_PATH) -> tuple[Path, Path]:
    private_path = _private_key_path(path)
    public_path = _public_key_path(path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    if private_path.exists() and public_path.exists():
        return private_path, public_path

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)
    return private_path, public_path


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


def apply_and_record(
    patch: IdentityPatch,
    path: str | Path = DEFAULT_DB_PATH,
    requested_by: str | None = None,
    approved_by: str | None = None,
    applied_by: str | None = None,
) -> IdentityCapsule:
    from .compiler import apply_patch as apply_patch_compiler

    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        latest_row = conn.execute(
            """
            SELECT version, capsule_json, capsule_hash
            FROM identities
            WHERE agent_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (patch.agent_id,),
        ).fetchone()
        if latest_row is None:
            raise ValueError(f"agent '{patch.agent_id}' not found")

        current_version = int(latest_row["version"])
        if patch.from_version != current_version:
            raise ValueError(
                "version mismatch: "
                f"patch from_version={patch.from_version} current version={current_version}"
            )

        current_capsule = IdentityCapsule.model_validate_json(latest_row["capsule_json"])
        updated = apply_patch_compiler(current_capsule, patch)
        now_iso = utc_now_iso()
        next_version = current_version + 1
        updated.version = next_version
        updated.updated_at = datetime.fromisoformat(now_iso)
        updated.rollback_version = max(1, updated.rollback_version)

        previous_hash = _coalesce_hash(latest_row)
        capsule_hash = compute_capsule_hash(updated)
        payload = updated.model_dump_json()
        conn.execute(
            """
            INSERT INTO identities(agent_id, version, capsule_json, capsule_hash, previous_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (updated.agent_id, next_version, payload, capsule_hash, previous_hash, now_iso, now_iso),
        )
        conn.execute(
            """
            INSERT INTO patches(
                agent_id, from_version, to_version, patch_json, requested_by, approved_by, applied_by, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                updated.agent_id,
                patch.from_version,
                next_version,
                patch.model_dump_json(),
                requested_by,
                approved_by,
                applied_by,
                now_iso,
            ),
        )
        _record_event(
            conn,
            updated.agent_id,
            "patch_applied",
            {
                "from_version": patch.from_version,
                "to_version": next_version,
                "capsule_hash": capsule_hash,
                "previous_hash": previous_hash,
                "requested_by": requested_by,
                "approved_by": approved_by,
                "applied_by": applied_by,
            },
            now_iso,
        )
        conn.commit()
        return updated
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


def import_json(
    path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    force_lineage: bool = False,
    force_hash: bool = False,
) -> IdentityCapsule:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"agent_id", "version", "capsule_hash", "previous_hash", "exported_at", "capsule"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"import envelope missing required keys: {', '.join(missing)}")

    capsule = IdentityCapsule.model_validate(payload["capsule"])
    if capsule.agent_id != payload["agent_id"]:
        raise ValueError("import envelope agent_id does not match capsule.agent_id")

    computed_hash = compute_capsule_hash(capsule)
    if payload["capsule_hash"] != computed_hash and not force_hash:
        raise ValueError("import rejected: capsule_hash mismatch (use --force-hash to override)")

    local = load_identity(capsule.agent_id, db_path)
    local_integrity = load_identity_integrity(capsule.agent_id, db_path)
    incoming_version = int(payload["version"])
    incoming_previous_hash = payload["previous_hash"]

    if local is None:
        if incoming_version != 1 and not force_lineage:
            raise ValueError("import rejected: version lineage mismatch (expected version 1 for new agent)")
        if incoming_previous_hash not in (None, "") and not force_lineage:
            raise ValueError("import rejected: previous_hash must be null for new agent")
    else:
        assert local_integrity is not None
        expected_next = local.version + 1
        if incoming_version != expected_next and not force_lineage:
            raise ValueError(
                f"import rejected: version lineage mismatch (expected {expected_next}, got {incoming_version})"
            )
        if incoming_previous_hash != local_integrity["capsule_hash"] and not force_lineage:
            raise ValueError("import rejected: previous_hash lineage mismatch (use --force-lineage to override)")

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
                "force_lineage": force_lineage,
                "force_hash": force_hash,
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


def load_history(agent_id: str, path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    conn = _connect(path)
    try:
        identity_rows = conn.execute(
            """
            SELECT version, capsule_hash, previous_hash, updated_at
            FROM identities
            WHERE agent_id = ?
            ORDER BY version
            """,
            (agent_id,),
        ).fetchall()
        patch_rows = conn.execute(
            """
            SELECT to_version, patch_json, requested_by, approved_by, applied_by, created_at
            FROM patches
            WHERE agent_id = ?
            ORDER BY to_version
            """,
            (agent_id,),
        ).fetchall()
    finally:
        conn.close()

    by_version: dict[int, dict[str, Any]] = {}
    for row in patch_rows:
        patch_payload = json.loads(row["patch_json"])
        by_version[int(row["to_version"])] = {
            "summary": patch_payload.get("summary", ""),
            "requested_by": row["requested_by"],
            "approved_by": row["approved_by"],
            "applied_by": row["applied_by"],
            "timestamp": row["created_at"],
        }

    history: list[dict[str, Any]] = []
    for row in identity_rows:
        version = int(row["version"])
        patch_meta = by_version.get(
            version,
            {
                "summary": "",
                "requested_by": None,
                "approved_by": None,
                "applied_by": None,
                "timestamp": row["updated_at"],
            },
        )
        history.append(
            {
                "version": version,
                "hash": row["capsule_hash"],
                "previous_hash": row["previous_hash"],
                "summary": patch_meta["summary"],
                "requested_by": patch_meta["requested_by"],
                "approved_by": patch_meta["approved_by"],
                "applied_by": patch_meta["applied_by"],
                "timestamp": patch_meta["timestamp"],
            }
        )
    return history


def create_checkpoint(agent_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    integrity = load_identity_integrity(agent_id, db_path)
    if integrity is None:
        raise ValueError(f"agent '{agent_id}' not found")
    ensure_checkpoint_keypair(db_path)
    private_key = _load_private_key(db_path)
    public_key = _load_public_key(db_path)

    payload = {
        "agent_id": integrity["agent_id"],
        "version": integrity["version"],
        "capsule_hash": integrity["capsule_hash"],
        "previous_hash": integrity["previous_hash"],
        "created_at": utc_now_iso(),
    }
    signature = private_key.sign(_canonical_json_bytes(payload))
    checkpoint = {
        **payload,
        "public_key": base64.b64encode(
            public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("utf-8"),
        "signature": base64.b64encode(signature).decode("utf-8"),
    }
    out_path = _checkpoint_path(agent_id, db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return out_path


def verify_checkpoint(agent_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path(agent_id, db_path)
    if not checkpoint_path.exists():
        return {"ok": False, "reason": "checkpoint missing", "checkpoint_path": str(checkpoint_path)}

    private_path = _private_key_path(db_path)
    public_path = _public_key_path(db_path)
    if not private_path.exists() or not public_path.exists():
        return {"ok": False, "reason": "checkpoint keypair missing", "checkpoint_path": str(checkpoint_path)}

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload_keys = {"agent_id", "version", "capsule_hash", "previous_hash", "created_at"}
    if not payload_keys.issubset(checkpoint.keys()):
        return {"ok": False, "reason": "checkpoint payload invalid", "checkpoint_path": str(checkpoint_path)}

    payload = {key: checkpoint[key] for key in payload_keys}
    try:
        signature = base64.b64decode(str(checkpoint["signature"]))
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "checkpoint signature invalid encoding", "checkpoint_path": str(checkpoint_path)}

    try:
        public_key = _load_public_key(db_path)
        public_key.verify(signature, _canonical_json_bytes(payload))
    except (InvalidSignature, ValueError):
        return {"ok": False, "reason": "checkpoint signature verification failed", "checkpoint_path": str(checkpoint_path)}

    integrity = load_identity_integrity(agent_id, db_path)
    if integrity is None:
        return {"ok": False, "reason": "agent not found in DB", "checkpoint_path": str(checkpoint_path)}

    if int(checkpoint["version"]) != int(integrity["version"]):
        return {"ok": False, "reason": "version mismatch vs checkpoint", "checkpoint_path": str(checkpoint_path)}
    if str(checkpoint["capsule_hash"]) != str(integrity["capsule_hash"]):
        return {"ok": False, "reason": "capsule_hash mismatch vs checkpoint", "checkpoint_path": str(checkpoint_path)}
    if checkpoint.get("previous_hash") != integrity.get("previous_hash"):
        return {"ok": False, "reason": "previous_hash mismatch vs checkpoint", "checkpoint_path": str(checkpoint_path)}

    return {
        "ok": True,
        "reason": "verified",
        "checkpoint_path": str(checkpoint_path),
        "agent_id": integrity["agent_id"],
        "version": integrity["version"],
        "capsule_hash": integrity["capsule_hash"],
        "previous_hash": integrity["previous_hash"],
    }
