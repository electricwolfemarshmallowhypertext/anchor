from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from anchor.models import IdentityCapsule, IdentityField, IdentityPatch
from anchor.store import (
    append_patch,
    export_json,
    import_json,
    init_db,
    load_identity,
    load_identity_integrity,
    rollback,
    save_identity,
)


def test_save_and_load_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    init_db(db_path)

    capsule = IdentityCapsule(agent_id="demo", purpose="Test persistence")
    saved = save_identity(capsule, db_path)
    loaded = load_identity("demo", db_path)

    assert saved.version == 1
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.purpose == "Test persistence"


def test_append_patch_and_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    init_db(db_path)

    original = save_identity(IdentityCapsule(agent_id="demo", purpose="v1"), db_path)
    updated = original.model_copy(deep=True)
    updated.purpose = "v2"
    saved_v2 = save_identity(updated, db_path)

    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        summary="purpose update",
        field_updates=[
            IdentityField(
                key="purpose",
                value="v2",
                source="explicit_user_instruction",
                confidence=0.95,
            )
        ],
    )
    append_patch("demo", patch, db_path)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT from_version, to_version FROM patches WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
            ("demo",),
        ).fetchone()
    finally:
        conn.close()

    restored = rollback("demo", 1, db_path)
    latest = load_identity("demo", db_path)

    assert row is not None
    assert row[0] == 1
    assert row[1] == 2
    assert saved_v2.version == 2
    assert restored.rollback_version == 1
    assert latest is not None
    assert latest.version == 3
    assert latest.purpose == "v1"


def test_export_json(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    out_path = tmp_path / "identity.anchor.json"
    save_identity(IdentityCapsule(agent_id="demo", purpose="export me"), db_path)

    exported = export_json("demo", out_path, db_path)
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert exported == out_path
    assert payload["agent_id"] == "demo"
    assert payload["capsule"]["purpose"] == "export me"
    assert isinstance(payload["capsule_hash"], str) and payload["capsule_hash"]
    assert "previous_hash" in payload


def test_rollback_restores_exact_previous_capsule_content(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    v1 = IdentityCapsule(
        agent_id="demo",
        purpose="first",
        working_style={"tone": "direct"},
        user_preferences=[
            IdentityField(
                key="user_preferences.prefers_laconic_technical_answers",
                value=True,
                source="explicit_user_instruction",
                confidence=0.99,
                evidence=["User: keep it concise."],
            )
        ],
        tool_boundaries=[{"tool": "web", "allowed_when": "verification needed"}],
    )
    saved_v1 = save_identity(v1, db_path)
    v2 = saved_v1.model_copy(deep=True)
    v2.purpose = "second"
    save_identity(v2, db_path)

    restored = rollback("demo", 1, db_path)

    assert restored.purpose == saved_v1.purpose
    assert restored.working_style == saved_v1.working_style
    assert [item.model_dump() for item in restored.user_preferences] == [
        item.model_dump() for item in saved_v1.user_preferences
    ]
    assert restored.tool_boundaries == saved_v1.tool_boundaries


def test_concurrent_saves_do_not_silently_corrupt_versions(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    init_db(db_path)

    def _save(index: int) -> int:
        saved = save_identity(IdentityCapsule(agent_id="demo", purpose=f"p{index}"), db_path)
        return saved.version

    with ThreadPoolExecutor(max_workers=4) as pool:
        versions = list(pool.map(_save, range(8)))

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT version FROM identities WHERE agent_id = ? ORDER BY version", ("demo",)).fetchall()
    finally:
        conn.close()

    stored_versions = [row[0] for row in rows]
    assert len(versions) == 8
    assert stored_versions == list(range(1, 9))


def test_identity_integrity_hash_chain(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    first = save_identity(IdentityCapsule(agent_id="demo", purpose="first"), db_path)
    second_capsule = first.model_copy(deep=True)
    second_capsule.purpose = "second"
    save_identity(second_capsule, db_path)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT version, capsule_hash, previous_hash FROM identities WHERE agent_id = ? ORDER BY version",
            ("demo",),
        ).fetchall()
    finally:
        conn.close()

    assert rows[0][1]
    assert rows[0][2] is None
    assert rows[1][1]
    assert rows[1][2] == rows[0][1]


def test_append_patch_stores_actor_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    original = save_identity(IdentityCapsule(agent_id="demo", purpose="v1"), db_path)
    updated = original.model_copy(deep=True)
    updated.purpose = "v2"
    save_identity(updated, db_path)
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        summary="actor test",
        field_updates=[
            IdentityField(
                key="purpose",
                value="v2",
                source="explicit_user_instruction",
                confidence=0.95,
            )
        ],
    )
    append_patch(
        "demo",
        patch,
        db_path,
        requested_by="alice",
        approved_by="bob",
        applied_by="charlie",
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT requested_by, approved_by, applied_by
            FROM patches
            WHERE agent_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("demo",),
        ).fetchone()
    finally:
        conn.close()
    assert row == ("alice", "bob", "charlie")


def test_import_rejects_lineage_mismatch_without_force(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    export_path = tmp_path / "identity.anchor.json"
    save_identity(IdentityCapsule(agent_id="demo", purpose="v1"), db_path)
    export_json("demo", export_path, db_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    payload["version"] = 99
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="version lineage mismatch"):
        import_json(export_path, db_path, force=False)


def test_import_force_allows_lineage_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    export_path = tmp_path / "identity.anchor.json"
    save_identity(IdentityCapsule(agent_id="demo", purpose="v1"), db_path)
    export_json("demo", export_path, db_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    payload["version"] = 99
    payload["capsule_hash"] = "invalid"
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    imported = import_json(export_path, db_path, force=True)
    latest_integrity = load_identity_integrity("demo", db_path)
    assert imported.version == 2
    assert latest_integrity is not None
    assert latest_integrity["version"] == 2
