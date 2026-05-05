from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anchor.models import IdentityCapsule, IdentityField, IdentityPatch
from anchor.store import append_patch, export_json, init_db, load_identity, rollback, save_identity


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
    assert payload["purpose"] == "export me"


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
