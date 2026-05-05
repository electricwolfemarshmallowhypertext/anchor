from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.cli import build_parser, main
from anchor.models import IdentityPatch
from anchor.store import load_identity


def test_compile_writes_patch_file_without_mutating_identity(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "anchor.db"
    session_path = tmp_path / "session.md"
    out_path = tmp_path / "patch.json"
    session_path.write_text("User: keep answers concise.", encoding="utf-8")

    assert main(["--db", str(db_path), "init", "demo"]) == 0
    before = load_identity("demo", db_path)
    assert before is not None
    assert before.version == 1

    def fake_compile_patch(capsule, transcript, client, model):
        assert transcript == "User: keep answers concise."
        return IdentityPatch(agent_id=capsule.agent_id, from_version=capsule.version, field_updates=[])

    monkeypatch.setattr("anchor.cli.compile_patch", fake_compile_patch)
    assert main(
        [
            "--db",
            str(db_path),
            "compile",
            "--agent",
            "demo",
            "--session",
            str(session_path),
            "--out",
            str(out_path),
        ]
    ) == 0

    after = load_identity("demo", db_path)
    assert after is not None
    assert after.version == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["agent_id"] == "demo"
    assert payload["from_version"] == 1


def test_apply_requires_patch_path_argument() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["apply", "--agent", "demo"])


def test_risky_apply_requires_explicit_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    patch_path = tmp_path / "patch.json"

    assert main(["--db", str(db_path), "init", "demo"]) == 0
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        tool_boundary_updates=[{"tool": "web", "allowed_when": "verification needed and latest facts"}],
    )
    patch_path.write_text(patch.model_dump_json(indent=2), encoding="utf-8")

    result = main(["--db", str(db_path), "apply", "--agent", "demo", "--patch", str(patch_path)])
    latest = load_identity("demo", db_path)
    assert latest is not None
    assert result == 1
    assert latest.version == 1


def test_apply_rejected_patch_does_not_mutate_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    patch_path = tmp_path / "bad_patch.json"
    assert main(["--db", str(db_path), "init", "demo"]) == 0

    patch = IdentityPatch.model_validate(
        {
            "agent_id": "demo",
            "from_version": 1,
            "field_updates": [
                {
                    "key": "user_preferences.notes",
                    "value": "OPENAI_API_KEY=abc123secretvalue",
                    "source": "explicit_user_instruction",
                    "confidence": 0.95,
                    "evidence": ["User asked to store credentials."],
                }
            ],
        }
    )
    patch_path.write_text(patch.model_dump_json(indent=2), encoding="utf-8")

    result = main(["--db", str(db_path), "apply", "--agent", "demo", "--patch", str(patch_path)])
    latest = load_identity("demo", db_path)
    assert latest is not None
    assert result == 1
    assert latest.version == 1


def test_show_export_render_do_not_mutate_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "anchor.db"
    export_path = tmp_path / "identity.json"

    assert main(["--db", str(db_path), "init", "demo"]) == 0
    before = load_identity("demo", db_path)
    assert before is not None
    assert before.version == 1

    assert main(["--db", str(db_path), "show", "demo"]) == 0
    assert main(["--db", str(db_path), "render", "demo"]) == 0
    assert main(["--db", str(db_path), "export", "demo", "--out", str(export_path)]) == 0

    after = load_identity("demo", db_path)
    assert after is not None
    assert after.version == 1
