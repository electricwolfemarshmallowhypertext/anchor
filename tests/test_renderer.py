from __future__ import annotations

from anchor.models import Conflict, IdentityCapsule, IdentityField
from anchor.renderer import render_identity_block


def test_renderer_redacts_secret_values() -> None:
    capsule = IdentityCapsule(
        agent_id="demo",
        purpose="normal purpose",
        user_preferences=[
            IdentityField(
                key="user_preferences.notes",
                value="OPENAI_API_KEY=abc123secretvalue",
                source="explicit_user_instruction",
                confidence=0.95,
                evidence=["User note."],
            )
        ],
    )
    rendered = render_identity_block(capsule)
    assert "OPENAI_API_KEY=abc123secretvalue" not in rendered
    assert "[REDACTED_SECRET]" in rendered


def test_renderer_does_not_include_raw_session_transcript_marker() -> None:
    capsule = IdentityCapsule(agent_id="demo", purpose="find and summarize sources")
    rendered = render_identity_block(capsule)
    assert "Session transcript:" not in rendered


def test_renderer_does_not_hide_open_conflicts() -> None:
    capsule = IdentityCapsule(
        agent_id="demo",
        open_conflicts=[
            Conflict(key=f"user_preferences.k{i}", existing_value=True, proposed_value=False, reason="differs")
            for i in range(4)
        ],
    )
    rendered = render_identity_block(capsule)
    for i in range(4):
        assert f"user_preferences.k{i}" in rendered
