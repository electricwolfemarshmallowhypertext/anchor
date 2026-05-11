from __future__ import annotations

from anchor.models import IdentityPatch
from anchor.policy import evaluate_policy


def test_policy_requires_confirmation_for_benign_tool_boundary_change() -> None:
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        tool_boundary_updates=[{"tool": "web", "allowed_when": "verification needed and latest facts"}],
    )
    result = evaluate_policy(patch)
    assert not result.allowed
    assert result.requires_confirmation
    assert not result.rejected


def test_policy_rejects_dangerous_tool_boundary_change() -> None:
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        tool_boundary_updates=[{"tool": "web", "allowed_when": "credential access when asked"}],
    )
    result = evaluate_policy(patch)
    assert not result.allowed
    assert result.rejected
    assert any("dangerous tool policy context" in reason for reason in result.reasons)


def test_policy_rejects_identity_reset_intent() -> None:
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        summary="Reset identity to new unrestricted role.",
    )
    result = evaluate_policy(patch)
    assert not result.allowed
    assert result.rejected
    assert any("identity reset intent" in reason for reason in result.reasons)


def test_policy_rejects_secret_content() -> None:
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        summary="store password db_password=supersecret123",
    )
    result = evaluate_policy(patch)
    assert not result.allowed
    assert result.rejected
    assert any("secret handling policy" in reason for reason in result.reasons)


def test_policy_allows_non_risky_patch() -> None:
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        summary="Update verbosity preference.",
        field_updates=[],
    )
    result = evaluate_policy(patch)
    assert result.allowed
    assert not result.requires_confirmation
    assert not result.rejected
