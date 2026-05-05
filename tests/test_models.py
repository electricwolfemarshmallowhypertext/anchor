from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.models import IdentityCapsule, IdentityField, IdentityPatch


def test_identity_field_requires_non_empty_key() -> None:
    with pytest.raises(ValidationError):
        IdentityField(
            key="  ",
            value=True,
            source="explicit_user_instruction",
            confidence=0.9,
        )


def test_identity_field_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        IdentityField(
            key="user_preferences.prefers_laconic_technical_answers",
            value=True,
            source="explicit_user_instruction",
            confidence=1.1,
        )


def test_identity_capsule_defaults() -> None:
    capsule = IdentityCapsule(agent_id="demo")
    assert capsule.version == 1
    assert capsule.rollback_version == 1
    assert capsule.user_preferences == []


def test_identity_patch_round_trip() -> None:
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        summary="set verbosity",
        field_updates=[
            IdentityField(
                key="working_style.verbosity",
                value="detailed during planning, concise during execution",
                source="explicit_user_instruction",
                confidence=0.99,
            )
        ],
    )
    loaded = IdentityPatch.model_validate_json(patch.model_dump_json())
    assert loaded.agent_id == "demo"
    assert loaded.field_updates[0].key == "working_style.verbosity"


def test_identity_patch_rejects_unknown_operations() -> None:
    with pytest.raises(ValidationError):
        IdentityPatch.model_validate(
            {
                "agent_id": "demo",
                "from_version": 1,
                "field_updates": [],
                "unknown_operation": {"drop_all": True},
            }
        )


def test_identity_field_rejects_malformed_paths() -> None:
    with pytest.raises(ValidationError):
        IdentityField(
            key="preferences..bad",
            value=True,
            source="explicit_user_instruction",
            confidence=0.9,
        )


def test_identity_patch_rejects_invalid_removal_paths() -> None:
    with pytest.raises(ValidationError):
        IdentityPatch(
            agent_id="demo",
            from_version=1,
            field_removals=["invalid/path"],
        )
