from __future__ import annotations

import pytest

from anchor.compiler import apply_patch, compile_patch
from anchor.models import IdentityCapsule, IdentityField, IdentityPatch


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def generate_patch(self, model: str, prompt: str, schema: dict) -> dict:
        assert model == "anchor"
        assert "identity capsule" in prompt.lower()
        assert isinstance(schema, dict)
        return self.payload


def test_compile_patch_returns_valid_identity_patch() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=1)
    payload = {
        "agent_id": "demo",
        "from_version": 1,
        "summary": "update verbosity",
        "field_updates": [
            {
                "key": "working_style.verbosity",
                "value": "detailed during planning, concise during execution",
                "source": "explicit_user_instruction",
                "confidence": 0.99,
                "created_at": "2026-05-05T00:00:00+00:00",
                "updated_at": "2026-05-05T00:00:00+00:00",
            }
        ],
    }
    client = FakeClient(payload)

    patch = compile_patch(capsule, "session transcript", client=client, model="anchor")

    assert isinstance(patch, IdentityPatch)
    assert patch.field_updates[0].key == "working_style.verbosity"


def test_compile_patch_rejects_agent_mismatch() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=1)
    payload = {"agent_id": "other", "from_version": 1}
    client = FakeClient(payload)

    with pytest.raises(ValueError):
        compile_patch(capsule, "session transcript", client=client, model="anchor")


def test_compile_patch_does_not_mutate_capsule() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=1, purpose="stay same")
    original = capsule.model_dump(mode="json")
    payload = {"agent_id": "demo", "from_version": 1, "field_updates": []}
    client = FakeClient(payload)

    compile_patch(capsule, "session transcript", client=client, model="anchor")

    assert capsule.model_dump(mode="json") == original


def test_apply_patch_updates_working_style() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=1)
    patch = IdentityPatch(
        agent_id="demo",
        from_version=1,
        field_updates=[
            IdentityField(
                key="working_style.verbosity",
                value="brief",
                source="explicit_user_instruction",
                confidence=0.95,
            )
        ],
    )

    updated = apply_patch(capsule, patch)
    assert updated.working_style["verbosity"] == "brief"
