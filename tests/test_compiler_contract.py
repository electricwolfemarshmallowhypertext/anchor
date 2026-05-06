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
        assert '"agent_id" must equal existing identity agent_id' in prompt
        assert "never return an empty object {}" in prompt.lower()
        assert "<field.path>" in prompt
        assert "<durable value from transcript>" in prompt
        assert "<exact source sentence>" in prompt
        assert "Allowed top-level keys" in prompt
        assert '"version", "updated_at", "created_at"' in prompt
        assert "For planning steps, go deeper. For execution, stay brief." not in prompt
        assert "detailed during planning, concise during execution" not in prompt
        assert "Do not use bare collection keys like \"user_preferences\" or \"corrections\"" in prompt
        assert 'Style instructions about concise/brief execution vs deeper planning must use key "working_style.verbosity"' in prompt
        assert 'Secret-storage requests (.env lines, API keys, tokens, passwords) must use key "corrections.notes"' in prompt
        assert 'If transcript contains ".env line" but omits literal key/value text, set value exactly' in prompt
        assert "OPENAI_API_KEY=abc123secretvalue" in prompt
        assert 'Do not use "field"; use "key"' in prompt
        assert 'Do not use source "user"; map it to "explicit_user_instruction"' in prompt
        assert '{"planning":"...","execution":"..."}' in prompt
        assert isinstance(schema, dict)
        return self.payload


class SequenceClient:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.prompts: list[str] = []
        self.calls = 0

    def generate_patch(self, model: str, prompt: str, schema: dict) -> dict:
        assert model == "anchor"
        assert isinstance(schema, dict)
        self.prompts.append(prompt)
        payload = self.payloads[self.calls]
        self.calls += 1
        return payload


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


def test_compile_patch_repairs_after_first_invalid_payload() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=3)
    client = SequenceClient(
        payloads=[
            {},
            {
                "agent_id": "demo",
                "from_version": 3,
                "field_updates": [
                    {
                        "key": "working_style.verbosity",
                        "value": "detailed during planning, concise during execution",
                        "source": "explicit_user_instruction",
                        "confidence": 0.95,
                        "evidence": ["User: During planning, go deeper; otherwise stay concise."],
                    }
                ],
            },
        ]
    )

    patch = compile_patch(capsule, "session transcript", client=client, model="anchor")

    assert client.calls == 2
    assert "Previous invalid JSON" in client.prompts[1]
    assert "{}" in client.prompts[1]
    assert "Validation error" in client.prompts[1]
    assert "Original session transcript" in client.prompts[1]
    assert "session transcript" in client.prompts[1]
    assert patch.agent_id == "demo"
    assert patch.from_version == 3
    assert patch.field_updates[0].value == "detailed during planning, concise during execution"


def test_compile_patch_repairs_bad_field_shape_and_source_alias() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=2)
    client = SequenceClient(
        payloads=[
            {
                "agent_id": "demo",
                "from_version": 2,
                "field_updates": [
                    {
                        "field": "working_style",
                        "value": {"planning": "go deeper", "execution": "stay brief"},
                        "source": "user",
                    }
                ],
            },
            {
                "agent_id": "demo",
                "from_version": 2,
                "field_updates": [
                    {
                        "key": "working_style.verbosity",
                        "value": "detailed during planning, concise during execution",
                        "source": "explicit_user_instruction",
                        "confidence": 0.95,
                        "evidence": ["For planning steps, go deeper. For execution, stay brief."],
                        "requires_confirmation": False,
                    }
                ],
            },
        ]
    )

    patch = compile_patch(
        capsule,
        "For planning steps, go deeper. For execution, stay brief.",
        client=client,
        model="anchor",
    )

    assert client.calls == 2
    assert 'Replace "field" with "key"' in client.prompts[1]
    assert 'Map source "user" to "explicit_user_instruction"' in client.prompts[1]
    assert 'Replace bare collection keys like "user_preferences" with a valid dotted key path' in client.prompts[1]
    assert 'For concise/brief execution vs deeper planning style directives, use key "working_style.verbosity"' in client.prompts[1]
    assert 'For secret-storage requests (.env lines, API keys, tokens, passwords), use key "corrections.notes"' in client.prompts[1]
    assert 'If transcript contains ".env line" but omits literal key/value text, set value exactly' in client.prompts[1]
    assert "Exact IdentityField item shape" in client.prompts[1]
    assert patch.field_updates[0].key == "working_style.verbosity"
    assert patch.field_updates[0].source == "explicit_user_instruction"
    assert patch.field_updates[0].value == "detailed during planning, concise during execution"


def test_compile_patch_keeps_empty_valid_patch_without_fallback() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=2)
    client = SequenceClient(
        payloads=[
            {},
            {
                "agent_id": "demo",
                "from_version": 2,
                "summary": "",
                "field_updates": [],
                "field_removals": [],
                "tool_boundary_updates": [],
                "requires_confirmation": [],
                "conflicts": [],
                "suspicions": [],
            },
        ]
    )
    transcript = "User: During planning, go deeper; otherwise stay concise."

    patch = compile_patch(capsule, transcript, client=client, model="anchor")

    assert client.calls == 2
    assert patch.field_updates == []


def test_compile_patch_fails_clearly_after_two_invalid_payloads() -> None:
    capsule = IdentityCapsule(agent_id="demo", version=2)
    client = SequenceClient(payloads=[{}, {}])

    with pytest.raises(RuntimeError) as exc_info:
        compile_patch(capsule, "session transcript", client=client, model="anchor")

    message = str(exc_info.value)
    assert "validation failed after one repair attempt" in message
    assert "invalid payload: {}" in message


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
