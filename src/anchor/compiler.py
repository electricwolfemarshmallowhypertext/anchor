from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import IdentityCapsule, IdentityField, IdentityPatch


def build_compile_prompt(capsule: IdentityCapsule, transcript: str) -> str:
    capsule_json = capsule.model_dump_json(indent=2)
    return (
        "Existing identity capsule:\n"
        f"{capsule_json}\n\n"
        "Session transcript:\n"
        f"{transcript}\n\n"
        "Return only a JSON object matching the IdentityPatch schema."
    )


def compile_patch(
    capsule: IdentityCapsule,
    transcript: str,
    client,
    model: str = "anchor",
) -> IdentityPatch:
    prompt = build_compile_prompt(capsule, transcript)
    schema = IdentityPatch.model_json_schema()
    patch_dict = client.generate_patch(model=model, prompt=prompt, schema=schema)
    patch = IdentityPatch.model_validate(patch_dict)
    if patch.agent_id != capsule.agent_id:
        raise ValueError("patch agent_id does not match capsule agent_id")
    return patch


def _upsert_field(target: list[IdentityField], field: IdentityField) -> None:
    for index, item in enumerate(target):
        if item.key == field.key:
            target[index] = field
            return
    target.append(field)


def _remove_field(target: list[IdentityField], key: str) -> None:
    remaining = [item for item in target if item.key != key]
    target.clear()
    target.extend(remaining)


def _get_target_collection(capsule: IdentityCapsule, key: str) -> list[IdentityField]:
    if key.startswith("user_preferences."):
        return capsule.user_preferences
    if key.startswith("decisions."):
        return capsule.decisions
    return capsule.corrections


def apply_patch(capsule: IdentityCapsule, patch: IdentityPatch) -> IdentityCapsule:
    if patch.agent_id != capsule.agent_id:
        raise ValueError("patch agent_id does not match target capsule")
    if patch.from_version != capsule.version:
        raise ValueError(
            f"patch from_version {patch.from_version} does not match current version {capsule.version}"
        )

    updated = capsule.model_copy(deep=True)

    for field in patch.field_updates:
        if field.key == "purpose":
            updated.purpose = str(field.value)
            continue
        if field.key.startswith("working_style."):
            style_key = field.key.split(".", 1)[1]
            updated.working_style[style_key] = field.value
            continue
        _upsert_field(_get_target_collection(updated, field.key), field)

    for key in patch.field_removals:
        if key == "purpose":
            updated.purpose = ""
            continue
        if key.startswith("working_style."):
            style_key = key.split(".", 1)[1]
            updated.working_style.pop(style_key, None)
            continue
        _remove_field(_get_target_collection(updated, key), key)

    if patch.tool_boundary_updates:
        updated.tool_boundaries = patch.tool_boundary_updates

    updated.open_conflicts = patch.conflicts
    updated.recent_identity_changes.append(
        f"{datetime.now(timezone.utc).isoformat()}:{patch.summary or 'patch_applied'}"
    )
    updated.updated_at = datetime.now(timezone.utc)
    return updated


def patch_to_json(patch: IdentityPatch) -> str:
    return json.dumps(patch.model_dump(mode="json"), indent=2)
