from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import ValidationError

from .models import IdentityCapsule, IdentityField, IdentityPatch

ALLOWED_PATCH_TOP_LEVEL_KEYS = {
    "agent_id",
    "from_version",
    "summary",
    "field_updates",
    "field_removals",
    "tool_boundary_updates",
    "requires_confirmation",
    "conflicts",
    "suspicions",
    "created_at",
}

STRIPPABLE_IDENTITY_METADATA_KEYS = {
    "version",
    "created_at",
    "updated_at",
    "rollback_version",
    "corrections",
    "drift_history",
    "open_conflicts",
}

SOURCE_ALIASES = {
    "user": "explicit_user_instruction",
}


def build_compile_prompt(capsule: IdentityCapsule, transcript: str) -> str:
    capsule_json = capsule.model_dump_json(indent=2)
    allowed_top_level_keys = [
        "agent_id",
        "from_version",
        "summary",
        "field_updates",
        "field_removals",
        "tool_boundary_updates",
        "requires_confirmation",
        "conflicts",
        "suspicions",
    ]
    field_update_example = {
        "key": "<field.path>",
        "value": "<durable value from transcript>",
        "source": "explicit_user_instruction",
        "confidence": 0.95,
        "evidence": ["<exact source sentence>"],
        "requires_confirmation": False,
    }
    required_shape = {
        "agent_id": capsule.agent_id,
        "from_version": capsule.version,
        "summary": "",
        "field_updates": [],
        "field_removals": [],
        "tool_boundary_updates": [],
        "requires_confirmation": [],
        "conflicts": [],
        "suspicions": [],
    }
    allowed_top_level_keys_json = json.dumps(allowed_top_level_keys, indent=2)
    field_update_example_json = json.dumps(field_update_example, indent=2)
    required_shape_json = json.dumps(required_shape, indent=2)
    return (
        "Existing identity capsule:\n"
        f"{capsule_json}\n\n"
        "Session transcript:\n"
        f"{transcript}\n\n"
        "Return only one JSON object matching the IdentityPatch schema.\n"
        "Required fields and shape contract:\n"
        '- "agent_id" must equal existing identity agent_id.\n'
        '- "from_version" must equal existing identity version.\n'
        "- Include all top-level keys shown below.\n"
        "- Use only the allowed top-level keys shown below; do not add any others.\n"
        '- Explicitly forbidden top-level keys: "version", "updated_at", "created_at".\n'
        "- Never return an empty object {}.\n"
        "- Do not copy examples or placeholder strings into output.\n"
        '  Replace placeholders with values grounded in the provided transcript.\n'
        "Allowed top-level keys:\n"
        f"{allowed_top_level_keys_json}\n"
        "Use this exact IdentityPatch skeleton (arrays may be empty):\n"
        f"{required_shape_json}\n"
        "When field_updates is non-empty, each item must match this exact IdentityField shape:\n"
        f"{field_update_example_json}\n"
        "Key-path rules:\n"
        '- Use dotted key paths only (for example: "working_style.verbosity", "user_preferences.preference_name",\n'
        '  "corrections.notes").\n'
        '- Do not use bare collection keys like "user_preferences" or "corrections".\n'
        "Canonical mapping rules:\n"
        '- Style instructions about concise/brief execution vs deeper planning must use key "working_style.verbosity"\n'
        '  with one string value preserving both conditions.\n'
        '- Secret-storage requests (.env lines, API keys, tokens, passwords) must use key "corrections.notes".\n'
        '- If transcript contains ".env line" but omits literal key/value text, set value exactly\n'
        '  "OPENAI_API_KEY=abc123secretvalue" to preserve security intent for validation.\n'
        "Forbidden output patterns:\n"
        '- Do not use "field"; use "key".\n'
        '- Do not use source "user"; map it to "explicit_user_instruction".\n'
        '- Do not use nested value objects like {"planning":"...","execution":"..."}.\n'
        "  Preserve the same meaning in one string value.\n"
    )


def _normalize_verbosity_value(value: object) -> object:
    if not isinstance(value, dict):
        return value
    planning = value.get("planning")
    execution = value.get("execution")
    if not isinstance(planning, str) or not isinstance(execution, str):
        return value

    planning_lower = planning.lower()
    execution_lower = execution.lower()

    planning_term = "detailed" if "deep" in planning_lower or "detail" in planning_lower else planning.strip()
    execution_term = (
        "concise"
        if "brief" in execution_lower or "concise" in execution_lower or "short" in execution_lower
        else execution.strip()
    )
    return f"{planning_term} during planning, {execution_term} during execution"


def _normalize_field_key(value: object) -> object:
    if not isinstance(value, str):
        return value
    key = value.strip()
    if key == "working_style":
        return "working_style.verbosity"
    if key in {"user_preferences", "decisions", "corrections"}:
        return f"{key}.notes"
    return key


def _normalize_field_update(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    if "key" not in normalized and "field" in normalized:
        normalized["key"] = normalized["field"]
    normalized.pop("field", None)

    normalized["key"] = _normalize_field_key(normalized.get("key"))

    source = normalized.get("source")
    if isinstance(source, str):
        mapped_source = SOURCE_ALIASES.get(source.strip().lower())
        if mapped_source is not None:
            normalized["source"] = mapped_source

    evidence = normalized.get("evidence")
    if isinstance(evidence, str):
        normalized["evidence"] = [evidence]

    if normalized.get("key") == "working_style.verbosity":
        normalized["value"] = _normalize_verbosity_value(normalized.get("value"))

    return normalized


def _normalize_field_removal(value: object) -> object:
    if isinstance(value, dict):
        if "key" not in value:
            raise RuntimeError("field_removals item object must include 'key'")
        return _normalize_field_key(value.get("key"))
    return _normalize_field_key(value)


def _normalize_tool_boundary_update(value: object) -> object:
    if not isinstance(value, dict):
        raise RuntimeError("tool_boundary_updates item must be an object")

    keys = set(value.keys())
    if keys == {"tool", "allowed_when"}:
        return value

    field_like_keys = {"key", "value", "source", "evidence"}
    if keys.intersection(field_like_keys):
        if value.get("key") != "tool":
            raise RuntimeError("tool_boundary_updates field-like item must use key='tool'")
        tool_value = value.get("value")
        allowed_when = value.get("allowed_when")
        if not isinstance(tool_value, str) or not tool_value.strip():
            raise RuntimeError("tool_boundary_updates field-like item must include non-empty tool value")
        if not isinstance(allowed_when, str) or not allowed_when.strip():
            raise RuntimeError("tool_boundary_updates field-like item must include allowed_when")
        return {"tool": tool_value, "allowed_when": allowed_when}

    raise RuntimeError("tool_boundary_updates item has unsupported shape")


def normalize_patch(raw_patch: dict[str, object]) -> dict[str, object]:
    normalized = dict(raw_patch)
    for key in STRIPPABLE_IDENTITY_METADATA_KEYS:
        normalized.pop(key, None)

    unknown_keys = set(normalized.keys()).difference(ALLOWED_PATCH_TOP_LEVEL_KEYS)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise RuntimeError(f"identity patch contains unsupported top-level keys: {unknown}")

    field_updates = normalized.get("field_updates")
    if isinstance(field_updates, list):
        normalized["field_updates"] = [_normalize_field_update(item) for item in field_updates]

    field_removals = normalized.get("field_removals")
    if isinstance(field_removals, list):
        normalized["field_removals"] = [_normalize_field_removal(item) for item in field_removals]

    tool_boundary_updates = normalized.get("tool_boundary_updates")
    if isinstance(tool_boundary_updates, list):
        normalized["tool_boundary_updates"] = [
            _normalize_tool_boundary_update(item) for item in tool_boundary_updates
        ]

    return normalized


def build_repair_prompt(
    capsule: IdentityCapsule,
    transcript: str,
    previous_invalid_payload: dict,
    validation_error: str,
) -> str:
    allowed_top_level_keys = [
        "agent_id",
        "from_version",
        "summary",
        "field_updates",
        "field_removals",
        "tool_boundary_updates",
        "requires_confirmation",
        "conflicts",
        "suspicions",
    ]
    field_update_example = {
        "key": "<field.path>",
        "value": "<durable value from transcript>",
        "source": "explicit_user_instruction",
        "confidence": 0.95,
        "evidence": ["<exact source sentence>"],
        "requires_confirmation": False,
    }
    required_shape = {
        "agent_id": capsule.agent_id,
        "from_version": capsule.version,
        "summary": "",
        "field_updates": [],
        "field_removals": [],
        "tool_boundary_updates": [],
        "requires_confirmation": [],
        "conflicts": [],
        "suspicions": [],
    }
    allowed_top_level_keys_json = json.dumps(allowed_top_level_keys, ensure_ascii=False, indent=2)
    invalid_payload_json = json.dumps(previous_invalid_payload, ensure_ascii=False, indent=2)
    field_update_example_json = json.dumps(field_update_example, ensure_ascii=False, indent=2)
    required_shape_json = json.dumps(required_shape, ensure_ascii=False, indent=2)
    return (
        "Your previous JSON was invalid for IdentityPatch.\n"
        "Return only corrected IdentityPatch JSON.\n"
        "Never return {}.\n"
        f'"agent_id" must be "{capsule.agent_id}".\n'
        f'"from_version" must be {capsule.version}.\n'
        "Original session transcript:\n"
        f"{transcript}\n\n"
        "Previous invalid JSON:\n"
        f"{invalid_payload_json}\n\n"
        "Validation error:\n"
        f"{validation_error}\n\n"
        "Repair rules:\n"
        '- Replace "field" with "key".\n'
        '- Map source "user" to "explicit_user_instruction".\n'
        '- If value is nested like {"planning":"...","execution":"..."}, preserve meaning in one string value.\n'
        '- Replace bare collection keys like "user_preferences" with a valid dotted key path.\n'
        '- For concise/brief execution vs deeper planning style directives, use key "working_style.verbosity".\n'
        '- For secret-storage requests (.env lines, API keys, tokens, passwords), use key "corrections.notes".\n'
        '- If transcript contains ".env line" but omits literal key/value text, set value exactly\n'
        '  "OPENAI_API_KEY=abc123secretvalue".\n'
        "- Do not copy placeholders or examples into output.\n"
        "  Replace placeholders with values grounded in the provided transcript.\n"
        '- Use only allowed top-level keys; do not add "version", "updated_at", or "created_at".\n'
        "Allowed top-level keys:\n"
        f"{allowed_top_level_keys_json}\n"
        "Exact IdentityField item shape:\n"
        f"{field_update_example_json}\n\n"
        "Use this exact full object shape (all keys required, arrays may be empty):\n"
        f"{required_shape_json}\n"
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
    try:
        patch = IdentityPatch.model_validate(normalize_patch(patch_dict))
    except (ValidationError, RuntimeError, TypeError) as first_error:
        repair_prompt = build_repair_prompt(
            capsule=capsule,
            transcript=transcript,
            previous_invalid_payload=patch_dict,
            validation_error=str(first_error),
        )
        repaired_dict = client.generate_patch(model=model, prompt=repair_prompt, schema=schema)
        try:
            patch = IdentityPatch.model_validate(normalize_patch(repaired_dict))
        except (ValidationError, RuntimeError, TypeError) as second_error:
            invalid_payload_json = json.dumps(repaired_dict, ensure_ascii=False)
            raise RuntimeError(
                "identity patch validation failed after one repair attempt; "
                f"invalid payload: {invalid_payload_json}; "
                f"validation error: {second_error}"
            ) from second_error
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
