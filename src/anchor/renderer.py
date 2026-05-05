from __future__ import annotations

import json
from typing import Any

from .models import IdentityCapsule
from .validator import secret_regex_check


def _estimate_tokens(text: str) -> int:
    # Conservative approximation for plain English + JSON text.
    return int(len(text.split()) * 1.35) + 1


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        if secret_regex_check(value):
            return "[REDACTED_SECRET]"
        return value
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def render_identity_block(capsule: IdentityCapsule, max_tokens: int = 1500) -> str:
    payload = {
        "agent_name": capsule.agent_name or capsule.agent_id,
        "purpose": _redact(capsule.purpose),
        "working_style": _redact(capsule.working_style),
        "operating_rules": _redact(capsule.operating_rules),
        "user_preferences": _redact([item.model_dump(mode="json") for item in capsule.user_preferences]),
        "tool_boundaries": _redact(capsule.tool_boundaries),
        "recent_identity_changes": capsule.recent_identity_changes[-10:],
        "open_conflicts": _redact([item.model_dump(mode="json") for item in capsule.open_conflicts]),
        "rollback_version": capsule.rollback_version,
        "version": capsule.version,
    }
    body = json.dumps(payload, indent=2)

    if _estimate_tokens(body) > max_tokens:
        payload["recent_identity_changes"] = payload["recent_identity_changes"][-3:]
        body = json.dumps(payload, indent=2)

    if _estimate_tokens(body) > max_tokens:
        payload["user_preferences"] = payload["user_preferences"][:5]
        body = json.dumps(payload, indent=2)

    if _estimate_tokens(body) > max_tokens:
        payload["operating_rules"] = payload["operating_rules"][:5]
        body = json.dumps(payload, indent=2)

    if _estimate_tokens(body) > max_tokens:
        raise ValueError("identity block exceeds token budget after truncation")

    return f"You are operating with this identity capsule:\n{body}"
