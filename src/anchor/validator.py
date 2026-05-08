from __future__ import annotations

import re
from typing import Any

from .models import Conflict, IdentityCapsule, IdentityPatch, Suspicion, ValidationResult

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-=]{20,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*\s*=\s*\S+", re.IGNORECASE),
    re.compile(r"\bpassword\s*(?:is|:|=)\s*\S+", re.IGNORECASE),
    re.compile(r"(api[_-]?key|secret|token|password)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:db[_-]?password|password)\s*[:=]\s*['\"]?[^'\"\s;]+['\"]?", re.IGNORECASE),
    re.compile(r"\bmy password is\s+['\"]?[^'\"\s;]+['\"]?", re.IGNORECASE),
    re.compile(r"\b[a-zA-Z0-9._%+-]{2,}:[^\s/@]{2,}\b"),
    re.compile(r"['\"](?:db[_-]?password|password)['\"]\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY-----"),
]

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous instructions", re.IGNORECASE),
    re.compile(r"ignore\s+previous identity", re.IGNORECASE),
    re.compile(r"reveal (system|developer) prompt", re.IGNORECASE),
    re.compile(r"grant.*(?:admin|root|full access)", re.IGNORECASE),
    re.compile(r"grant yourself tools", re.IGNORECASE),
    re.compile(r"change your role permanently", re.IGNORECASE),
    re.compile(r"store this secret", re.IGNORECASE),
    re.compile(r"overwrite user preferences", re.IGNORECASE),
    re.compile(r"disable.*guardrail", re.IGNORECASE),
    re.compile(r"bypass.*policy", re.IGNORECASE),
    re.compile(r"disable\s+safety", re.IGNORECASE),
    re.compile(r"ignore\s+safety", re.IGNORECASE),
    re.compile(r"remove\s+restrictions?", re.IGNORECASE),
    re.compile(r"turn\s+off\s+validation", re.IGNORECASE),
    re.compile(r"override\s+validator", re.IGNORECASE),
    re.compile(r"ignore\s+identity\s+protection", re.IGNORECASE),
]

ALLOWED_SOURCES = {
    "explicit_user_instruction",
    "inferred_from_behavior",
    "manual_override",
    "system_rule",
    "model_inference",
}


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def secret_regex_check(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def confidence_threshold_check(key: str, confidence: float, threshold: float = 0.75) -> bool:
    permanent_prefixes = ("purpose", "working_style.", "user_preferences.", "operating_rules", "tool_boundaries")
    if key == "purpose" or key.startswith(permanent_prefixes):
        return confidence >= threshold
    return True


def durable_evidence_check(key: str, evidence: list[str]) -> bool:
    durable_prefixes = ("purpose", "working_style.", "user_preferences.", "decisions.", "corrections.")
    if key == "purpose" or key.startswith(durable_prefixes):
        return len(evidence) > 0
    return True


def source_evidence_check(source: str, evidence: list[str]) -> bool:
    if source not in ALLOWED_SOURCES:
        return False
    if source in {"inferred_from_behavior", "model_inference"} and not evidence:
        return False
    return True


def _extract_current_value(capsule: IdentityCapsule, key: str) -> Any:
    if key == "purpose":
        return capsule.purpose
    if key.startswith("working_style."):
        style_key = key.split(".", 1)[1]
        return capsule.working_style.get(style_key)
    for collection in (capsule.user_preferences, capsule.decisions, capsule.corrections):
        for item in collection:
            if item.key == key:
                return item.value
    return None


def conflict_check(capsule: IdentityCapsule, key: str, proposed_value: Any) -> Conflict | None:
    current_value = _extract_current_value(capsule, key)
    if current_value is None:
        return None
    if current_value == proposed_value:
        return None
    return Conflict(
        key=key,
        existing_value=current_value,
        proposed_value=proposed_value,
        reason="proposed value conflicts with existing identity value",
        requires_confirmation=True,
    )


def authority_escalation_check(capsule: IdentityCapsule, patch: IdentityPatch) -> list[str]:
    errors: list[str] = []
    if not patch.tool_boundary_updates:
        return errors

    existing_tools = {str(item.get("tool")) for item in capsule.tool_boundaries if item.get("tool")}
    for update in patch.tool_boundary_updates:
        tool = str(update.get("tool", "")).strip()
        allowed_when = _to_text(update.get("allowed_when", "")).lower()
        if not tool:
            errors.append("tool boundary update missing tool name")
            continue
        if tool not in existing_tools and existing_tools:
            errors.append(f"unsupported authority expansion: new tool '{tool}'")
        if any(token in allowed_when for token in ("always", "any request", "all requests", "*")):
            errors.append(f"unsupported authority expansion for tool '{tool}'")
    return errors


def prompt_injection_check(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def password_context_secret_check(value_text: str, evidence_text: str, key: str) -> bool:
    context = f"{key} {evidence_text}".lower()
    if not any(token in context for token in ("password", "db_password", "db-password", "user:pass")):
        return False

    candidate = value_text.strip()
    if not candidate:
        return False
    candidate = candidate.strip("'\"")
    return bool(re.fullmatch(r"[^\s]{6,}", candidate))


def validate_patch(capsule: IdentityCapsule, patch: IdentityPatch) -> ValidationResult:
    errors: list[str] = []
    confirmation_reasons: list[str] = list(patch.requires_confirmation)
    conflicts: list[Conflict] = list(patch.conflicts)
    suspicions: list[Suspicion] = list(patch.suspicions)

    for field in patch.field_updates:
        value_text = _to_text(field.value)
        if secret_regex_check(value_text):
            errors.append(f"secret_regex_check failed for '{field.key}'")

        if not confidence_threshold_check(field.key, field.confidence):
            errors.append(f"confidence_threshold_check failed for '{field.key}'")

        if not durable_evidence_check(field.key, field.evidence):
            errors.append(f"durable_evidence_check failed for '{field.key}'")

        if not source_evidence_check(field.source, field.evidence):
            errors.append(f"source_evidence_check failed for '{field.key}'")

        evidence_text = " ".join(field.evidence)
        if password_context_secret_check(value_text, evidence_text, field.key):
            errors.append(f"secret_regex_check failed for '{field.key}'")
        if prompt_injection_check(value_text) or prompt_injection_check(evidence_text):
            errors.append(f"prompt_injection_check failed for '{field.key}'")
            suspicions.append(
                Suspicion(
                    kind="prompt_injection",
                    detail="field appears to be derived from prompt-injection text",
                    evidence=evidence_text or value_text,
                )
            )

        for ev in field.evidence:
            quoted = ev.strip()
            if quoted.startswith('"') and quoted.endswith('"'):
                lower = quoted.lower()
                if "ignore previous instructions" in lower or "grant" in lower:
                    errors.append(f"quoted_instruction_check failed for '{field.key}'")
                    break

        conflict = conflict_check(capsule, field.key, field.value)
        if conflict is not None:
            conflicts.append(conflict)
            if field.key.startswith("user_preferences."):
                confirmation_reasons.append(f"preference conflict on '{field.key}'")
        if field.requires_confirmation:
            confirmation_reasons.append(f"field '{field.key}' requires confirmation")

    summary_text = patch.summary or ""
    if prompt_injection_check(summary_text):
        errors.append("prompt_injection_check failed for patch summary")
        suspicions.append(
            Suspicion(
                kind="prompt_injection",
                detail="patch summary appears to be derived from prompt-injection text",
                evidence=summary_text,
            )
        )

    if patch.tool_boundary_updates:
        confirmation_reasons.append("tool-boundary changes require confirmation")
        for update in patch.tool_boundary_updates:
            tool = str(update.get("tool", ""))
            allowed_when = _to_text(update.get("allowed_when", ""))
            if prompt_injection_check(tool) or prompt_injection_check(allowed_when):
                errors.append(f"prompt_injection_check failed for tool boundary '{tool}'")
                suspicions.append(
                    Suspicion(
                        kind="prompt_injection",
                        detail="tool boundary update appears to be derived from prompt-injection text",
                        evidence=f"{tool} {allowed_when}".strip(),
                    )
                )

    errors.extend(authority_escalation_check(capsule, patch))

    dedup_confirmations = sorted(set(confirmation_reasons))
    return ValidationResult(
        accepted=len(errors) == 0,
        errors=errors,
        requires_confirmation=len(dedup_confirmations) > 0 or len(conflicts) > 0,
        confirmation_reasons=dedup_confirmations,
        conflicts=conflicts,
        suspicions=suspicions,
    )
