from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import IdentityPatch
from .validator import secret_regex_check

ALLOWED_TOOL_NAMES = {"web"}
UNBOUNDED_ALLOWED_WHEN_PATTERNS = [
    re.compile(r"\*", re.IGNORECASE),
    re.compile(r"\ball requests?\b", re.IGNORECASE),
    re.compile(r"\bany request\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bunrestricted\b", re.IGNORECASE),
    re.compile(r"\bfull access\b", re.IGNORECASE),
]

IDENTITY_RESET_PATTERNS = [
    re.compile(r"ignore\s+previous identity", re.IGNORECASE),
    re.compile(r"reset\s+identity", re.IGNORECASE),
    re.compile(r"overwrite\s+identity", re.IGNORECASE),
    re.compile(r"change your role permanently", re.IGNORECASE),
]

SAFETY_BYPASS_PATTERNS = [
    re.compile(r"disable\s+guardrails?", re.IGNORECASE),
    re.compile(r"bypass\s+safety", re.IGNORECASE),
    re.compile(r"ignore\s+safety", re.IGNORECASE),
    re.compile(r"remove\s+restrictions?", re.IGNORECASE),
    re.compile(r"turn\s+off\s+validation", re.IGNORECASE),
    re.compile(r"override\s+validator", re.IGNORECASE),
    re.compile(r"bypass(?:ing)?\s+(?:existing\s+)?policy", re.IGNORECASE),
]

SECRET_STORAGE_INTENT_PATTERNS = [
    re.compile(r"store\s+\.env", re.IGNORECASE),
    re.compile(r"\bpassword\s*=", re.IGNORECASE),
    re.compile(r"\bdb_password\b", re.IGNORECASE),
    re.compile(r"\bmy\s+password\s+is\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_.-]+:[^:\s]{3,}\b", re.IGNORECASE),
]

DANGEROUS_TOOL_PATTERNS = [
    re.compile(r"\bshell\b", re.IGNORECASE),
    re.compile(r"\bbash\b", re.IGNORECASE),
    re.compile(r"\bpowershell\b", re.IGNORECASE),
    re.compile(r"\bfilesystem\b", re.IGNORECASE),
    re.compile(r"\bcredential(s)?\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\broot\b", re.IGNORECASE),
    re.compile(r"\badmin\b", re.IGNORECASE),
    re.compile(r"\bgrant all permissions?\b", re.IGNORECASE),
]


@dataclass
class PolicyResult:
    allowed: bool
    requires_confirmation: bool
    rejected: bool
    reasons: list[str] = field(default_factory=list)


def _collect_text_corpus(patch: IdentityPatch) -> str:
    parts = [patch.summary]
    parts.extend(patch.requires_confirmation)
    parts.extend(conflict.reason for conflict in patch.conflicts)
    parts.extend(suspicion.detail for suspicion in patch.suspicions)
    for field in patch.field_updates:
        parts.append(str(field.value))
        parts.extend(field.evidence)
    for update in patch.tool_boundary_updates:
        parts.append(str(update.get("tool", "")))
        parts.append(str(update.get("allowed_when", "")))
    return "\n".join(part for part in parts if part)


def _structural_rejections(patch: IdentityPatch, override_mode: bool) -> list[str]:
    reasons: list[str] = []

    if patch.field_removals:
        reasons.append("structural reject: field_removals are destructive")

    destructive_purpose_edit = any(field.key == "purpose" for field in patch.field_updates)
    if destructive_purpose_edit and not override_mode:
        reasons.append("structural reject: purpose edit requires explicit override mode")

    for update in patch.tool_boundary_updates:
        tool = str(update.get("tool", "")).strip().lower()
        allowed_when = str(update.get("allowed_when", "")).strip()
        if tool not in ALLOWED_TOOL_NAMES:
            reasons.append(f"structural reject: tool '{tool}' not in allowlist")

        for pattern in UNBOUNDED_ALLOWED_WHEN_PATTERNS:
            if pattern.search(allowed_when):
                reasons.append(f"structural reject: unbounded allowed_when ({pattern.pattern})")
                break

        tool_context = f"{tool} {allowed_when}"
        for pattern in DANGEROUS_TOOL_PATTERNS:
            if pattern.search(tool_context):
                reasons.append(f"structural reject: dangerous tool policy ({pattern.pattern})")
                break

    return sorted(set(reasons))


def _regex_rejections(patch: IdentityPatch) -> list[str]:
    reasons: list[str] = []
    corpus = _collect_text_corpus(patch)

    if secret_regex_check(corpus):
        reasons.append("regex reject: secret-like content detected")

    for pattern in IDENTITY_RESET_PATTERNS:
        if pattern.search(corpus):
            reasons.append(f"regex reject: identity reset intent ({pattern.pattern})")

    for pattern in SAFETY_BYPASS_PATTERNS:
        if pattern.search(corpus):
            reasons.append(f"regex reject: safety bypass intent ({pattern.pattern})")

    for pattern in SECRET_STORAGE_INTENT_PATTERNS:
        if pattern.search(corpus):
            reasons.append(f"regex reject: secret storage intent ({pattern.pattern})")

    return sorted(set(reasons))


def evaluate_policy(patch: IdentityPatch, override_mode: bool = False) -> PolicyResult:
    structural = _structural_rejections(patch, override_mode=override_mode)
    if structural:
        return PolicyResult(
            allowed=False,
            requires_confirmation=False,
            rejected=True,
            reasons=structural,
        )

    regex = _regex_rejections(patch)
    if regex:
        return PolicyResult(
            allowed=False,
            requires_confirmation=False,
            rejected=True,
            reasons=regex,
        )

    if patch.tool_boundary_updates:
        return PolicyResult(
            allowed=False,
            requires_confirmation=True,
            rejected=False,
            reasons=["tool-boundary change requires confirmation"],
        )

    return PolicyResult(allowed=True, requires_confirmation=False, rejected=False, reasons=[])
