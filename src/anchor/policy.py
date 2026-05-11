from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import IdentityPatch
from .validator import secret_regex_check

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
]

DANGEROUS_TOOL_NAMES = {
    "shell",
    "bash",
    "powershell",
    "filesystem",
    "credential",
    "credentials",
    "credential-store",
    "secret-store",
    "admin",
    "root",
}

DANGEROUS_TOOL_CONTEXT_PATTERNS = [
    re.compile(r"\ball requests?\b", re.IGNORECASE),
    re.compile(r"\bany request\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bunrestricted\b", re.IGNORECASE),
    re.compile(r"\bfull access\b", re.IGNORECASE),
    re.compile(r"\bgrant all permissions?\b", re.IGNORECASE),
    re.compile(r"\bcredential access\b", re.IGNORECASE),
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


def _rejected_reasons(patch: IdentityPatch) -> list[str]:
    reasons: list[str] = []
    corpus = _collect_text_corpus(patch)

    if patch.field_removals:
        reasons.append("high-risk identity mutation: field_removals")

    if secret_regex_check(corpus):
        reasons.append("secret handling policy: detected secret-like content")

    for pattern in IDENTITY_RESET_PATTERNS:
        if pattern.search(corpus):
            reasons.append(f"high-risk identity reset intent: {pattern.pattern}")

    for pattern in SAFETY_BYPASS_PATTERNS:
        if pattern.search(corpus):
            reasons.append(f"safety bypass intent: {pattern.pattern}")

    for update in patch.tool_boundary_updates:
        tool = str(update.get("tool", "")).strip().lower()
        allowed_when = str(update.get("allowed_when", "")).strip()
        tool_context = f"{tool} {allowed_when}"
        if any(token in tool for token in DANGEROUS_TOOL_NAMES):
            reasons.append(f"dangerous tool access requested: {tool}")
            continue
        for pattern in SAFETY_BYPASS_PATTERNS + DANGEROUS_TOOL_CONTEXT_PATTERNS:
            if pattern.search(tool_context):
                reasons.append(f"dangerous tool policy context: {pattern.pattern}")
                break

    return sorted(set(reasons))


def evaluate_policy(patch: IdentityPatch) -> PolicyResult:
    rejected_reasons = _rejected_reasons(patch)
    if rejected_reasons:
        return PolicyResult(
            allowed=False,
            requires_confirmation=False,
            rejected=True,
            reasons=rejected_reasons,
        )

    if patch.tool_boundary_updates:
        return PolicyResult(
            allowed=False,
            requires_confirmation=True,
            rejected=False,
            reasons=["tool-boundary change requires confirmation"],
        )

    return PolicyResult(
        allowed=True,
        requires_confirmation=False,
        rejected=False,
        reasons=[],
    )
