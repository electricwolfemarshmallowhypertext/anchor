from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


FIELD_PATH_RE = re.compile(
    r"^(purpose|working_style\.[A-Za-z0-9_][A-Za-z0-9_.-]*|"
    r"user_preferences\.[A-Za-z0-9_][A-Za-z0-9_.-]*|"
    r"decisions\.[A-Za-z0-9_][A-Za-z0-9_.-]*|"
    r"corrections\.[A-Za-z0-9_][A-Za-z0-9_.-]*)$"
)


def is_valid_field_path(value: str) -> bool:
    return bool(FIELD_PATH_RE.match(value))


class IdentityField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any
    source: Literal[
        "explicit_user_instruction",
        "inferred_from_behavior",
        "manual_override",
        "system_rule",
        "model_inference",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    requires_confirmation: bool = False
    evidence: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def key_must_not_be_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("key must not be empty")
        if not is_valid_field_path(trimmed):
            raise ValueError(f"invalid key path: {trimmed}")
        return trimmed


class Conflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    existing_value: Any
    proposed_value: Any
    reason: str
    requires_confirmation: bool = True


class Suspicion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    detail: str
    evidence: str | None = None


class IdentityPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    from_version: int = Field(ge=1)
    summary: str = ""
    field_updates: list[IdentityField] = Field(default_factory=list)
    field_removals: list[str] = Field(default_factory=list)
    tool_boundary_updates: list[dict[str, Any]] = Field(default_factory=list)
    requires_confirmation: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    suspicions: list[Suspicion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("field_removals")
    @classmethod
    def field_removals_must_use_supported_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            if not is_valid_field_path(value):
                raise ValueError(f"invalid removal path: {value}")
        return values

    @field_validator("tool_boundary_updates")
    @classmethod
    def tool_boundary_updates_must_be_known_shape(
        cls, values: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for item in values:
            keys = set(item.keys())
            if keys != {"tool", "allowed_when"}:
                raise ValueError("tool boundary updates may only include 'tool' and 'allowed_when'")
            if not str(item.get("tool", "")).strip():
                raise ValueError("tool boundary update 'tool' must be non-empty")
            if not str(item.get("allowed_when", "")).strip():
                raise ValueError("tool boundary update 'allowed_when' must be non-empty")
        return values


class IdentityCapsule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    agent_name: str | None = None
    purpose: str = ""
    working_style: dict[str, Any] = Field(default_factory=dict)
    operating_rules: list[str] = Field(default_factory=list)
    user_preferences: list[IdentityField] = Field(default_factory=list)
    decisions: list[IdentityField] = Field(default_factory=list)
    corrections: list[IdentityField] = Field(default_factory=list)
    tool_boundaries: list[dict[str, Any]] = Field(default_factory=list)
    recent_identity_changes: list[str] = Field(default_factory=list)
    open_conflicts: list[Conflict] = Field(default_factory=list)
    drift_history: list[str] = Field(default_factory=list)
    rollback_version: int = Field(default=1, ge=1)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    errors: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_reasons: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    suspicions: list[Suspicion] = Field(default_factory=list)
