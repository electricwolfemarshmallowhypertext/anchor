from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compiler import apply_patch, compile_patch, patch_to_json
from .models import IdentityCapsule, IdentityPatch
from .ollama_client import OllamaClient
from .renderer import render_identity_block
from .store import DEFAULT_DB_PATH, append_patch, export_json, init_db, load_identity, rollback, save_identity
from .validator import validate_patch


def _db_path(args: argparse.Namespace) -> Path:
    return Path(args.db)


def _must_load_identity(agent_id: str, db_path: Path) -> IdentityCapsule:
    capsule = load_identity(agent_id, db_path)
    if capsule is None:
        raise ValueError(f"agent '{agent_id}' not found; run 'anchor init {agent_id}' first")
    return capsule


def cmd_init(args: argparse.Namespace) -> int:
    db_path = _db_path(args)
    init_db(db_path)
    existing = load_identity(args.agent_id, db_path)
    if existing is not None:
        print(f"agent '{args.agent_id}' already exists at version {existing.version}")
        return 0

    capsule = IdentityCapsule(
        agent_id=args.agent_id,
        agent_name=args.agent_name or args.agent_id,
        purpose=args.purpose or "",
        working_style={
            "tone": "direct",
            "verbosity": "brief unless planning",
            "citation_policy": "cite non-obvious claims",
        },
        tool_boundaries=[
            {"tool": "web", "allowed_when": "fresh facts, niche claims, or verification needed"},
        ],
    )
    saved = save_identity(capsule, db_path)
    print(f"initialized agent '{saved.agent_id}' version={saved.version}")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    db_path = _db_path(args)
    capsule = _must_load_identity(args.agent, db_path)
    transcript = Path(args.session).read_text(encoding="utf-8")
    client = OllamaClient(base_url=args.base_url, timeout=args.timeout)
    patch = compile_patch(capsule=capsule, transcript=transcript, client=client, model=args.model)
    out_path = Path(args.out)
    out_path.write_text(patch_to_json(patch), encoding="utf-8")
    print(f"compiled patch -> {out_path}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    db_path = _db_path(args)
    capsule = _must_load_identity(args.agent, db_path)
    patch_payload = json.loads(Path(args.patch).read_text(encoding="utf-8"))
    patch = IdentityPatch.model_validate(patch_payload)

    result = validate_patch(capsule, patch)
    if not result.accepted:
        print("patch rejected:")
        for err in result.errors:
            print(f"- {err}")
        return 1

    if result.requires_confirmation and not args.confirm_risky:
        print("patch requires confirmation:")
        for reason in result.confirmation_reasons:
            print(f"- {reason}")
        print("re-run with --confirm-risky to apply")
        return 1

    updated = apply_patch(capsule, patch)
    saved = save_identity(updated, db_path)
    append_patch(args.agent, patch, db_path)
    print(f"patch applied; new version={saved.version}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    capsule = _must_load_identity(args.agent_id, _db_path(args))
    print(capsule.model_dump_json(indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    capsule = _must_load_identity(args.agent_id, _db_path(args))
    print(render_identity_block(capsule))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    db_path = _db_path(args)
    restored = rollback(args.agent_id, args.version, db_path)
    print(
        "rollback completed: "
        f"agent={restored.agent_id} rollback_version={restored.rollback_version} version={restored.version}"
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    out = export_json(args.agent_id, args.out, _db_path(args))
    print(f"exported -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchor")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("agent_id")
    init_parser.add_argument("--agent-name")
    init_parser.add_argument("--purpose")
    init_parser.set_defaults(func=cmd_init)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--agent", required=True)
    compile_parser.add_argument("--session", required=True)
    compile_parser.add_argument("--model", default="anchor")
    compile_parser.add_argument("--base-url", default="http://localhost:11434")
    compile_parser.add_argument("--timeout", type=float, default=60.0)
    compile_parser.add_argument("--out", default="patch.json")
    compile_parser.set_defaults(func=cmd_compile)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--agent", required=True)
    apply_parser.add_argument("--patch", required=True)
    apply_parser.add_argument("--confirm-risky", action="store_true")
    apply_parser.set_defaults(func=cmd_apply)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("agent_id")
    show_parser.set_defaults(func=cmd_show)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("agent_id")
    render_parser.set_defaults(func=cmd_render)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("agent_id")
    rollback_parser.add_argument("--version", type=int, required=True)
    rollback_parser.set_defaults(func=cmd_rollback)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("agent_id")
    export_parser.add_argument("--out", required=True)
    export_parser.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
