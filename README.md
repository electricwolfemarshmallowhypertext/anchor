# Anchor

Anchor gives local AI agents a small, versioned identity file.

It remembers what should stay true:
- role
- operating style
- user preferences
- tool boundaries
- corrections
- unresolved conflicts

It compiles session transcripts into safe identity patches, validates them, and lets you roll back.

## Scope

- Local, CLI-first workflow
- SQLite source of truth with JSON export
- Deterministic validation gates before apply
- No vector database
- No web UI

## Quickstart

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]

anchor init research-assistant
anchor compile --agent research-assistant --session examples/session.md --out patch.json
anchor apply --agent research-assistant --patch patch.json
anchor render research-assistant
```

## Commands

```text
anchor init <agent_id>
anchor compile --agent <agent_id> --session <file> [--out patch.json]
anchor apply --agent <agent_id> --patch <patch.json> [--confirm-risky]
anchor show <agent_id>
anchor render <agent_id>
anchor rollback <agent_id> --version <n>
anchor export <agent_id> --out <file>
```

## Model Integration

Anchor expects an Ollama model that returns an `IdentityPatch` JSON object.
Use `format=<IdentityPatch JSON schema>` with `/api/generate`. Output is parsed and validated before any write.
