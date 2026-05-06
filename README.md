# Anchor

A local identity continuity layer for AI agents: versioned identity patches, validation, and rollback.

Anchor gives local AI agents a small, versioned identity file so they stay consistent across sessions, update safely, and roll back bad changes.

## What Anchor Solves

- Identity drift across sessions, model swaps, and restarts
- Unsafe memory updates without auditability
- Inability to inspect what changed, why it changed, and when
- Lack of rollback when identity updates go wrong

## Install / Dev Setup

```bash
python -m venv .venv
```

PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install package + dev dependencies:

```bash
pip install -e .[dev]
```

## CLI Quickstart

```bash
anchor init research-assistant
anchor compile --agent research-assistant --session examples/session.md --out patch.json
# apply is separate and can require --confirm-risky for gated patches
anchor apply --agent research-assistant --patch patch.json
anchor render research-assistant
```

## Evals

GitHub tests/evals run offline and do not require Ollama.

```bash
python -m evals.run_evals
python -m pytest -q
```

Live benchmark (structured output via Python client, not `ollama run`):

```bash
python -m evals.run_ollama_benchmark --model yourname/anchor --case 02_explicit_preference_planning_depth
```

## Ollama Model Usage

Public Ollama model: `yourname/anchor`  
Local dev model: `anchor`

Published Ollama model usage:

```bash
ollama pull yourname/anchor
```

Private/local publish command:

```bash
ollama push tionne/anchor
```

Local development can still use the Modelfile to create the model manually:

```bash
ollama create anchor -f Modelfile
```

Anchor expects an Ollama model that returns an `IdentityPatch` JSON object.
Use Ollama JSON mode (`format="json"`) with `/api/generate`, then validate locally with Pydantic (`IdentityPatch`) before any write.

## Safety Model

- Compile and apply are separate commands; patches are not auto-applied
- Deterministic validator checks run before apply
- Rejects secret-like values and known prompt-injection patterns
- Rejects unsupported authority expansion in tool boundaries
- Requires explicit confirmation for risky changes
- Versioned storage keeps audit trail and rollback path

## Current Limitations

- local identity capsule only
- patch quality depends on compiler/model output
- validator is deterministic but not complete security
- no hosted sync
- no multi-agent governance yet
