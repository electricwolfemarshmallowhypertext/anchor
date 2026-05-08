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
anchor init demo
anchor compile --agent demo --session examples/session.md --out patch.json
anchor apply --agent demo --patch patch.json
anchor render demo
anchor rollback demo --version 1
```

`anchor render demo` prints a prompt-ready identity block you can copy into another agent's system prompt.

## Evals

GitHub tests/evals run offline and do not require Ollama.

```bash
python -m evals.run_evals
python -m pytest -q
```

Live benchmark (structured output via Python client, not `ollama run`):

```bash
python -m evals.run_ollama_benchmark --model tionne/anchor --case 02_explicit_preference_planning_depth
```

Minimal live publish gate (blocks release if critical cases miss threshold):

```bash
python -m evals.run_publish_gate --model anchor --repeat 3 --min-pass-rate 0.90
```

## Ollama Model Usage

Public Ollama model: `tionne/anchor`  
Local dev model: `anchor`

Published Ollama model usage:

```bash
ollama pull tionne/anchor
ollama run tionne/anchor
```

Local development can still use the Modelfile to create the model manually:

```bash
ollama create anchor -f Modelfile
```

Anchor is CLI-first: the model drafts patches, then the CLI normalizes, validates, applies, renders, and rolls back identity state.
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
