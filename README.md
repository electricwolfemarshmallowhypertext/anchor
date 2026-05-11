# Anchor

A local identity continuity layer for AI agents: versioned identity patches, validation, and rollback.
License: Apache-2.0

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

Real bench baseline (live local model path):

```bash
python -m bench.run_real_bench --model anchor --repeat 5 --output results/baseline-anchor.json
```

Baseline outputs:
- `results/baseline-anchor.json`
- `results/baseline-anchor.md`

Baseline metrics:
- `compile_valid_rate`
- `positive_case_success_rate`
- `safety_block_rate`
- `unsafe_accept_rate`
- `timeout_rate`
- `median_latency_ms`
- `p95_latency_ms`
- per-category pass rates
- failure-class counts

## Ollama Model Usage

## Run

```bash
ollama run tionne/anchor
```

## Best used with the CLI

GitHub:

```text
https://github.com/electricwolfemarshmallowhypertext/anchor
```

The Ollama model drafts patches; the CLI is the trusted write path.
The model returns IdentityPatch JSON drafts. The CLI normalizes, validates, applies, renders, and rolls them back.

CLI examples:

```bash
anchor history demo
anchor export demo --out identity.anchor.json
anchor import --in identity.anchor.json
anchor import --in identity.anchor.json --force-lineage
anchor import --in identity.anchor.json --force-hash
```

## Safety Model

- Compile and apply are separate commands; patches are not auto-applied
- Apply uses one atomic SQLite transaction for identity write + patch record + event record
- Version checks happen transactionally inside the write path
- Policy gate runs before writes
- Structural policy checks run before regex policy checks
- Integrity hash chain (`capsule_hash`, `previous_hash`) is stored per version
- Actor metadata (`requested_by`, `approved_by`, `applied_by`) is recorded on apply events
- Export/import uses an integrity envelope (`agent_id`, `version`, `capsule_hash`, `previous_hash`, `capsule`)
- Import bypasses are scoped (`--force-lineage`, `--force-hash`) with explicit warnings
- `anchor history` provides inspectable version/audit history without SQL access

## V1 hardening

- apply is atomic
- unsafe policy paths fail closed
- history is inspectable without SQL
- import bypasses are explicit and scoped

## Current Limitations

- local-first only; no hosted sync
- no multi-agent roles or delegation tokens
- model patch quality still varies, but compile output is normalized and validated before apply
- validator/policy are deterministic guardrails, not a complete security boundary
- integrity chain is tamper-evident inside local storage, not cryptographically signed externally
