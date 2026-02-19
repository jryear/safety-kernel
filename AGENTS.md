# Co-Architect Agent Instructions (Failure-First Safety Kernel)

This repository is safety-critical architecture work. Agents must be precise, fail-closed, and evidence-driven.

## Current state

- **Current gate:** Gate 2 (NDJSON telemetry + H3 checker)
- **Last completed:** Gate 1 (187/187 tests pass)
- **Tests:** `pytest tests/ -v` — all must pass before any PR

## Read first (in order)

1. `README.md` (project mental model + current gates)
2. `docs/failure_first_safety_kernel_cumulative_spec.md` (source-of-truth spec)
3. `docs/rfcs/RFC-0001-contract-artifact.md` (contract identity + merge doctrine)
4. `docs/co-architect-primer.md` (epistemic protocol for proposals/decisions)

## Process rules (non-negotiable)

- **Spec is law.** Do not “discover requirements” in code. If underspecified, harden the spec/RFC first.
- **Gates are hard stops.** Do not implement beyond the current gate unless explicitly authorized.
- **Fail closed.** Unknown fields, unknown versions, missing evidence, or partial observability ⇒ conservative behavior.
- **ABIs freeze early.** Any change to the following requires an RFC update + new test vectors:
  - Contract canonicalization / hashing / merge semantics (RFC-0001)
  - Telemetry schema / canonical JSON bytes / hash-chain rules
  - Gate acceptance criteria
- **Determinism is a feature.** Simulation time, ordering, hashing, and serialization must be reproducible from seed + artifacts.

## Hard prohibitions

- Do NOT relax erosion margins without RFC-level justification
- Do NOT add state machine states or transitions without spec update
- Do NOT change NDJSON field names, ordering, or hash semantics post-Gate 2
- Do NOT skip the primer for any work touching acceptance criteria
- Do NOT assume test passage equals spec compliance — cite the spec clause

## When you must use the co-architect primer

The full protocol in `docs/co-architect-primer.md` is required for:

- ABI changes (interfaces between layers)
- Any safety relaxation (wider limits, weaker erosion, lower conservatism)
- Gate acceptance criteria changes
- Contract identity / merge / signing changes
- Telemetry semantics changes (fields, hashing, ordering)
- Hypothesis definitions (H1–H5) or falsification criteria
- Door-closing architectural decisions (e.g., barrier vs critic, new states)

## Local commands (expected)

- Run tests: `.venv/bin/pytest -q`
- If adding new hashing/identity logic: add/extend deterministic test vectors in `tests/`
