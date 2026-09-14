# Gate 2 accepted checkpoint

Date: 2026-09-14
Boundary: software command suppression and evidence integrity only
Gate 2 decision: APPROVED for telemetry/checker evidence integrity
Gate 3: not entered

Fresh architectural approval: Orca message `msg_8066f641febf` confirmed the
prior technical audit and closed the final documentation finding. The approval
is bounded to `software_command_suppression_only`; strict H3, plant tracking,
physical stopping, and deployment certification remain unaccepted.

## Landed behavior

- Contract monitors decide authority before L0-C, L0-R, and the MuJoCo plant step.
- REVOKED, RTA_STOP, and L0-R near-miss paths suppress the final software command.
- Telemetry 0.2.0 emits nine closed event types in the RFC-0002 order with a
  continuous SHA-256 chain, newline-inclusive episode CRC32, exact binary-float
  evidence, immutable producer completion, independent finalization, and
  structural replay.
- The receipt binds the producer files and independently recomputed checker.
- The checker runs in a fresh interpreter with its own telemetry schema,
  canonicalization, and hash implementation. It recomputes RFC-0001 contract
  identity and validates embedded MJCF/config provenance.
- The only authority claim is `software_command_suppression_only`.

## Verification

The candidate load was produced twice with:

```text
.venv/bin/python scripts/run_gate2_acceptance.py \
  --out runs/<fresh-directory>/bundle --episodes 100 --steps 200
```

Both runs produced identical results:

```text
episodes:                  100
steps per episode:         200
producer events:           63,897
events.ndjson SHA-256:     0f6c6892a1af1c86550ef63c409602c5e58cb4566c49b99393c37b2351ab4bef
completion receipt SHA-256:a97b49c919f60f085d47fb83a730ffe5e901afd0d442cf9af1ae08d1e5ea4fee
evidence integrity:        PASS
structural replay:         PASS
H3 predicate availability: PASS (all five computable)
detection SLA:             PASS
transition legality:       PASS
flap predicate:            PASS
degraded command cap:      PASS
strict H3 L0-R:            FAIL (7,667 position/collision findings)
repository tests:          PASS (256)
```

The retained local bundle is
`runs/gate2-acceptance-20260914-final-g/bundle/`; `runs/` is intentionally ignored
because the NDJSON is large. The hashes above make accidental
drift detectable but are not a substitute for retaining the local bundle.

The H3 failure is not waived. The observations do not establish their physical
cause; they demonstrate that software command suppression does not by itself
establish plant tracking or physical stopping. Gate 2 acceptance concerns the
evidence pipeline and checker only. Gate 3 mechanical acceptance remains closed.
