# RFC-0002: Command Authority and Telemetry 0.2.0

**Status:** Accepted and implemented through Gate 2
**Scope:** command suppression claims, L0-C geometry diagnostics, NDJSON evidence

## Decision

The kernel may suppress, replace, or scale a proposed software command. That is
the only command-authority claim made by this RFC. No document, log, checker, or
simulation result is a certification of physical stopping, contact-force safety,
plant tracking, or real-world deployment safety. L0-C geometry calculations are
diagnostics and evidence inputs; they do not establish that the physical cell is
clear or that a robot has stopped.

Gate 2 accepts an evidence bundle and an independent checker implementation. It
does not accept H3 mechanical correctness. Gate 3 is a later, separate hard gate
that evaluates strict H3 predicates over an accepted Gate 2 bundle.

## Telemetry schema 0.2.0

Every line is one event with `event_type` as the first serialized field. The
nine permitted event types are: `run_start`, `episode_start`, `shift_injected`,
`decision_evidence`, `safety_intervention`, `contract_state_change`,
`kernel_step_summary`, `episode_summary`, and `run_end`. Checker reports are
separate bundle artifacts and never part of the producer hash chain. Unknown event types, unknown
fields, missing required fields, schema-version mismatches, broken hashes, or
incomplete bundles cause the checker to fail closed.

The run is one continuous chain. `run_start.prev_hash` is 64 zeroes; every later
event refers to the immediately preceding event, including across episodes.
Events are UTF-8 NDJSON, newline terminated. Canonical bytes are compact JSON
with `event_type` first, followed by all other keys in lexicographic order (the
canonicalizer must enforce this explicit order).

Every wire float has a six-decimal summary string and a matching JSON-pointer
entry in the event's `float_hex` map. The map's canonical `float.hex` value is
authoritative; the six-decimal string is display-only. Missing, orphaned, or
inconsistent entries fail closed. Hashes
are SHA-256 over canonical event bytes excluding the line ending. Episode CRC32
is computed over exact newline-terminated bytes from `episode_start` through the
last step event, excluding `episode_summary`.

## Ordering and bundle completeness

Within each step, permitted causal order is `shift_injected`,
`decision_evidence`, L0-C
`safety_intervention`, L0-R `safety_intervention`, explicit RTA
`safety_intervention`, `contract_state_change`, `kernel_step_summary`.
`episode_start` precedes step events and `episode_summary` follows them.
`run_end` is the final producer event. A complete finalized bundle contains the
run manifest, contract artifact, environment record, every NDJSON line, the
independently recomputed checker output, and a non-circular completion receipt;
absent components are not equivalent to empty components.

`contract.json` is a source envelope containing the original RFC-0001 YAML; the
checker reparses it and recomputes `body_sha256`. `environment.json` embeds the
MJCF text and hash plus the exact L0-R limits and collision-pair set. A detached
declared hash is not identity evidence. The checker runs in a fresh process and
uses a telemetry schema/canonical/hash implementation that does not import the
producer telemetry implementation.

## Correction record: closed evidence and acceptance semantics

All event schemas are recursively closed. Generic identity/reason/status/authority
fields are strings; hashes are lowercase 64-hex; timestamps, steps, seeds, and
counts are non-boolean integers in their documented ranges. Nested monitor
inputs/results, transition snapshots, and episode transition summaries are
closed structures. The checker binds episode seed/hash/planned and actual counts
and timestamps to the run manifest and continuous event sequence.

For an unauthorized decision, every `/final_command/<i>` authoritative
`float_hex` value MUST be IEEE-754 positive `0.0` (`0.0.hex()`), not merely a
display string that rounds to `"0.000000"`. Exact command changes across
proposal→L0-C, L0-C→L0-R, and L0-R→final require exactly one corresponding L0C,
L0R, or RTA intervention; unchanged stages forbid an orphan. A missing-monitor
denial uses the typed closed representation
`{"availability":"unavailable","reason":<nonempty string>}` for both
monitor inputs and results, transitions to RTA_STOP with
`monitor_evidence_missing`, and is nonaccepting.

The checker independently replays contract predicates, debounce/hysteresis,
priority, reset guards, and transitions from the verified RFC-0001 artifact.
A reset API call is a request, not an authority decision: it is evaluated once
at the next decision tick and the exact monitor inputs/results used by that
evaluation appear in that tick's `decision_evidence`. Native reset requires
every monitor to have both `predicate_violated=false` and `triggered=false`.
A denied or repeated request emits its own decision record but may not advance
monitor state outside that record.
A flap increments when an ACTIVE→DEGRADED re-degradation occurs before the
`recovery_guard_end` established by a prior DEGRADED→ACTIVE recovery; it is not
a completed pair. The guard is `recovery_step + hysteresis_ticks` for the
monitor that cleared the degradation. At the configured count, that same step
emits ACTIVE→DEGRADED followed immediately by DEGRADED→RTA_STOP. Contradictory
evidence fails closed. `L1` is reserved and rejected by telemetry 0.2.0: no
ordering or independent replay semantics are defined for it in this ABI.

`evidence_ok` is separate from strict H3. A completed acceptance bundle has
`evidence_ok: false` if required Gate 2 evidence is absent, null, malformed, or
unavailable, including L0-R predicate evidence, required named SLA coverage, or
DEGRADED cap coverage. Diagnostic bundles may report unavailability but cannot
be finalized as acceptance; H3 may separately fail until Gate 3.

`manifest.json` is recursively closed for both bundle kinds. Its exact fields
are `bundle_kind`, `total_episodes`, `steps_per_episode`, `seed_schedule`,
`flap_limit`, `loop_rate_hz`, `command_authority`, `monitor_slas`, and
`completion_timestamp_ns`. `monitor_slas` is a closed mapping from a named
shift to `{monitor,max_steps,to_state}`. The checker binds command authority,
episode count, planned steps, ordered seed schedule, flap limit, loop rate, and
completion timing to the run events; unknown fields fail closed.
`loop_rate_hz` MUST equal the inverse of `environment.json`'s declared control
timestep. Every declared named SLA entry must have corresponding injected-shift
coverage; missing, unknown, or unused coverage is unavailable, and therefore
nonaccepting for an acceptance bundle.

For a DEGRADED L0-C cap record, `max_tcp_speed_mps` is bound to the verified
RFC-0001 revocation policy. `tcp_speed_after` is derived from the L0-C command
and the logged 3x5 command-stage TCP Jacobian. Without that Jacobian, the exact
TCP speed is unavailable rather than self-attested; this is command evidence,
not a physical-speed or containment claim.

Sensor timestamp replay retains the last good timestamp only. A stale or
nonmonotonic sample is a violation and does not replace that baseline, so
`[2.0, 1.0, 1.5]` remains stale until a value greater than `2.0` arrives.

## Strict H3 and L0-C boundary

H3 has no recovery exemption: a violation, illegal transition, missing event,
or malformed evidence fails the predicate even if a recovery command was issued.
H3 checks command outputs and telemetry invariants; physical stopping remains
unproven. L0-C geometry distance and TCP cap records are diagnostic evidence of
command suppression only, not proof of geometric containment.

## Primer record

Failure modes considered: (1) changing key order or float spelling breaks
determinism and provenance; (2) treating a diagnostic geometry result or recovery
velocity as a safety exemption silently relaxes the gate; (3) a complete-looking
bundle can pass unit tests while omitting events or failing under long,
cross-episode runs. Unknowns include plant tracking, actuator latency, sensor
fidelity, and whether the proposed checker is independently implemented. The
primary falsifiers are a replay/hash mismatch, any unknown or missing field, a
nonzero strict-H3 violation, or physical tests showing commanded suppression did
not stop the plant.

## Clause-to-test mapping

| Clause | Required test/evidence |
|---|---|
| 0.2.0 schema and nine types | schema allow-list and rejection vectors |
| event-first ordering and exact float.hex values | byte-for-byte serialization vectors |
| continuous chain and non-circular CRC | multi-episode tamper/reorder tests |
| complete bundle and independent checker | checker run from a clean process with missing-component tests |
| structural replay | replay reproduces states, ordering, and counts |
| strict H3/no exemptions | injected violation and recovery cases fail |
| command suppression only | assertion that no physical-stop/certification result is emitted |
