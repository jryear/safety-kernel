# Co-Architect Primer (Epistemic Contract for Agents)

**Purpose:** force computation that can’t be retrieved; prevent fluent-but-wrong work on safety-critical architecture.

This is a contract for *how* we reason, propose, implement, and accept changes in this repository.

---

## When this primer is REQUIRED

Use the full protocol for any of the following:

- ABI changes (interfaces between layers or subsystems)
- Safety relaxations (loosening any constraint, erosion margin, threshold, monitor, or gate criterion)
- Gate acceptance criteria (defining or modifying “pass”)
- Contract identity / merge / signing semantics (RFC-0001 and successors)
- Telemetry semantics (what is logged, how it is hashed, event ordering)
- Hypothesis definitions (H1–H5 scope, falsifiers, metrics)
- Architectural decisions that close doors (state machine topology, barrier vs critic, etc.)

For routine work implementing to a frozen spec, it is optional.

---

## Engagement protocol

### Before answering / proposing

1. **Generate three failure modes of the obvious answer.**
   - One should be “this breaks determinism / provenance.”
   - One should be “this silently relaxes safety.”
   - One should be “this passes tests but fails at scale / under shift.”

2. **Identify what you’d need to know to be confident (that you don’t).**
   - Missing measurements, missing spec clauses, missing test vectors, unknown environment constraints.

3. **Name the assumption you’re most likely wrong about.**
   - Prefer the assumption that, if wrong, would cause silent harm (not just a failing test).

4. **Evidence rule (replaces ‘confidence introspection’):**
   - High confidence is allowed **only** if supported by **external evidence**:
     - a specific spec/RFC clause, a unit/integration test, or a measurement.
   - If you can’t cite evidence, treat the claim as an assumption and label it.

### During answering / designing

1. **Actively search for the version of the problem where your answer breaks.**
   - Adversarial inputs, schema edge cases, rounding boundaries, concurrency, missing sensors.

2. **Hold competing hypotheses until forced to collapse.**
   - Don’t prematurely pick a single story when multiple explain the data.

3. **Refuse hidden degrees of freedom.**
   - If an outcome depends on a choice (ordering, rounding, time base), specify it.

### After answering / proposing

1. **State what would change your answer.**
2. **Identify the experiment/test that resolves remaining uncertainty.**
3. **Point to the falsifier.**
   - The smallest test/log invariant that would prove you wrong.

---

## Meta-constraint

- If this primer feels easy to follow, you’re probably not following it.
- The primer should create **friction**, not fluency.

---

## Required output format (for primer-scoped work)

1. **Answer** (decision/proposal)
2. **What I don’t know that matters** (unknowns + missing evidence)
3. **What would falsify this** (tests/measurements/log checks)

Optional (recommended for architectural changes):

4. **Failure modes considered** (3 items)
5. **Most-likely-wrong assumption**

