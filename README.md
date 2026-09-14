# Failure-First Safety Kernel

A safety kernel for robot deployments that inverts the standard paradigm: instead of training a robot to succeed and hoping it doesn't fail, we train a **failure model** and use it as the primary safety layer.

The unit of operation is **(robot x cell x assumptions)**. One robot, one workspace, one set of certified assumptions. If any assumption breaks, the system detects it, revokes trust, and degrades gracefully.

---

## How we got here

This project started from a practical observation: every major humanoid company (Tesla, Figure, 1X, Boston Dynamics) ships robots with bolted-on safety -- force limits, emergency stops, compliance hardware, human oversight. Public materials do not establish a physical-stop guarantee when assumptions break. This project therefore records which software commands were suppressed and what evidence supports that claim.

The origin conversation (`Robotics Development Challenges.pdf`) identified three structural problems with success-first robot learning:

1. **Success policies are fragile under distribution shift.** A policy trained to grasp objects in one cell layout fails unpredictably when the layout changes. The failure modes are high-dimensional and task-specific.

2. **Failure manifolds are lower-dimensional and more stable.** The ways a robot can violate a joint limit, collide with a fixture, or exceed an energy budget are geometrically simpler and transfer better across tasks than the ways it can successfully complete those tasks.

3. **No existing system separates trust by epistemic status.** Joint limits are known from the manufacturer. Cell geometry is known from the integrator. Learned safety adapts online. Task policy is untrusted. These have fundamentally different certainty profiles, but existing architectures flatten them into a single safety layer.

These observations led to the layered trust architecture and the failure-first learning hypothesis that the current codebase implements and tests.

---

## Framework required to understand this project

### Core invariant

> Untrusted capability under trusted constraint, with authority revocable at runtime.

Every layer in the system has a defined trust level. Lower layers constrain upper layers. Trust is never assumed -- it is certified, monitored, and revocable.

### Layered trust hierarchy

The architecture separates safety into four layers based on **epistemic status** -- not importance, not severity, but *how we know what we know*:

| Layer | What it knows | Who certifies it | Mutability |
|-------|--------------|-------------------|-----------|
| **L0-R** (Robot intrinsics) | Joint limits, self-collision, energy caps, torque limits | Robot manufacturer | Immutable. Physics the robot knows about itself. |
| **L0-C** (Cell contract) | Workspace geometry, anchor positions, sensor health, safety I/O | Systems integrator | Revocable. Physics the deployment claims about the workspace. |
| **L1** (Learned safety) | Failure critic, boundary residuals, contact dynamics | Online calibration | Adaptive. Physics the robot learns about the world. |
| **L2** (Task policy) | How to do the job | Any method (RL, planning, teleop) | Swappable. Zero safety responsibility. |

L0-R is always enforced, regardless of what any other layer does. L0-C is enforced only while its assumptions hold -- if anchors drift or sensors fail, L0-C is revoked and L1 switches to full conservative mode. L1 is calibrated online with conformal prediction and provides learned safety only after it has warmed up. L2 proposes actions; every layer above it decides what's actually allowed.

### Contract state machine

The cell contract drives a 4-state machine:

```
ACTIVE ──(degrade)──> DEGRADED ──(revoke)──> REVOKED ──(rta_stop)──> RTA_STOP
  │                      │                      │                       │
  │<──(clear+hysteresis)─┘                      │<──(operator reset)────┘
  │<────────────────────────────────────────────┘<──(operator reset)────┘
  │──(revoke)──> REVOKED                        │──(rta_stop)──> RTA_STOP
  │──(rta_stop)──> RTA_STOP
```

Every transition has a logged trigger. Illegal transitions raise a typed runtime
error. Under RFC-0002, a flap is an ACTIVE->DEGRADED re-degradation inside the
recovery guard established by a prior DEGRADED->ACTIVE transition; at the configured threshold,
the limiter emits DEGRADED->RTA_STOP on that same tick.

### Failure-first learning (the thesis)

The central hypothesis: a failure critic Q_fail(s,a) transfers better across tasks and environments than success-trained policies, because the failure manifold is lower-dimensional and more physics-stable. This is tested empirically across Gates 4-8, not assumed.

The advantage is expected to be strongest for geometric/kinematic failures (collision, joint limits, energy) and weaker for contact-rich manipulation (stick-slip, deformable contact, grasp stability). The experimental pipeline is designed to measure this distinction, not assume it away.

### Key technical concepts

- **Minkowski erosion**: L0-R enforces limits tighter than physical limits by a sensor-uncertainty margin. The kernel operates on `q_max_eroded = q_max_physical - (eps_q + eps_qdot * dt)`. This is a command-space conservatism claim; physical stopping and plant tracking remain unproven.

- **Sequential projection**: L0-R applies constraints in order (velocity box clamp, position lookahead, energy cap, self-collision check). Each constraint set is star-convex around zero velocity, so scaling toward zero preserves feasibility of all previously satisfied constraints.

- **Conformal calibration**: L1's failure threshold adapts online using conformal prediction with a target false-negative rate alpha. The threshold starts at infinity (L1 disabled) and tightens only after sufficient boundary events are observed during warmup.

- **Contract canonicalization (RFC-0001)**: Cell contracts are YAML documents canonicalized to deterministic JSON, SHA-256 hashed, and optionally signed. The robot verifies the hash before it moves. Provenance is a product.

- **Shift injection**: Deterministic fault injection for testing monitor SLAs. The shift injector owns the "perfect baseline" (contract expected values) and applies drift/fault overlays at specified steps.

---

## What exists today

### Implemented and tested (Gate 1 baseline; test count is not a certification claim)

**L0-R projection** (`src/safety_kernel/l0r/`)
- Sequential 4-stage projection: velocity clamp, position lookahead, energy cap, self-collision
- Capsule-capsule surface distance computation for 5 active collision pairs
- Minkowski erosion with configurable sensor uncertainty parameters
- P99 latency: 0.165ms (<1ms target)

**L0-C contract layer** (`src/safety_kernel/contract/`)
- RFC-0001 canonicalization (SHA-256, ROUND_HALF_UP quantization, ordering invariance)
- Typed schema with frozen dataclasses (Contract, Keepout, Anchor, MonitorConfig, etc.)
- Contract loader with hash verification and schema version validation
- 5 assumption monitors with tick-based debounce/hysteresis:
  - `anchors_consistent` (250ms debounce, 500ms hysteresis, revoke severity)
  - `gate_interlock_ok` (0ms/0ms, rta_stop severity)
  - `estop_chain_ok` (0ms/0ms, rta_stop severity)
  - `front_cam_health` (500ms/1000ms, degrade severity)
  - `robot_mount_integrity` (250ms/500ms, revoke severity)
- 12-row contract state machine with flap limiter, operator reset, snapshot/restore
- L0-C TCP speed cap enforcement (uniform scaling in DEGRADED, pre-L0-R)
- Deterministic shift injector (anchor_drift, safety_io_flip, sensor_stale, base_drift)

**Simulation environment** (`src/safety_kernel/env/`)
- CellArm 5-DOF MuJoCo environment (Z-Y-Z-Y-Z joint pattern, ~1m reach, 11.5kg)
- 15D proprioceptive observation (q, qdot, tau)
- Optional contract integration (backward-compatible with no-contract mode)
- Full L0-C + L0-R pipeline in step(): shift injection, monitor evaluation, state machine, L0-C enforce, L0-R project

**Track A contract** (`contracts/cell_arm_track_a.yaml`)
- 2 keepouts (fixture_column, back_wall), 1 allowed region
- 2 fiducial anchors (fiducial_fixture, fiducial_wall)
- 5 assumption monitors with configured SLAs
- Verified body hash

### What the tests prove (and don't prove)

| Claim | Evidence | Confidence | What would falsify it |
|-------|----------|------------|----------------------|
| Zero L0-R violations | 20,000 steps across 100 episodes, 0 violations | High for tested envelope | Any qdot_exec exceeding eroded limits under adversarial L2 or noisy observations |
| Detection SLA | 5/5 monitors trigger within spec'd tick budget | High for deterministic injection | SLA miss under sensor noise or concurrent multi-fault injection |
| No illegal transitions | All observed transitions match 12-row table | High (exhaustive for single-fault) | Concurrent fault + L0-R near-miss producing unlisted transition |
| No flapping | RFC-0002 limiter fires on the threshold re-degradation inside a recovery guard | High for mechanism | Hysteresis window too short for real oscillatory fault patterns |
| Determinism | Bitwise identical qdot_exec across 2 runs, same seed | Confirmed | Any platform-dependent floating-point divergence |
| Backward compat | Gate 0 tests pass unchanged with contract integration present | Confirmed | Any contract-mode code path altering no-contract behavior |

**Honestly not yet tested:**
- Calibration stability (H2) -- MVP conformal thresholding exists, but no online stability or shift-recovery claim is validated yet
- Telemetry flywheel integrity (H4) -- Gate 2 NDJSON telemetry and independent checking now exist; H4 flywheel quality/operations remain unvalidated
- Failure-first transfer advantage (H1/H5) -- MVP failure critic exists, but no success baseline comparison or ManiSkill transfer evidence exists yet
- Behavior under observation noise (proprioceptive noise injection starts at Gate 4)
- Multi-fault concurrent injection
- Plant fidelity with nsubsteps >= 10

---

## Project structure

```
robot/
  docs/
    failure_first_safety_kernel_cumulative_spec.md   # Source of truth (all sections)
    rfcs/
      RFC-0001-contract-artifact.md                  # Contract canonicalization RFC
      RFC-0002-command-authority-and-telemetry.md    # Command claims + telemetry 0.2.0
  contracts/
    cell_arm_track_a.yaml                            # Track A cell contract
    cell_contract_v0.1_example.yaml                  # Template contract
  models/
    cell_arm.xml                                     # MuJoCo MJCF model (5-DOF arm)
  src/safety_kernel/
    l0r/
      projection.py                                  # L0-R sequential projection
      capsule.py                                     # Capsule-capsule distance
      params.py                                      # L0RParams from MJCF
    contract/
      canonicalize.py                                # RFC-0001 canonicalization
      canonical.py                                   # Canonical form utilities
      strict_yaml.py                                 # Strict YAML parsing
      schema.py                                      # Typed contract dataclasses
      loader.py                                      # Contract load + hash verify
      monitors.py                                    # 4 monitor types + debounce/hysteresis
      state_machine.py                               # 12-row transition table + flap limiter
      l0c_enforce.py                                 # TCP speed cap (DEGRADED state)
      shift_injector.py                              # Deterministic fault injection
      errors.py                                      # Error classes
    env/
      cell_arm.py                                    # CellArm 5-DOF Gymnasium env
    l1/
      critic.py                                      # Deterministic NumPy failure critic
      calibration.py                                 # Conformal risk threshold
      labels.py                                      # Sparse L0-R boundary labels
      metrics.py                                     # Safety metric aggregation
      rollouts.py                                    # Deterministic PPO-style rollout adapter
      ablations.py                                   # Erosion x coverage ablation runner
    telemetry/
      __init__.py                                    # Implemented Gate 2 bundle/schema/adapter/checker/CLI surface
  tests/
    test_l0r_projection.py                           # L0-R unit tests (15 tests)
    test_gate0_acceptance.py                         # Gate 0 smoke + determinism (2 tests)
    test_canonicalization.py                          # RFC-0001 hash vectors (38 tests)
    test_contract_identity.py                        # Contract identity tests (6 tests)
    test_contract_loader.py                          # Load/validate/reject (13 tests)
    test_monitors.py                                 # 4 monitor types + timing (28 tests)
    test_state_machine.py                            # All transitions + flapping (39 tests)
    test_l0c_enforce.py                              # TCP speed cap (12 tests)
    test_shift_injector.py                           # Fault injection (20 tests)
    test_gate1_integration.py                        # SLA + integration (14 tests)
    test_l1_failure_manifold.py                      # L1 critic + ablation smoke tests
  experiments/
    run_manifest_template.yaml                       # Run manifest template
    run_failure_manifold_ablation.py                 # L1 erosion x coverage ablations
  benchmarks/
    bench_l0r.py                                     # L0-R latency benchmark
  pyproject.toml
```

---

## Implementation roadmap

### Track A -- Safety-Gymnasium: kernel mechanics + calibration

- [x] **Gate 0: L0-R hard projection**
  - *Pass criteria:*
    - [x] Zero kernel-output violations across 100 episodes (20,000 steps)
    - [x] All four L0-R constraints verified per step: velocity limits, position lookahead, energy cap, self-collision
    - [x] Bitwise determinism confirmed (same seed produces identical qdot_exec)
    - [x] Erosion strictly tighter than physical limits
    - [x] L2 intervention rate in 5-20% band (measured: 11.4% at K_p=0.5)
    - [x] P99 per-step latency < 1ms (measured: 0.165ms)
  - *What this evidenced:* The projection algorithm produced constraint-satisfying software commands for the tested operating envelope. Physical stopping and plant realization were not tested to acceptance.
  - *Known gap carried forward:* Plant tracking gap (~64k diagnostic violations from MuJoCo Euler integration); this remains an unproven physical-safety boundary.

- [x] **Gate 1: L0-C schema + monitors + shift injector**
  - *Pass criteria:*
    - [x] Contract canonicalization produces deterministic hashes (ordering invariance, quantization boundary, number formatting)
    - [x] Contract loader rejects hash mismatch, missing integrity block, wrong schema version
    - [x] All 5 monitors fire within detection SLA (debounce_ticks + 1 steps from first violation)
    - [x] All state machine transitions match the 12-row transition table; illegal transitions fail with a typed runtime error
    - [x] Flap limiter emits DEGRADED->RTA_STOP on the threshold ACTIVE->DEGRADED re-degradation inside the RFC-0002 recovery guard
    - [x] L0-C TCP speed cap enforces uniform scaling in DEGRADED (monotone, deterministic, pre-L0-R)
    - [x] Shift injector produces correct perfect baseline and applies each shift type deterministically
    - [x] Full backward compatibility: all Gate 0 tests pass unchanged with contract integration present
    - [x] Gate 1 test suite passes in the recorded baseline (exact count is not current acceptance evidence)
  - *What this proved:* The contract layer correctly parses, validates, monitors, and revokes. Detection SLAs hold under deterministic single-fault injection. The state machine is legal and the flap limiter is functional.
  - *Known gap carried forward:* All testing under perfect observations. No sensor noise, no multi-fault combinations, no real perception stack.

- [x] **Gate 2: NDJSON telemetry + H3 checker** *(architecturally approved; Gate 3 remains closed)*
  - *Pass criteria:*
    - [x] Event serialization is byte-deterministic (same seed, same run produces identical NDJSON)
    - [x] Hash chain is continuous across the entire run (no per-episode breaks)
    - [x] `run_start` event anchors chain with prev_hash = 64 zeros
    - [x] `kernel_step_summary` emitted every step with fixed schema
    - [x] `safety_intervention` emitted only on actual modification, with `layer` field ("L0C"/"L0R"/"RTA")
    - [x] Deterministic causal ordering within each step: shift, decision evidence, L0C, L0R, RTA, state change, summary
    - [x] Gate 2 evidence bundle, independent checker, and structural replay received fresh architectural approval (H3 mechanical acceptance is Gate 3)
    - [x] Replay from NDJSON rehydrates states, ordering, and counts
    - [x] All prior tests still pass (backward compat)
  - *Expected outcome:* A complete, hash-verifiable evidence bundle for command suppression. This does not prove physical stopping or H3 mechanical correctness.
  - *Spec reference:* RFC-0002 and Appendix C (C.1-C.11), frozen with nine event types (including decision_evidence), continuous producer hash chain, exact float.hex values, complete-bundle rules, and structural replay.
  - *Accepted Gate 2 evidence:* two independent 100-episode x 200-step productions emitted byte-identical 63,897-event NDJSON (`0f6c6892...1ab4bef`). Evidence integrity passed and every H3 predicate was computable; detection SLA, transition, flap, and degraded-cap predicates passed. Strict H3 reported 7,667 L0-R findings and therefore remains unaccepted for Gate 3. See `docs/gate2-acceptance-checkpoint.md`.

### Parallel MVP: failure-manifold critic ablations

This track is additive and does not relax Gate 2. It produces the first
CoRL-facing safety plots around the fresh stack: hard projection, contract
canonicalization, deterministic shift injection, and learned boundary scores.

```bash
.venv/bin/python experiments/run_failure_manifold_ablation.py --quick
.venv/bin/python experiments/run_failure_manifold_ablation.py \
  --out experiments/results/failure_manifold \
  --compare-calibration-units
```

The runner writes `diagnostics.csv`, `metrics.csv`, `summary.json`, and SVG plots for false
positive rate, false negative rate, empirical unsafe coverage, coverage gap,
intervention rate, AUC/PR-AUC, and SLA hit rate across erosion scales and
conformal coverage settings. The CSV also includes per-constraint activation
rates, component base rates, and intervention severity quantiles so a dead or
degenerate erosion axis is visible before classifier metrics are interpreted. See
`docs/failure_manifold_mvp.md` for scope and limitations.

- [ ] **Gate 3: H3 mechanical validation (no L1)**
  - *Pass criteria:*
    - [ ] H3(a): Zero L0-R violations across a full Track A run (target: 500+ episodes; no recovery exemptions)
    - [ ] H3(b): 100% detection SLA compliance verified by H3 checker against NDJSON logs
    - [ ] H3(c): Zero illegal transitions in logged events
    - [ ] H3(d): No flapping beyond threshold (N=4) per episode
    - [ ] All four predicates checked by the H3 offline checker (not just unit tests)
    - [ ] Run manifest records exact configuration: seed, contract hash, env config hash
  - *Expected outcome:* Strict H3 mechanical evidence is evaluated at scale with full telemetry. This remains a hard checkpoint and makes no physical-stopping claim.
  - *What it means if it fails:* Either L0-R projection has an edge case, or the monitor/state-machine has a bug exposed by scale. Both are findable and fixable.

- [ ] **Gate 4: Conformal calibration loop + cold-start protocol**
  - *Pass criteria:*
    - [ ] Calibration reaches warmth within n_warmup_episodes (default: 200) under stationary conditions
    - [ ] Warmth criterion: |FN_empirical - alpha| <= coverage_tolerance AND 95% Clopper-Pearson CI width <= 2x coverage_tolerance
    - [ ] Post-warmup FN rate <= alpha (default: 0.01) with proprioceptive noise active
    - [ ] Proprioceptive noise injection active: observed != truth for q, qdot, tau
    - [ ] Plant fidelity criterion: with nsubsteps >= 10, per-joint p99 tracking errors < eps_q (position) and < eps_qdot (velocity)
    - [ ] L1 logs risk scores but does NOT filter actions during warmup (tau_0 = +inf)
    - [ ] Epsilon sweep across near-violation label distances (percentile-based per constraint type)
  - *Expected outcome:* The conformal loop converges and produces a meaningful threshold under noisy observations. This is the first gate where learned components exist; cold-start command suppression remains limited to the specified software layers.
  - *What it means if it fails:* Calibration doesn't converge within budget -- either not enough boundary events in 200 episodes, or the noise injection creates too much variance for conformal bounds to tighten.

- [ ] **Gate 5: Calibration under shift (H2)**
  - *Pass criteria:*
    - [ ] Pre-shift: FN <= alpha (post-warmup, proprioceptive noise active)
    - [ ] Post-shift: FN spikes then recovers to <= alpha within bounded episodes
    - [ ] Threshold trajectory adapts (not stuck) -- logged and plotted
    - [ ] Tested under single-factor shifts: geometry drift, sensor degradation, dynamics perturbation
  - *Expected outcome:* H2 validated -- calibration is stable under distribution shift. The conformal threshold adapts in bounded time.
  - *What it means if it fails:* The conformal method can't track non-stationary distributions fast enough, or the failure manifold changes shape too quickly under shift. Would need to revisit calibration mechanism (windowed vs. online, different score function).

- [ ] **Gate 6: L1 predictive safety -- critic vs. barrier comparison**
  - *Pass criteria:*
    - [ ] **Arm A** (failure critic Q_fail with threshold-based shielding):
      - [ ] Dual-head separation demonstrated: residual head (ACTIVE) and full head (REVOKED) produce different risk scores on the same states
      - [ ] Smooth handoff on state machine transition (no risk score discontinuity beyond configured tolerance)
      - [ ] < 10ms decision time
    - [ ] **Arm B** (learned barrier h_theta with CBF-QP projection):
      - [ ] Barrier certificate holds (h(s) >= 0 on safe set)
      - [ ] CBF-QP solves within 5ms time budget
      - [ ] < 10ms decision time
    - [ ] Both arms enforce correctly under standard test suite
    - [ ] Comparison data logged for architectural decision (critic vs. barrier)
    - [ ] Full-head warmth maintenance: synthetic revocation episodes keep full head calibrated
    - [ ] ENABLE_PREDICTIVE_SHIELD promoted from false to true (if both arms pass)
  - *Expected outcome:* Both arms work. The comparison data informs which approach (or hybrid) to carry forward. Gate 6 forces the barrier-vs-critic decision with real data.
  - *What it means if it fails:* One or both predictive approaches don't meet timing/accuracy requirements. If the barrier can't solve in time, critic wins by default. If neither works, the architecture falls back to L0-R + L0-C only (still safe, just less capable).

- [ ] **Gate 7: H4 telemetry flywheel (3 arms + novel failure discovery)**
  - *Pass criteria:*
    - [ ] Mixed arm (80/20 telemetry-guided / random) is not dominated by either pure arm on:
      - [ ] Sample efficiency (boundary accuracy / FN at fixed FP)
      - [ ] Novel failure discovery rate
    - [ ] Telemetry-guided arm does not create blind spots (anti-self-reinforcement check)
    - [ ] All three arms run under identical compute budget
  - *Expected outcome:* H4 validated -- the telemetry flywheel improves boundary learning without self-deception. If the mixed arm dominates, the system genuinely gets smarter from its own intervention data.
  - *What it means if it fails:* The telemetry creates blind spots (self-reinforcing its own coverage gaps). Would need decorrelation, diversity injection, or a different sampling strategy.

### Track B -- ManiSkill: manipulation thesis

- [ ] **Gate 8a: H1 transfer -- failure critic vs. success policy baseline**
  - *Pass criteria:*
    - [ ] Transfer degradation curves: critic shows gradual degradation vs. success policy cliff under compound shifts
    - [ ] Success baseline: SAC (PPO secondary) trained on same ManiSkill tasks under same compute budget
    - [ ] Same ensemble size (3-5) for fair comparison
    - [ ] Transfer measured under identical shift schedule
    - [ ] Geometric vs. contact-dynamics failure decomposition: measure ratio of transferable-to-non-transferable failure volume
  - *Expected outcome:* H1 validated at least for geometric failures. Contact-dynamics may show weaker advantage (this is a higher-variance bet). Partial validation (geometric-only dominance) is a meaningful result.
  - *What it means if it fails:* Failure-first advantage doesn't hold for the tested task distribution. Either the failure manifold is as high-dimensional as the success manifold (likely for contact-rich tasks), or the baseline is strong enough that the advantage is not statistically significant.

- [ ] **Gate 8b: H5 Pareto frontier (success vs. FN vs. steps)**
  - *Pass criteria:*
    - [ ] Pareto frontier shows failure-first dominates the low-FN region
    - [ ] Tested under compound shifts (factorial / fractional factorial)
    - [ ] Result holds across at least 2 ManiSkill task families
  - *Expected outcome:* Failure-first approach Pareto-dominates in the safety-critical region of the tradeoff space, even if success-first approaches dominate in the high-capability, high-risk region.
  - *What it means if it fails:* The failure-first approach doesn't dominate even in the low-FN region. Would need to revisit the failure critic architecture, the ensemble, or accept that the approach is complementary rather than dominant.

---

## Hypotheses

| ID | Statement | Tested at | Status |
|----|-----------|-----------|--------|
| **H3** | The kernel is mechanically correct: zero L0-R violations, detection SLAs met, no illegal transitions, no flapping | Gates 0-3 | Partially validated (Gates 0-1 pass under deterministic/ideal conditions; Gates 2-3 validate at scale with telemetry) |
| **H2** | Conformal calibration is stable under distribution shift: FN recovers within bounded episodes | Gates 4-5 | Not started |
| **H4** | Telemetry-guided sampling improves boundary learning without creating blind spots | Gate 7 | Not started |
| **H1** | Failure critic transfer degrades gradually vs. success policy cliff under shift | Gate 8a | Not started |
| **H5** | Failure-first dominates the low-FN region of the Pareto frontier | Gate 8b | Not started |

---

## Running the project

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific gate tests
pytest tests/test_gate0_acceptance.py -v    # Gate 0 smoke test
pytest tests/test_gate1_integration.py -v   # Gate 1 SLAs + integration

# Run L0-R latency benchmark
python benchmarks/bench_l0r.py

# Run L1 failure-manifold MVP ablation
.venv/bin/python experiments/run_failure_manifold_ablation.py --quick
```

Requires Python >= 3.10, MuJoCo >= 3.0, NumPy >= 1.26, Gymnasium >= 0.29.

---

## Design doctrine

- **Spec is law.** No code "discovers requirements." If the spec is underspecified, harden the spec first, then write the code.
- **Gates are hard stops.** A failed gate blocks all subsequent work. No working ahead.
- **Determinism is a feature.** Every run is reproducible. Artifacts are hashable. Provenance is product.
- **Assumptions are first-class.** Unknown or partial observability triggers conservative behavior (degrade, revoke), never optimistic continuation.
- **Interfaces freeze early.** The kernel is small, stable, boring. Swap capability above it.
- **Always name the falsifier.** For every claim, write the shortest test or log entry that would prove it wrong.

---

## Known limitations

1. **Single robot, single cell.** Multi-contract composition is deferred.
2. **Proprioceptive only.** No vision. Vision adds a fundamentally different uncertainty profile.
3. **CPU simulation only.** M-series Mac, no GPU. Vision/large ensembles require rented GPU.
4. **Plant fidelity gap (Gates 0-3).** MuJoCo Euler integration at 50Hz overshoots physical limits. This is a diagnostic limitation, not evidence of physical stopping. Re-measurement with nsubsteps >= 10 is planned at Gate 4+.
5. **Success baseline strength.** H1 depends on comparing against the strongest reasonable success policy under the same compute budget. A weak baseline makes the result uninterpretable.
6. **Contact-dynamics uncertainty.** Failure-first advantage is a higher-variance bet for contact-rich manipulation than for geometric-only scenarios. The experimental pipeline measures this distinction rather than assuming it away.

---

## References

- `docs/failure_first_safety_kernel_cumulative_spec.md` -- Complete specification (sections 0-10, appendices A-D)
- `docs/rfcs/RFC-0001-contract-artifact.md` -- Contract canonicalization protocol
- `docs/rfcs/RFC-0002-command-authority-and-telemetry.md` -- Command authority boundary and telemetry 0.2.0
- `docs/co-architect-primer.md` -- Epistemic protocol for co-architect agents (required for ABI/safety/gate decisions)
- `1x_safety_kernel_onepager.md` -- Commercial framing and motivation
- `Robotics Development Challenges.pdf` -- Origin conversation
