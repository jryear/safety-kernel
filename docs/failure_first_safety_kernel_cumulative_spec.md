# Failure‑First Safety Kernel for Robot Deployments
**Unit of operation:** *(robot × cell × assumptions)*  
**Date:** 2026‑02‑06  
**Status:** Cumulative spec (pre‑implementation; gates defined through Track B)

---

## 0. Objective

Prove in simulation that a **failure‑first** safety architecture is:

1. **Mechanically correct** — L0‑R is never violated; L0‑C revocation is deterministic; the contract state machine is legal.
2. **Calibration‑stable under shift** — conformal thresholds adapt; the false‑negative (FN) rate recovers within bounded time.
3. **Meaningfully advantaged for manipulation** — failure models transfer better than success policies where contact dynamics make “success learning” expensive.
4. **Self‑improving without self‑deception** — intervention telemetry improves boundary learning **without** creating blind spots.

---

## 1. Core invariant

> **Untrusted capability under trusted constraint, with authority revocable at runtime.**

- **L0‑R** is physics the robot knows about itself (intrinsic, vendor‑certified).
- **L0‑C** is physics the deployment claims about the cell (relational but certified, integrator‑certified, revocable).
- **L1** is physics the robot learns about the world (relational, uncertainty‑aware, calibrated online).
- **L2** is task competence (swappable, untrusted).
- **RTA** is the weak‑assumption floor (always available).

---

## 2. Layered architecture

### 2.1 L0‑R — Robot intrinsics (vendor‑certified, immutable)
**Scope (intrinsic):**
- joint position/velocity limits
- actuator torque/current limits
- self‑collision manifold
- kinetic energy / momentum caps
- thermal limits

**State uncertainty (set‑valued):**
- maintain a conservative state set \\(\\hat X_t\\) from sensor specs + timing jitter + estimation error
- enforce a **robustly safe** set via erosion:
  - \\(S_{0R,\\text{enforced}} = S_{0R,\\text{physical}} \\ominus \\varepsilon_{\\text{sensor}}\\)

**Enforcement:**
- hard projection on the *current* believed state set (no rollout, no prediction)
- deterministic, < 1ms target

**Predictive shield guard (compile-time):**
- `ENABLE_PREDICTIVE_SHIELD = false` until Gate 6 calibration governance is online
- Predictive shields (barrier certificates, CBF-QP) are diagnostic-only while this flag is false
- Rationale: integration error between kernel prediction and plant realization is not additive state uncertainty — treating it as such (Path 2) contaminates erosion budgets. The kernel is safe because it re-projects from actual state every tick. Predictive enforcement requires calibrated error models (Gate 6+).

#### 2.1.1 L0-R Projection Algorithm (CellArm)

**Action/observation model:**
- L2 speaks joint velocity commands `qdot_cmd` (5D)
- MuJoCo actuators: velocity servos with `ctrlrange` (velocity limits) and `forcerange` (torque limits)
- L0-R sits between L2 and MuJoCo: receives `qdot_cmd`, produces `qdot_exec`

**Sequential projection (ordered constraints):**

1. **Velocity box clamp** — per-joint clamp to eroded velocity limits: `qdot_i = clip(qdot_cmd_i, -qdot_max_eroded_i, +qdot_max_eroded_i)` where `qdot_max_eroded = qdot_max_physical - eps_qdot`.

2. **Position lookahead clamp** — uses the candidate velocity from step 1: `q_next = q + qdot_candidate * dt`. Per-joint clamp: `qdot_i ∈ [(q_min_eroded - q_i) / dt, (q_max_eroded - q_i) / dt]`, where `q_min/max_eroded = q_min/max_physical ± (eps_q + eps_qdot * dt)` (i.e., `q_min_eroded = q_min_physical + eps_q + eps_qdot * dt`, `q_max_eroded = q_max_physical - eps_q - eps_qdot * dt`). After position clamping, re-clamp to eroded velocity limits (position clamp can widen velocity beyond step 1's bounds). **Infeasibility:** if `q_i` is already outside eroded position bounds, the feasible velocity interval is empty on that side — command maximum recovery velocity toward the safe region (clamped to eroded velocity limits) and flag RTA near-miss.

3. **Kinetic energy cap** — `E_k = 0.5 * qdot^T * M(q) * qdot`. If `E_k > E_max_safe`: uniform scaling `qdot *= sqrt(E_max_safe / E_k)`. Energy margin is multiplicative: `E_max_safe = E_max * 0.98`. **Known conservatism:** uniform scaling penalizes low-inertia joints unnecessarily; upgrade to ellipsoidal projection deferred. Log `energy_scaling_factor` in intervention telemetry to quantify performance left on the table.

4. **Self-collision check** — FK at predicted `q_next = q + qdot * dt`, capsule-capsule **surface distance** for active non-adjacent link pairs (see Appendix D for active set). Response: zero velocity on all joints (Gate 0 MVP). Log `collision_pair_id` and `collision_distance` per pair (not just min). Upgrade to directional damping deferred.

**Ordering rationale:** All constraints define sets that are star-convex around `qdot = 0`. Scaling toward zero preserves feasibility of all previously satisfied constraints. Order goes cheapest/most-independent to most-expensive/most-coupled.

**Infeasibility handling:** Handled within step 2 (see above). If any joint is outside eroded bounds at entry, the projection still produces a valid output (recovery velocity) but flags RTA near-miss in the intervention telemetry.

**Output verification (debug assertion):** After step 4, assert that `qdot_exec` satisfies all four constraints. This is a debug-mode assertion, not a runtime cost in production.

**Timing budget:** ~600 ns worst case for 5-DOF (well under 1ms target).

---

### 2.2 L0‑C — Certified cell intrinsics (integrator‑certified, revocable)
**What it is:**
- a **contract artifact** describing static geometry + anchors + monitors + revocation policy
- canonical YAML → deterministic hash → optional signature → runtime verification

**Geometry model (v0.1):**
- convex primitives only: `box`, `cylinder`, `sphere`
- keep‑out zones and (recommended) allowed regions
- all geometry in a single canonical `cell_frame`

**Assumption monitors (bounded time; no learned perception):**
- anchor pose consistency
- safety IO health (E‑stop chain, interlocks, PLC heartbeat)
- sensor health (staleness, calibration checksum, monotonic timestamps)
- robot mount integrity (base transform drift)

#### 2.2.1 Monitor computation semantics

**anchor_pose_consistency:**
- Input: anchor poses from perception (simulated in Track A)
- Predicate: for each required anchor, `||pose_observed - pose_expected||_pos > pos_tol OR ||pose_observed - pose_expected||_rot > rot_tol`
- Aggregation `any_required_out_of_tolerance`: if ANY required anchor fails predicate, monitor fires
- Debounce: monitor must report failure for `debounce_ms` consecutive ms before trigger
- Hysteresis: after trigger clears, monitor must report OK for `hysteresis_ms` continuously before allowing state recovery

**safety_io_health:**
- Input: binary signal value from safety PLC (simulated as boolean)
- Predicate: `signal_value != expected`
- Debounce/hysteresis: as configured (0/0 for E-stop = immediate trigger, no recovery delay)

**sensor_health:**
- Input: sensor telemetry metadata (timestamp, frame age)
- Predicate: `frame_age > max_frame_age_ms OR !monotonic_timestamps OR checksum_mismatch`
- In simulation: inject staleness via shift injector

**robot_mount_integrity:**
- Input: estimated base transform vs. contract-specified base pose
- Predicate: `||base_drift_pos|| > pos_tol OR ||base_drift_rot|| > rot_tol`
- Computed over a rolling window (configurable N recent samples), not instantaneous
- In simulation: inject base transform drift via shift injector

**Revocability:**
- if assumptions fail → **REVOKE** L0‑C and force L1 into full conservative mode
- L0‑R remains active regardless of L0‑C state

---

### 2.3 L1 — Learned relational safety (adaptive; calibrated)
**Objects:**
- failure critic \\(Q_{\\text{fail}}(s,a)\\) and/or barrier \\(h_\\theta(s)\\)
- conservative scoring via ensemble UCB:
  - \\(Q^{\\text{UCB}} = \\mu + k\\sigma\\)

**Relational unsafe set decomposition:**
- \\(U_{\\text{rel}} = U_{\\text{geom}}(G_C) \\cup U_{\\text{non‑geom}}\\)

Where:
- \\(G_C\\) is contract geometry (trusted only when L0‑C ACTIVE)
- \\(U_{\\text{geom}}\\): violating cell geometry constraints
- \\(U_{\\text{non‑geom}}\\): unsafe due to contact/object/human/dynamics

**Two operating modes (contract‑state dependent):**
1) **L0‑C ACTIVE → residual mode**  
   L1 models **only**:
   - \\(U_{\\text{non‑geom}}\\)
   - plus boundary residuals from observation/dynamics mismatch near contract boundaries  
   Contract geometry features are treated as **trusted** inputs.

2) **L0‑C REVOKED → full mode**  
   L1 models **full** \\(U_{\\text{rel}}\\), including geometry.  
   Geometry features are treated as **untrusted** (masked/unknown or high‑uncertainty).

**Governance (online calibration):**
- conformal calibration using near‑violation labels
- FN target \\(\\alpha\\) is a first‑class config

**Near‑violation labels:**
- label \\(y = 1[\\min_{t\\le H} \\text{dist}(s_t, \\partial U) < \\varepsilon]\\)
- \\(\\varepsilon\\) is **percentile‑based per constraint type** and swept

**Calibration cold‑start protocol:**
- initial threshold \\(\\tau_0 = +\\infty\\) (L1 effectively disabled; RTA is the safety floor)
- during warmup: L0‑R + L0‑C + RTA enforce safety; L1 **logs risk scores but does not filter actions**
- warmup criterion: `n_warmup_episodes` completed episodes required before the conformal threshold is meaningful
  - with \\(\\alpha = 0.01\\), default `n_warmup_episodes = 200` (provides sufficient boundary events to estimate and tighten coverage bounds)
- warmup detection: calibration declared "warm" when **both** conditions hold:
  1. at least `n_warmup_episodes` episodes completed
  2. \\(|\\text{FN}_{\\text{empirical}} - \\alpha| \\le\\) `coverage_tolerance` **and** the 95% Clopper‑Pearson confidence interval width on the observed FN rate \\(\\le 2 \\times\\) `coverage_tolerance`
  - this prevents declaring warmth from a handful of events; the CI width shrinks only with sufficient near‑violation samples
  - default `coverage_tolerance = 0.005`
- `n_warmup_episodes` and `coverage_tolerance` are **first‑class config** in the run manifest (see Appendix B)
- post‑warmup: threshold \\(\\tau\\) is live and adapts online per the conformal loop

---

### 2.4 D6 — Critic architecture under contract state (fixed decision)
**MVP decision: shared trunk + dual heads**
- shared trunk \\(f_\\theta(\\cdot)\\)
- residual head (ACTIVE): predicts risk for \\(U_{\\text{non‑geom}}\\) + boundary residuals
- full head (REVOKED): predicts risk for full \\(U_{\\text{rel}}\\), including geometry

**Feature gating rules:**
- ACTIVE: geometry features trusted (low uncertainty)
- REVOKED: geometry features masked to sentinel or treated as observed with high uncertainty

**Handoff semantics:**
- hard switch on state machine transition
- hysteresis window before switching back to residual head
- each head has its own calibration trajectory (separate thresholds)

**Full‑head warmth maintenance:**
- **Problem:** In long deployments where L0‑C is always ACTIVE, the full head's calibration data goes stale. Revocation would trigger a switch to a head that is uncalibrated for the current cell.
- **Training mitigation:** Schedule periodic **synthetic revocation episodes** during training — brief windows where L0‑C is artificially revoked — to keep the full head's calibration current.
- **Synthetic revocation bookkeeping:** Every episode with synthetic revocation **must** set `synthetic_revocation: true` in the `episode_summary` NDJSON event, along with `synthetic_revocation_params` (trigger step, duration, injected monitor). **Evaluation rule:** Gates 5–8 acceptance criteria are computed *excluding* episodes where `synthetic_revocation = true`. These episodes are used *only* for full‑head calibration maintenance.
- **Runtime fallback:** If full head has fewer than `n_warmup_episodes` calibration samples at revocation time, set \\(\\tau_{\\text{full}} = +\\infty\\) (RTA is the effective safety floor until the full head warms up).
- **Telemetry:** Full head calibration staleness (episodes since last full‑head calibration sample) is a logged field per decision event.

---

### 2.5 L2 — Task policy (untrusted, swappable)
- any method (planning, RL, heuristics, teleop)
- proposes \\(a_{\\text{prop}}\\)
- no safety responsibility

---

### 2.6 RTA — Runtime assurance (weak‑assumption floor)
**Triggers:**
- L0‑C violation / revocation
- calibration collapse
- OOD detection
- margin below threshold
- operator command
- L0‑R near‑miss (panic)

**Actions:**
- safe stop
- retreat to safe state / home
- operator handback

---

## 3. Runtime execution semantics (ordering)

At each control tick:

1) **Sense & estimate**
- produce `truth` (sim only), `observed`, and `believed` (kernel internal)

2) **Propose**
- L2 emits `a_prop`

3) **Enforce trusted constraints**
- apply L0‑R projection last (always)
- apply L0‑C constraints **only if contract state ∈ {ACTIVE, DEGRADED}**

4) **Apply learned safety (if enabled)**
- if L1 enabled:
  - evaluate head = residual/full based on contract state
  - compute risk score + calibrated threshold
  - shield / CBF / MPC produces `a_safe`

5) **Finalize**
- L0‑R hard projection produces `a_exec`

6) **Log**
- NDJSON event: truth/observed/believed + action + contract state + calibration state

---

## 4. Contract state machine

States: `ACTIVE`, `DEGRADED`, `REVOKED`, `RTA_STOP`

Illegal transitions **assert‑fail** in validation.

### 4.1 Transition table

| From | Trigger | To | Action |
|------|---------|-----|--------|
| ACTIVE | degrade-severity monitor | DEGRADED | tighten_caps per revocation_policy |
| ACTIVE | revoke-severity monitor | REVOKED | disable L0-C, L1 → full mode |
| ACTIVE | L0-R near-miss (panic) | RTA_STOP | safe stop sequence |
| ACTIVE | rta_stop-severity monitor | RTA_STOP | safe stop sequence |
| DEGRADED | revoke-severity monitor | REVOKED | disable L0-C, L1 → full mode |
| DEGRADED | all degrade monitors clear + hysteresis elapsed | ACTIVE | restore normal caps |
| DEGRADED | L0-R near-miss (panic) | RTA_STOP | safe stop sequence |
| DEGRADED | rta_stop-severity monitor | RTA_STOP | safe stop sequence |
| REVOKED | operator reset + all required monitors pass | ACTIVE | re-enable L0-C, L1 → residual mode |
| REVOKED | L0-R near-miss (panic) | RTA_STOP | safe stop sequence |
| REVOKED | rta_stop-severity monitor | RTA_STOP | safe stop sequence |
| RTA_STOP | operator reset + all required monitors pass | ACTIVE | full restart sequence |

### 4.2 Transition rules

- A **revoke-severity** event transitions directly to REVOKED from any non-RTA_STOP state. There is no requirement to pass through DEGRADED first — the severity level determines the target state.
- An **rta_stop-severity** event transitions directly to RTA_STOP from **any** state (including REVOKED). Both L0-R near-miss and rta_stop-severity monitors are legal RTA_STOP triggers from every row of the table.
- **DEGRADED → ACTIVE** requires hysteresis window to have elapsed AND all degrade-severity monitors clear. Without both conditions, transition is illegal.
- **REVOKED → ACTIVE** and **RTA_STOP → ACTIVE** require explicit operator reset. Auto-recovery from REVOKED is illegal.

### 4.3 Flapping definition

A state machine **flap** is a pair of transitions A→B→A within the hysteresis window. Max `N=4` flaps per episode means max 4 such pairs. After the 4th, force transition to RTA_STOP.

---

## 5. Telemetry & provenance (NDJSON)

**Three‑channel logging per decision:**
- `truth`: privileged sim state (ground truth margins, contacts, distances)
- `observed`: noisy sensor channel (may equal truth only in Gates 0–3 Track A MVP; starting at Gate 4, proprioceptive `observed ≠ truth`)
- `believed`: kernel belief at decision time (state set, risk, threshold, head, contract state)

**Append‑only integrity:**
- `prev_hash` chain per event
- per‑episode CRC32 summary
- run manifest includes: seed, contract hash, env config hash, model checkpoint hash

---

## 6. Shift injector

Shift factors:
- geometry drift (anchors / keep‑out displacement)
- sensor degradation (latency/jitter/dropout/calibration mismatch)
- dynamics perturbation (friction/mass/compliance proxies)
- task distribution shift (poses/clutter/targets)
- **proprioceptive noise** (additive Gaussian noise on joint position, velocity, and torque observations; parameterized by \\(\\sigma\\) per channel from sensor datasheets or swept)

Composition policy:
- **Track A:** single‑factor only (diagnostic / causality isolation)
  - **Proprioceptive noise is required for Gates 4–7.** Starting at Gate 4, `observed ≠ truth` for proprioception. This ensures calibration thresholds are stressed against deployment‑like sensor conditions. Noise levels are swept as a shift factor.
- **Track B:** single‑factor + compound (factorial / fractional factorial)

---

## 7. Experimental gates (sequential pipeline)

**Track A (Safety‑Gymnasium): kernel mechanics + calibration machinery**
- Gate 0: L0‑R hard projection
- Gate 1: L0‑C schema + monitors + shift injector
- Gate 2: NDJSON telemetry + H3 checker
- Gate 3: H3 mechanical validation (no L1)
- Gate 4: conformal calibration loop (+ ε sweep) **+ cold‑start protocol validation**
  - Acceptance (added): calibration reaches warmth within `n_warmup_episodes` under stationary conditions with proprioceptive noise active
- Gate 5: calibration under shift (H2)
- Gate 6: L1 predictive safety — **critic vs. barrier comparison**
  - Arm A: failure critic \\(Q_{\\text{fail}}\\) with threshold‑based shielding
  - Arm B: learned barrier \\(h_\\theta(s)\\) with CBF‑QP projection
  - Acceptance: both arms enforce correctly; Arm A shows dual‑head separation + smooth handoff; Arm A + Arm B each meet < 10 ms decision time; comparison data logged for architectural decision
- Gate 7: H4 telemetry vs random (3 arms + novel failure discovery)

**Track B (ManiSkill): manipulation thesis**
- Gate 8a: H1 transfer — failure critic vs. **success policy baseline** under compound shifts
  - Success baseline: SAC trained on same ManiSkill tasks under same compute budget (PPO as secondary if SAC is unstable)
  - Same ensemble size (3–5) for fair comparison
  - Transfer measured under identical shift schedule
- Gate 8b: H5 Pareto frontier (success vs FN vs steps)

---

## 8. Acceptance criteria highlights (machine‑checkable)

### H3 (mechanical correctness)
All must hold:
- (a) **Zero L0‑R violations** across all Track A runs
- (b) **Detection SLA** per monitor type (configured once loop rate is fixed)
- (c) **No illegal state transitions**; every transition must have a logged trigger
- (d) **No flapping** beyond configured threshold (default N=4 per episode)

#### 8.1 H3 checker predicates (machine-checkable)

**(a) Zero L0-R violations (kernel output check):**
- Predicate: for every step in every episode, `qdot_exec` satisfies all L0-R constraints:
  - `|qdot_exec[i]| ≤ qdot_max_eroded[i]` (velocity limits after erosion)
  - `q[i] + qdot_exec[i] * dt ∈ [q_min_eroded, q_max_eroded]` (position lookahead)
  - `0.5 * qdot_exec^T * M(q) * qdot_exec ≤ E_max_safe` (energy cap)
  - self-collision distance at predicted `q_next` ≥ d_safe (or qdot_exec == 0)
- Tolerance: floating-point ε = 1e-8 (smaller than sensor erosion by orders of magnitude)
- This checks **kernel outputs** (`qdot_exec`), not plant realized state — the kernel is responsible for what it commands, not for MuJoCo integration drift
- **Plant tracking diagnostic** (logged, not a gate blocker): additionally log `truth.q`, `truth.qdot`, `truth.E_k` violations as a diagnostic metric quantifying plant-integrator gap. Non-zero values are expected and informative but do not fail the gate.

**(a.gate4) Plant fidelity criterion (Gate 4+ only):**
- With `nsubsteps ≥ 10` active (see Appendix D.11), per-joint p99 tracking errors must fall below the within-tick margin budget:
  - `p99(|q_realized - q_predicted|) < eps_q` (joint position)
  - `p99(|qdot_realized - qdot_predicted|) < eps_qdot` (joint velocity)
- Exact thresholds TBD at Gate 4 based on empirical measurement with nsubsteps active
- This criterion validates that the simulated plant resembles a servo system (tracking the commanded velocity closely enough that within-tick prediction is meaningful)

**(b) Detection SLA:**
- Per monitor type: when a shift is injected that violates the monitor's predicate, the state machine must transition within `debounce_ms + 1 control tick` of the first violation
- Predicate: `transition_step - first_violation_step ≤ ceil(debounce_ms / (1000/loop_rate_hz)) + 1`
- 100% of injected violations must be detected within SLA
- Concrete SLA values from Track A contract (50 Hz loop rate):
  - anchors_consistent: ceil(250 / 20) + 1 = 14 steps
  - gate_interlock_ok: ceil(0 / 20) + 1 = 1 step
  - estop_chain_ok: ceil(0 / 20) + 1 = 1 step
  - front_cam_health: ceil(500 / 20) + 1 = 26 steps
  - robot_mount_integrity: ceil(250 / 20) + 1 = 14 steps

**(c) No illegal transitions:**
- Predicate: every `contract_state_change` event has `(from_state, trigger_severity) → to_state` matching the transition table from Section 4.1
- Checked by scanning all `contract_state_change` events in the NDJSON log

**(d) No flapping:**
- Predicate: count state machine flaps (A→B→A pairs within hysteresis window) per episode ≤ N (default 4)
- If exceeded, the episode must end in RTA_STOP (the flap limiter fires)

### H2 (calibration stability)
- **cold‑start:** calibration reaches warmth (see Section 2.3 warmup detection criteria) within `n_warmup_episodes`; this is a configured bound, not unbounded
- pre‑shift FN ≤ α (post‑warmup, with proprioceptive noise active)
- post‑shift FN spikes then recovers to ≤ α within bounded episodes
- threshold trajectory adapts (not stuck)

### H4 (telemetry flywheel integrity)
- mixed (80/20) arm is not dominated by telemetry‑only or random‑only on both:
  - sample efficiency (boundary accuracy / FN at fixed FP)
  - novel failure discovery rate

### H1/H5 (manipulation thesis)
- transfer degradation curves show critic: gradual vs policy: cliff (H1)
- Pareto frontier shows failure‑first dominates low‑FN region (H5), including compound shifts

**Scope and interpretation:**
- H1 measures the **ratio of transferable‑to‑non‑transferable failure volume** under shift. The claim is empirical, not structural.
- The failure‑first advantage is expected to be strongest for geometric/kinematic failures (collision, joint limits, energy bounds) where the failure manifold is lower‑dimensional and physics‑stable.
- For contact‑dynamics failures (stick‑slip, deformable contact, grasp stability), the failure manifold may share the high‑dimensionality and sim‑to‑real sensitivity of success policies. H1/H5 are therefore **higher‑variance bets** for contact‑rich manipulation than for geometric‑only scenarios.
- If Pareto dominance holds only for the geometric subset of failures, that is a **partial validation worth reporting**, not a failure of the experimental design. The experiment is designed to detect this distinction.

---

## 9. Compute envelope (M‑series Mac default)
- proprioceptive observations only
- ensemble size: 3–5
- shallow MLP: 64‑64‑1
- short horizon H: 10–20 steps
- CPU sim: Safety‑Gymnasium + ManiSkill (CPU mode)
- expand to vision/large ensembles only with rented GPU

---

## 10. Thesis scope, novelty, and limitations

### 10.1 Novelty decomposition
CBFs, conformal prediction, ensemble uncertainty, and safe RL all exist as independent techniques. **The novelty is in the composition:**
1. **Failure‑first inversion** — failure critic as primary learned artifact, not a bolt‑on constraint
2. **Layered trust hierarchy** — L0‑R / L0‑C / L1 / L2 with epistemic‑status‑based separation
3. **Contract schema with cryptographic provenance** — L0‑C as a certified, hashable, revocable deployment artifact
4. **Anti‑self‑reinforcement measurement** — H4 explicitly tests whether the telemetry flywheel creates blind spots

No single component is novel in isolation. The claim is that the composition produces a system with properties none of the individual techniques achieve alone.

### 10.2 Competitive landscape (as of 2026‑02)
- **Figure AI** — Created safety and reliability teams in 2025; no published safety architecture.
- **Tesla Optimus** — FSD‑derived neural networks with force limiting and "localized safety chip." Success‑first architecture.
- **1X NEO** — Most safety‑conscious hardware design (tendon‑driven compliance, soft exterior, human‑in‑the‑loop). Passive hardware + human oversight, not learned safety.
- **Boston Dynamics** — Long history of robust control; no published formal safety kernel with learned failure critics or contract state machines.

**No major humanoid robotics company is publishing formal safety kernels, contract state machines, or failure‑first learning.** The architecture is positioned against a landscape where formal safety guarantees are underweighted relative to demonstration capability.

### 10.3 Thesis scope
The failure‑first transfer advantage (H1) is a **strong claim for geometric/kinematic failures** — collision, joint limits, energy bounds — where the failure manifold is lower‑dimensional, physics‑stable, and well‑modeled in simulation.

For **contact‑rich manipulation failures** (stick‑slip transitions, deformable contact, grasp stability under perturbation), the claim is weaker. These failure modes share the high‑dimensionality and sim‑to‑real sensitivity of success policies. The failure manifold for a complex grasp may not be meaningfully lower‑dimensional than the success manifold for the same task.

This does not invalidate the project. It means:
- H1/H5 are **higher‑variance bets** than the geometric‑only case.
- The aggregate advantage depends on the **ratio of geometric‑to‑contact failure volume** in the target task distribution.
- The experimental pipeline (Track B) is designed to **measure** this ratio, not assume it away.
- A partial validation (dominance for geometric failures only) is a meaningful result that constrains the architecture's deployment profile.

### 10.4 Known limitations (carried forward)
1. **Sensor gap partially closed:** Proprioceptive noise injection is now required for Gates 4–7, but full sensor models (vision noise, latency profiles) remain out of scope for Track A.
2. **Single robot × single cell:** Multi‑contract composition is deferred.
3. **No vision:** All observations proprioceptive. Vision adds a fundamentally different uncertainty profile.
4. **Success policy baseline strength:** H1 depends on comparing against the strongest reasonable success policy under the same compute budget. A weak baseline would make the result uninterpretable.
5. **Full‑head cold start:** Addressed via synthetic revocation + runtime fallback, but the effectiveness of this mitigation is itself an empirical question tested at Gate 6.

---

# Appendix A — L0‑C Contract YAML (v0.1 template)

> This is a complete **template** you can canonically hash. Fill the `integrity.body_sha256` after canonicalization.

```yaml
schema_version: "0.1.0"

metadata:
  contract_id: "c8c7c2f0-7a0a-4d7e-9a3c-2f6c6f5c2c1a"
  cell_id: "cell_demo_01"
  created_at: "2026-02-06T00:00:00Z"
  created_by: "integrator:acme_automation"
  description: "Demo cell contract: primitives + anchors + monitors + revocation policy"
  robot_compat:
    robot_model: ["ur10e"]
    tool_flange: "ISO_9409_1_50_4_M6"
    payload_class_kg: [0.0, 5.0]
  validity:
    valid_from: "2026-02-06T00:00:00Z"
    expires_at:  null

frames:
  units: {length: "m", angle: "rad"}
  cell_frame:
    name: "cell"
    convention: "right_handed"
  robot_base_in_cell:
    xyz: [1.20, 0.35, 0.00]
    rpy: [0.00, 0.00, 1.57079632679]
    tolerance:
      pos_m: 0.005
      rot_rad: 0.01
  tool_frame:
    name: "tcp"
    xyz: [0.0, 0.0, 0.20]
    rpy: [0.0, 0.0, 0.0]

geometry:
  keepouts:
    - name: "fence_north"
      type: "box"
      pose_in_cell: {xyz: [0.0, 1.20, 0.90], rpy: [0.0, 0.0, 0.0]}
      dims: {x: 3.0, y: 0.05, z: 1.8}
      tolerance: {inflate_m: 0.02}
      enforcement:
        hard_keepout: true

    - name: "fixture_keepout"
      type: "cylinder"
      pose_in_cell: {xyz: [0.75, 0.40, 0.40], rpy: [0.0, 0.0, 0.0]}
      dims: {radius: 0.18, height: 0.80}
      tolerance: {inflate_m: 0.01}
      enforcement:
        hard_keepout: true

  allowed_regions:
    - name: "work_envelope"
      type: "box"
      pose_in_cell: {xyz: [0.70, 0.40, 0.85], rpy: [0.0, 0.0, 0.0]}
      dims: {x: 1.40, y: 1.20, z: 1.70}
      tolerance: {deflate_m: 0.02}
      enforcement:
        hard_keepout: false
        caps_when_active:
          max_tcp_speed_mps: 0.60

anchors:
  - name: "tag_A"
    type: "fiducial"
    sensor: "camera_front"
    expected_pose_in_cell: {xyz: [0.20, 0.10, 1.00], rpy: [0.0, 0.0, 0.0]}
    tolerance: {pos_m: 0.01, rot_rad: 0.03}
    required: true

  - name: "tag_B"
    type: "fiducial"
    sensor: "camera_front"
    expected_pose_in_cell: {xyz: [1.50, 0.10, 1.00], rpy: [0.0, 0.0, 0.0]}
    tolerance: {pos_m: 0.01, rot_rad: 0.03}
    required: true

assumption_monitors:
  - name: "anchors_consistent"
    type: "anchor_pose_consistency"
    inputs: ["camera_front"]
    params:
      anchors: ["tag_A", "tag_B"]
      aggregation: "any_required_out_of_tolerance"
    debounce_ms: 250
    hysteresis_ms: 500
    severity: "revoke"
    action: "revoke_L0C_enable_L1_full"

  - name: "gate_interlock_ok"
    type: "safety_io_health"
    inputs: ["safety_plc"]
    params:
      signal: "gate_closed"
      expected: true
    debounce_ms: 0
    hysteresis_ms: 0
    severity: "rta_stop"
    action: "safe_stop"

  - name: "estop_chain_ok"
    type: "safety_io_health"
    inputs: ["safety_plc"]
    params:
      signal: "estop_ok"
      expected: true
    debounce_ms: 0
    hysteresis_ms: 0
    severity: "rta_stop"
    action: "safe_stop"

  - name: "front_cam_health"
    type: "sensor_health"
    inputs: ["camera_front"]
    params:
      max_frame_age_ms: 100
      monotonic_timestamps: true
      calibration_checksum: "FILL_ME"
    debounce_ms: 500
    hysteresis_ms: 1000
    severity: "degrade"
    action: "tighten_caps"

  - name: "robot_mount_integrity"
    type: "robot_mount_integrity"
    inputs: ["robot_proprioception"]
    params:
      base_pose_drift_pos_m: 0.005
      base_pose_drift_rot_rad: 0.01
    debounce_ms: 250
    hysteresis_ms: 500
    severity: "revoke"
    action: "revoke_L0C_enable_L1_full"

revocation_policy:
  on_degrade:
    l0c_still_enforced: true
    caps:
      max_tcp_speed_mps: 0.20
      restrict_to_allowed_regions: true
  on_revoke:
    l0c_enabled: false
    l1_mode: "full_conservative"
  on_rta_stop:
    sequence: ["controlled_stop", "retreat_to_home", "operator_handoff"]

integrity:
  canonicalization:
    sort_keys: true
    float_quantization: 1.0e-6
    normalize_newlines: true
    unit_normalization: true
  body_sha256: "TBD_AFTER_CANONICALIZATION"
  signature: null
  signer_id: "integrator:acme_automation"
  signed_at: null
```

---

# Appendix B — Minimal run manifest (YAML)

```yaml
run_id: "run_20260206_001"
seed: 42
track: "A"
env: "safety_gymnasium:PointGoal1"
loop_rate_hz: 50
horizon_H_steps: 15

contract:
  contract_id: "c8c7c2f0-7a0a-4d7e-9a3c-2f6c6f5c2c1a"
  contract_body_sha256: "..."
  contract_file: "cell_contract.yaml"

models:
  critic_arch: "shared_trunk_dual_head"
  ensemble_size: 3
  checkpoint_sha256: "..."

l1_enforcement:
  arm: "critic"  # "critic" (Arm A) or "barrier_qp" (Arm B)
  critic_config:
    ucb_k: 2.0
    threshold_enforcement: "hard"
  barrier_qp_config:
    solver: "osqp"
    solver_time_budget_ms: 5.0
    barrier_type: "learned"
    cbf_class: 1
    relaxation: "soft_constraint"
    relaxation_penalty: 1000.0

calibration:
  target_fn_alpha: 0.01
  epsilon_percentiles: {obstacle_distance: [1,3,5,10,15], joint_margin: [1,3,5,10,15], energy_margin: [1,3,5,10,15]}
  conformal_variant: "online"
  cold_start:
    n_warmup_episodes: 200
    coverage_tolerance: 0.005
    initial_tau: "+inf"

proprioceptive_noise:
  enabled: true
  sigma_joint_pos_rad: 0.001
  sigma_joint_vel_radps: 0.01
  sigma_torque_nm: 0.5

logging:
  ndjson_path: "events.ndjson"
  hash_chain: "sha256_prev_hash"
  episode_crc32: true
```

---

# Appendix C — NDJSON Event Schemas (full field definitions)

Seven event types, one per line in the NDJSON log. All events share the `prev_hash` chain for integrity. Every event carries `event_type` as the first field for dispatch.

## C.1 `safety_intervention`

Emitted when a safety layer **actually modifies** the command (not on pass-through). Use `layer` to disambiguate which layer intervened. L0-C emits only when `scaling_factor < 1.0`. L0-R emits when `constraints_active` is non-empty. Layer-specific fields go in `layer_info` to keep the top-level schema stable across layers.

```
event_type            string   "safety_intervention"
timestamp_ns          int64    simulated wall-clock: int(data.time * 1e9)
episode_id            string
step                  int
layer                 enum     "L0R" | "L0C" | "L1"
prev_hash             string   SHA256 hex of previous event's canonical JSON bytes
qdot_cmd              float[5] input to this layer (L2 output for L0C, L0C output for L0R)
qdot_exec             float[5] output of this layer
constraints_active    list[string]  names of constraints that modified qdot_cmd
margin_to_L0R         dict     per-constraint minimum margin (float) — L0R only, {} for L0C
energy_ratio          float    E_k / E_max — L0R only, 0.0 for L0C
energy_scaling_factor float    1.0 if no scaling applied — L0R only
collision_pairs       list[{pair_id: string, distance: float}]  L0R only, [] for L0C
rta_nearmiss          bool     true if infeasibility triggered recovery velocity — L0R only
layer_info            dict     layer-specific fields (see below)
truth                 object   {q: float[5], qdot: float[5], tau: float[5],
                               tcp_to_keepout_distances: dict[string, float],
                               self_collision_surface_distances: dict[string, float]}
observed              object   {q: float[5], qdot: float[5], tau: float[5]}
believed              object   {q_interval: float[5][2], qdot_interval: float[5][2]}
```

**layer_info for L0C:**
```
tcp_speed_before      float    ||J(q) @ qdot_cmd|| before clamping
tcp_speed_after       float    ||J(q) @ qdot_exec|| after clamping
scaling_factor        float    v_max / speed (< 1.0 when emitted)
```

**layer_info for L0R:** `{}` (all L0R-specific fields are top-level)

## C.2 `contract_state_change`

```
event_type            string   "contract_state_change"
timestamp_ns          int64
episode_id            string
step                  int
prev_hash             string
from_state            enum     "ACTIVE" | "DEGRADED" | "REVOKED" | "RTA_STOP"
to_state              enum     "ACTIVE" | "DEGRADED" | "REVOKED" | "RTA_STOP"
trigger_monitor       string   monitor name, "l0r_nearmiss", "operator_reset", or "flap_limiter"
trigger_severity      enum     "degrade" | "revoke" | "rta_stop" | "clear" | "reset"
monitors_status       dict     {monitor_name: {"triggered": bool, "violated": bool}}
```

## C.3 `shift_injected`

```
event_type            string   "shift_injected"
timestamp_ns          int64
episode_id            string
step                  int     (or -1 for episode-level shifts)
prev_hash             string
shift_type            string   "anchor_drift" | "safety_io_flip" | "sensor_stale" | "base_drift"
shift_params          dict     type-specific parameters
affected_channels     list[string]  e.g. ["anchor_poses", "safety_io", "sensor_health", "base_pose_estimate"]
```

## C.4 `episode_summary`

```
event_type            string   "episode_summary"
episode_id            string
seed                  int
contract_hash         string
prev_hash             string
total_steps           int
total_interventions   dict     per-layer count: {"L0R": int, "L0C": int, "L1": int}
state_transitions     list[{step: int, from: string, to: string, trigger: string}]
final_state           enum     "ACTIVE" | "DEGRADED" | "REVOKED" | "RTA_STOP"
synthetic_revocation  bool
synthetic_revocation_params  object | null  {trigger_step, duration_steps, injected_monitor}
l1_arm                enum     "critic" | "barrier_qp" | "none"
full_head_staleness_episodes  int
min_margin_to_L0R     float    minimum across all steps
max_energy_ratio      float    maximum across all steps
fn_count              int      (if calibration active, else 0)
fp_count              int      (if calibration active, else 0)
episode_crc32         string   CRC32 of all events in this episode
```

## C.5 `run_start`

First event in every run. Anchors the hash chain.

```
event_type            string   "run_start"
timestamp_ns          int64    0 (run start)
prev_hash             string   "0000000000000000000000000000000000000000000000000000000000000000" (64 zeros)
manifest_hash         string   SHA256 hex of canonical run manifest JSON bytes
schema_version        string   "0.1.0"
contract_hash         string   body_sha256 of loaded contract
total_episodes        int      planned episode count
```

## C.6 `kernel_step_summary`

Emitted **every step**, fixed schema. The high-volume "black box flight recorder." Contains only scalar summaries — full vectors go in `safety_intervention` events.

```
event_type            string   "kernel_step_summary"
timestamp_ns          int64    int(data.time * 1e9)
episode_id            string
step                  int
prev_hash             string
contract_state        enum     "ACTIVE" | "DEGRADED" | "REVOKED" | "RTA_STOP"
l0r_intervened        bool     len(constraints_active) > 0
l0c_active            bool     L0-C enforcement ran (DEGRADED state)
l0c_intervened        bool     L0-C actually scaled (scaling_factor < 1.0)
l0c_scaling_factor    float    1.0 if no scaling
energy_ratio          float    E_k / E_max from L0-R
min_margin_to_L0R     float    min across all L0-R constraint margins
rta_nearmiss          bool     from L0-R
min_collision_distance float   min across all collision pairs (if computed)
min_keepout_distance  float    min across all keepout distances (if computed)
```

## C.7 Hash chain protocol

- The entire run is a **single continuous chain**. Deletion or reordering of events (including across episode boundaries) is detectable.
- First event in run: `run_start` with `prev_hash = "0"*64` (64-zero sentinel).
- Every subsequent event: `prev_hash = SHA256(canonical_json_bytes_of_prev_event)`.
- Episode boundaries are marked by `episode_summary` events within the chain but do **not** break it.
- Each event's `prev_hash` refers to the immediately preceding event in the NDJSON file, regardless of episode.

## C.8 NDJSON serialization rules

- **Timestamps:** simulated wall-clock — `timestamp_ns = int(data.time * 1e9)`. No real-time clock dependency. Deterministic across runs for identical seeds.
- **Canonical JSON:** `json.dumps(obj, sort_keys=True, separators=(',', ':'))` — sorted keys, no whitespace. Both `sort_keys` and compact `separators` are required for hash chain reproducibility.
- **Float quantization:** 6 decimal places for all floating-point values (`round(x, 6)`). Matches 1e-6 quantization. Prevents platform-dependent serialization drift.
- **Hash input:** the canonical JSON bytes of the event (UTF-8 encoded), **excluding** the trailing newline. `prev_hash = SHA256(canonical_json_bytes_of_prev_event)`.
- **One event per line**, UTF-8 encoded, newline-terminated (`\n`), no trailing comma.

## C.9 Deterministic event ordering within a step

Within a single step `k`, events are emitted in this fixed order:

1. `shift_injected` — if any shift begins or ends at step `k`
2. `safety_intervention` layer="L0C" — if L0-C intervened (scaling_factor < 1.0)
3. `safety_intervention` layer="L0R" — if L0-R intervened (constraints_active non-empty)
4. `contract_state_change` — if a state transition occurred
5. `kernel_step_summary` — always (every step)

This ordering makes replay and reasoning deterministic.

## C.10 Three-channel serialization

- `truth`: ground-truth `q`, `qdot`, `tau`, per-keepout TCP-point → box surface signed distances (analytical), per-collision-pair capsule surface distances
- `observed`: noisy `q`, `qdot`, `tau` (= truth in Gates 0-3, noisy in Gates 4+)
- `believed`: interval `[q_min_believed, q_max_believed]`, `[qdot_min_believed, qdot_max_believed]` per joint (the eroded state set at decision time)

## C.11 Manifest validation rules

- If `l1_enforcement.arm == "critic"`, only `critic_config` is used; `barrier_qp_config` is ignored (and vice versa).
- Both config blocks are present in the manifest for schema completeness.
- `l1_arm` in every `episode_summary` event must match the run manifest's `l1_enforcement.arm`.

---

# Appendix D — CellArm 5-DOF Environment Specification (Track A)

This appendix defines the concrete 5-DOF arm environment for Track A experiments. All numerical values in this appendix are authoritative — the Track A contract (A5), MJCF model (C1), and environment wrapper (C2) derive from this appendix.

## D.1 Kinematic chain — joint axis configuration

| Joint | Name | Axis | Type |
|-------|------|------|------|
| 1 | base | Z | yaw |
| 2 | shoulder | Y | pitch |
| 3 | elbow | Z | yaw |
| 4 | wrist_pitch | Y | pitch |
| 5 | wrist_roll | Z | yaw |

Z-Y-Z-Y-Z pattern: gives 3D reach (not confined to a plane) and allows distal links to fold back toward proximal links, creating meaningful self-collision scenarios between links 0-1 and links 3-5.

## D.2 Link geometry (capsules)

| Link | Name | Radius (m) | Length (m) | Mass (kg) | Notes |
|------|------|-----------|-----------|----------|-------|
| 0 | base | 0.06 | 0.15 | — | fixed mount |
| 1 | upper_arm | 0.05 | 0.30 | 4.0 | proximal, heaviest |
| 2 | forearm_proximal | 0.045 | 0.25 | 3.0 | |
| 3 | forearm_distal | 0.04 | 0.20 | 2.0 | |
| 4 | wrist | 0.035 | 0.15 | 1.5 | |
| 5 | end_effector | 0.03 | 0.10 | 1.0 | distal, lightest |

- **Max reach:** ~1.00m (sum of link lengths)
- **Total moving mass:** 11.5 kg

## D.3 Joint limits

| Joint | Position range (rad) | Velocity range (rad/s) | Torque limit (Nm) |
|-------|---------------------|----------------------|-------------------|
| 1 (base) | ±π | ±π | 50 |
| 2 (shoulder) | ±π | ±π | 50 |
| 3 (elbow) | ±π | ±π | 30 |
| 4 (wrist_pitch) | ±π | ±π | 15 |
| 5 (wrist_roll) | ±π | ±π | 10 |

## D.4 Energy cap

- **E_max** is a **tunable kernel budget**, not a fixed ISO constant. The ISO/TS 15066 transient contact limit (2.3 J for chest/abdomen) motivates the concept but is not directly applicable to joint-space kinetic energy (it applies to effective mass at contact point × impact velocity, which depends on configuration).
- Default **E_max = 5.0 J** (empirical starting point — at this arm's mass/speed profile, 2.3 J would produce near-constant clamping). Adjust up/down to hit the intervention band.
- **E_max_safe = E_max × 0.98** (multiplicative margin)
- The run manifest records the exact E_max used; it is not hardcoded.

## D.5 Sensor uncertainty (Minkowski erosion parameters)

| Parameter | Value | Description |
|-----------|-------|-------------|
| eps_q | 0.001 rad | joint position encoder uncertainty |
| eps_qdot | 0.01 rad/s | joint velocity estimation uncertainty |
| eps_tau | 0.5 Nm | torque estimation uncertainty |

## D.6 Self-collision pairs (active set)

Determined from geometry — non-adjacent pairs where the arm's kinematics can bring links close:

| Pair | Link A | Link B |
|------|--------|--------|
| 1 | link_0 (base) | link_3 (forearm_distal) |
| 2 | link_0 (base) | link_4 (wrist) |
| 3 | link_0 (base) | link_5 (end_effector) |
| 4 | link_1 (upper_arm) | link_4 (wrist) |
| 5 | link_1 (upper_arm) | link_5 (end_effector) |

- Distance metric: **surface distance** exclusively (minimum distance between capsule surfaces, not center-to-center). A surface distance of 0 means contact.
- Minimum safe distance: **d_safe = 0.02m** (surface-to-surface clearance margin)

## D.7 Cell geometry

- Robot base at cell origin `[0, 0, 0]`, mounted on ground plane
- **Allowed region:** box centered at `[0.4, 0.0, 0.5]`, dims `[0.8, 0.8, 1.0]` — covers ~70% of reachable workspace
- **Keepout 1** ("fixture_column"): box at `[0.6, 0.35, 0.3]`, dims `[0.2, 0.2, 0.6]` — fixture column inside workspace
- **Keepout 2** ("back_wall"): box at `[-0.3, 0.0, 0.5]`, dims `[0.1, 0.8, 1.0]` — wall behind arm
- ~30% of uniformly sampled Cartesian goals within allowed region would require passing near a keepout

## D.8 Observation and action spaces

**Observation space** (proprioceptive only):
- `q` — joint positions (5D)
- `qdot` — joint velocities (5D)
- `tau` — actuator torques (5D, read from MuJoCo)
- Total: **15D float vector**

**Action space:**
- `qdot_cmd` — joint velocity commands (5D), continuous, range ±π rad/s

## D.9 L2 policy (PD reaching controller, Track A)

- **Goals:** sampled uniformly within the allowed region (Cartesian)
- **Controller:** Jacobian pseudoinverse velocity mapping: `qdot_cmd = J_pinv @ (K_p * (x_goal - x_tcp))`
- **Gains:** configurable in run manifest. Default K_p tuned to produce **5-20% intervention rate** (fraction of steps where L0-R modifies qdot_cmd). Start with K_p = 2.0, adjust empirically during C5 smoke test.
- **Goal hold time:** 50 steps at goal (||x_tcp - x_goal|| < 0.05m), then resample
- **Purpose:** exercise the safety kernel with realistic reaching behavior — not too conservative (never triggers interventions) and not too aggressive (every step clamped)

## D.10 Control rate

**50 Hz** (dt = 0.02s)

## D.11 MuJoCo substeps (nsubsteps)

**Gates 0-3:** `nsubsteps = 1` (default Euler integration). Known plant tracking gap: MuJoCo integrates gravity, Coriolis, and contact forces within each timestep, causing the realized state to diverge from the kernel's one-step velocity prediction. Per-joint p99 position error ~0.231 rad (231x sensor erosion), velocity error ~11.57 rad/s (1157x sensor erosion). This is a plant fidelity issue, not a safety concern — the kernel re-projects from actual measured state every tick.

**Gates 4+:** `nsubsteps ≥ 10` (configurable in run manifest). With 10 substeps, MuJoCo takes 10 integration steps per control tick, producing a plant that tracks commanded velocities much more closely (resembling a real servo system). This is required for predictive components (L1 critic, barrier certificates) where within-tick prediction accuracy matters.

**Key invariant:** `nsubsteps` changes nothing about the kernel — only the plant accuracy. L0-R projection, monitors, and state machine are independent of `nsubsteps`. The kernel always operates on the actual measured state, not predictions.
