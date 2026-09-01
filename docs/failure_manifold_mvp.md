# Failure-Manifold Critic MVP

This experiment track adds an L1 learned boundary on top of the existing hard
safety stack. It does not replace L0-R projection or L0-C contract monitors.

## Claim Shape

The MVP claim is narrow:

> Given deterministic rollouts from a PPO-style policy interface, can a learned
> failure critic predict sparse L0-R boundary labels, and how do erosion depth
> and conformal coverage trade off false positives, false negatives,
> empirical coverage, intervention rate, and shift-injection SLA hits?

This is not a claim that the policy outperforms PPO, nor that the critic is a
standalone safety layer.

## Pipeline

1. Run deterministic CellArm rollouts through the Gymnasium/MuJoCo environment.
2. For each state-action pair, label the proposed action with the existing L0-R
   oracle:
   - unsafe if L0-R modifies the action by more than a physically meaningful
     tolerance (`0.05 rad/s` by default),
   - unsafe if the post-projection boundary margin is at or below the configured
     threshold,
   - unsafe if the state is an RTA near-miss.
3. Optionally run L0-C shift injection in the same rollout and count whether
   injected faults are detected by contract transitions.
4. Fit a deterministic NumPy logistic critic over `(q, qdot, tau, action)`.
5. Calibrate the risk threshold on held-out unsafe examples with conformal
   coverage using thinned samples by default. Raw sample and episode-block
   calibration are also supported.
6. Sweep erosion scale and conformal coverage.
7. Emit CSV/JSON metrics and SVG plots.

## Run

Quick smoke run:

```bash
.venv/bin/python experiments/run_failure_manifold_ablation.py --quick
```

Full default grid:

```bash
.venv/bin/python experiments/run_failure_manifold_ablation.py \
  --out experiments/results/failure_manifold \
  --compare-calibration-units
```

Old-actor/new-label attribution run:

```bash
.venv/bin/python experiments/run_failure_manifold_ablation.py \
  --policy-kind legacy_velocity \
  --out experiments/results/failure_manifold_old_actor_new_label
```

Outputs:

- `metrics.csv`
- `diagnostics.csv`
- `calibration_unit_comparison.csv` when `--compare-calibration-units` is used
- `summary.json`
- `plots/false_positive_rate.svg`
- `plots/false_negative_rate.svg`
- `plots/intervention_rate.svg`
- `plots/sla_hit_rate.svg`
- `plots/empirical_unsafe_coverage.svg`
- `plots/coverage_gap.svg`
- `plots/roc_auc.svg`
- `plots/pr_auc.svg`

Read `diagnostics.csv` before interpreting the metric plots. It reports
component base rates for each split and erosion scale:

- `intervention_gt_tol_rate`
- `boundary_margin_rate`
- `near_miss_rate`
- per-constraint activation rates
- intervention magnitude quantiles
- margin quantiles

## Metrics

- **Positive rate:** reported outcome of the physical label rule. It is not an
  acceptance target.
- **False positive rate:** critic flags unsafe when the L0-R oracle would pass.
- **False negative rate:** critic passes when the L0-R oracle would intervene.
- **Intervention rate:** fraction of proposed actions the critic would shield.
- **Empirical unsafe coverage:** recall on unsafe oracle labels at the chosen
  conformal threshold.
- **Coverage gap:** target coverage minus empirical unsafe coverage. Positive
  means the conformal target was missed on test rollouts.
- **ROC-AUC / PR-AUC:** critic separability independent of threshold choice.
- **Constraint activation rates:** per-constraint counts/rates in
  `diagnostics.csv`, used to verify whether erosion is actually binding before
  choosing a label interpretation.
- **SLA hit rate:** fraction of deterministic shift-injection faults that
  produced a contract transition within the expected monitor detection budget.

Higher conformal coverage should usually lower the threshold, reduce false
negatives, and increase false positives/interventions. Erosion scale changes the
oracle boundary itself. If AUC/PR-AUC are weak, the coverage-vs-FPR curve is a
classifier-quality result, not a conformal-calibration result.

## Deterministic PPO Baseline Interface

`DeterministicPPOBaseline` is an inference-only adapter with the same shape a
trained PPO actor needs:

- `begin_episode(episode)`
- `act(obs, step, action_low, action_high)`

The default implementation is deterministic and intentionally lightweight so
the safety pipeline remains dependency-free. A trained PPO export can replace it
without changing rollout labeling, critic training, calibration, or plotting.

## Label Rule

The default intervention threshold is `0.05 rad/s`, about `1.6%` of the
one-sided CellArm action range (`pi rad/s`). This is a physical tolerance: an
L0-R modification below that magnitude is tracked diagnostically but is not
treated as a failure precursor by default.

This threshold is not tuned to hit a target positive rate. If the diagnostic
table shows a high positive rate under this rule, that is evidence about the
actor/boundary regime, not a reason to move the label threshold.

The default actor is the `boundary` actor. The `legacy_velocity` actor preserves
the old saturating action distribution so label fixes and actor-distribution
changes can be separated.

## Sim-To-Real Notes

The deployable stack is the combination, not the learned critic alone:

- **Minkowski erosion** creates a conservative boundary around physical limits
  under sensor uncertainty.
- **Sequential projection** enforces safe actions even when L1 is wrong.
- **Contract canonicalization** pins robot/cell assumptions to a reproducible
  artifact hash.
- **Shift injection** tests monitor behavior against deterministic, replayable
  faults before real deployment.
- **Conformal calibration** makes the L1 threshold explicit and auditable.

For real transfer work, Gate 2 telemetry should become the audit substrate:
rollout samples, labels, critic scores, threshold decisions, monitor events, and
state transitions should be emitted as deterministic NDJSON with a hash chain.

## Current Limitations

- The default actor is a deterministic PPO-style adapter, not a trained
  Stable-Baselines/PyTorch PPO checkpoint.
- Labels come from L0-R boundary evidence, not from long-horizon task failure.
- Thinned calibration reduces but does not eliminate temporal correlation.
  Episode-block calibration is available with `--calibration-unit episode_block`
  but needs more calibration episodes to avoid coarse thresholds.
- `metric_cell_valid=false` marks cells with fewer than the configured unsafe
  test floor (`30` by default). Reported coverage/FNR/PR-AUC columns are null
  for those cells even though raw diagnostic counts remain available.
- SLA hit rate uses expected detection budgets for the default probe types;
  exact monitor debounce/hysteresis windows remain covered by Gate 1 tests.
- The critic is linear/logistic by design. That is useful for auditable MVP
  plots, but not the final representation if boundary geometry becomes complex.
