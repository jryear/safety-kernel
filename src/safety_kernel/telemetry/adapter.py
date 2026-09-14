"""Step-info adapter seam for the frozen Gate 2 producer event order."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from safety_kernel.telemetry.bundle import RunBundle
from safety_kernel.telemetry.errors import TelemetryError
from safety_kernel.telemetry.schema import COMMAND_AUTHORITY, SCHEMA_VERSION


class TelemetryRun:
    """Own event ordering while runtime integration supplies post-step evidence only."""

    def __init__(self, bundle: RunBundle, *, total_episodes: int):
        self.bundle, self.total_episodes = bundle, total_episodes
        self._episode: _Episode | None = None
        self._completed_episodes = 0
        self.bundle.append({"event_type": "run_start", "timestamp_ns": 0, "manifest_hash": bundle.manifest_hash,
                            "contract_hash": bundle.contract_hash, "environment_hash": bundle.environment_hash,
                            "total_episodes": total_episodes, "schema_version": SCHEMA_VERSION,
                            "command_authority": COMMAND_AUTHORITY})

    def start_episode(self, episode_id: str, seed: int, *, planned_steps: int, timestamp_ns: int = 0) -> None:
        if self._episode is not None:
            raise TelemetryError("episode already active")
        self.bundle.append({"event_type": "episode_start", "timestamp_ns": timestamp_ns, "episode_id": episode_id,
                            "seed": seed, "contract_hash": self.bundle.contract_hash, "planned_steps": planned_steps})
        self._episode = _Episode(episode_id, seed, planned_steps)

    def record_step(self, info: Mapping[str, Any], *, qdot_proposed: list[float] | None = None,
                    timestamp_ns: int | None = None, shifts: list[Mapping[str, Any]] = (),
                    l0r_evidence: Mapping[str, Any] | None = None) -> None:
        """Emit all causal events for zero-based tick `info['step']`.

        If `env.step()` exposes `authority_info`, this seam copies its authority,
        monitor, reset, and RTA facts. Missing runtime integration stays explicit as
        UNAVAILABLE rather than inventing safety evidence.
        """
        episode = self._require_episode()
        step = info.get("step")
        if not isinstance(step, int) or step < 0:
            raise TelemetryError("step info must carry a zero-based non-negative integer")
        if step != episode.steps:
            raise TelemetryError(
                f"step evidence must be contiguous: expected {episode.steps}, got {step}"
            )
        # Decision timestamps are evidence from the simulator's pre-step clock;
        # a control-period-derived adapter timestamp would be an invented time
        # base and can drift from the plant/model record.
        timestamp = (
            info.get("simulated_pre_step_timestamp_ns")
            if timestamp_ns is None
            else timestamp_ns
        )
        if not isinstance(timestamp, int):
            raise TelemetryError("timestamp_ns must be a deterministic integer")
        intervention = _require_mapping(info.get("intervention"), "intervention")
        l0c = _mapping(info.get("l0c"))
        authority = _require_mapping(info.get("authority_info"), "authority_info")
        truth = _mapping(info.get("pre_step_truth", info.get("truth")))
        observed = _mapping(info.get("pre_step_observed", info.get("observed")))
        believed = _mapping(info.get("pre_step_believed", info.get("believed")))
        proposal = list(qdot_proposed if qdot_proposed is not None else authority.get("proposal", intervention.get("qdot_cmd", [])))
        l0c_command = list(authority.get("l0c_command", intervention.get("qdot_cmd", proposal)))
        l0r_command = list(authority.get("l0r_command", intervention.get("qdot_exec", l0c_command)))
        final_command = list(authority.get("final_command", l0r_command))
        state_pre = str(authority["state_pre"])
        state_post = str(authority["state_post"])
        authorized = authority.get("authorized")
        if not isinstance(authorized, bool):
            raise TelemetryError("authority_info.authorized must be boolean")
        reason = str(authority["reason"])
        reset_input = authority.get("reset_input", info.get("reset_input", None))
        raw_monitor_inputs = authority.get("monitor_inputs", info.get("monitor_inputs"))
        raw_monitor_results = authority.get("monitor_results", info.get("monitor_results"))
        if isinstance(raw_monitor_inputs, Mapping) and isinstance(raw_monitor_results, Mapping):
            monitor_inputs, monitor_results = dict(raw_monitor_inputs), dict(raw_monitor_results)
        elif authorized is False:
            unavailable = {"availability": "unavailable", "reason": "monitor evidence unavailable"}
            monitor_inputs, monitor_results = dict(unavailable), dict(unavailable)
        else:
            raise TelemetryError("monitor evidence is required for an authorized decision")

        for shift in shifts:
            self.bundle.append({"event_type": "shift_injected", "timestamp_ns": timestamp, "episode_id": episode.episode_id,
                                "step": step, "shift_type": shift["shift_type"], "shift_params": dict(shift.get("params", {})),
                                "affected_channels": list(shift.get("affected_channels", []))})
        self.bundle.append({"event_type": "decision_evidence", "timestamp_ns": timestamp, "episode_id": episode.episode_id,
                            "step": step, "proposal": proposal, "l0c_command": l0c_command, "l0r_command": l0r_command,
                            "final_command": final_command, "truth": dict(truth), "observed": dict(observed), "believed": dict(believed),
                            "monitor_inputs": dict(monitor_inputs), "monitor_results": dict(monitor_results),
                            "authority_state_pre": state_pre, "authority_state_post": state_post, "authorized": authorized,
                            "authority_reason": reason, "reset_input": reset_input, "rta_nearmiss": bool(intervention.get("rta_nearmiss", False))})
        if _changed(proposal, l0c_command) and l0c.get("scaling_factor", 1.0) < 1.0:
            self.bundle.append(_intervention(timestamp, episode.episode_id, step, "L0C", proposal, l0c_command, l0c, truth, observed, believed)); episode.interventions["L0C"] += 1
        if _changed(l0c_command, l0r_command) and intervention.get("constraints_active"):
            self.bundle.append(_intervention(timestamp, episode.episode_id, step, "L0R", l0c_command, l0r_command, intervention, truth, observed, believed)); episode.interventions["L0R"] += 1
        if not authorized and _changed(l0r_command, final_command):
            source = {"constraints_active": [reason], "rta_nearmiss": bool(intervention.get("rta_nearmiss", False)), "layer_info": {"authority_reason": reason}}
            self.bundle.append(_intervention(timestamp, episode.episode_id, step, "RTA", l0r_command, final_command, source, truth, observed, believed)); episode.interventions["RTA"] += 1
        for state_change in info.get("contract_events", []):
            if not isinstance(state_change, Mapping):
                raise TelemetryError("contract event must be a mapping")
            self.bundle.append({"event_type": "contract_state_change", "timestamp_ns": timestamp, "episode_id": episode.episode_id, "step": step,
                                "from_state": state_change["from_state"], "to_state": state_change["to_state"],
                                "trigger_monitor": state_change["trigger_monitor"], "trigger_severity": state_change["trigger_severity"], "monitors_status": dict(state_change["monitors_status"])})
            episode.transitions.append({"step": step, "from": state_change["from_state"], "to": state_change["to_state"], "trigger": state_change["trigger_monitor"]})
        margins, collisions = _mapping(intervention.get("margin_to_L0R")), intervention.get("collision_pairs", [])
        min_collision = min((item.get("distance") for item in collisions if isinstance(item, Mapping)), default=0.0)
        min_keepout = min(_mapping(truth.get("tcp_to_keepout_distances")).values(), default=0.0)
        min_margin = min(margins.values(), default=0.0)
        summary_state = state_post if state_post != "UNAVAILABLE" else "ACTIVE"
        self.bundle.append({"event_type": "kernel_step_summary", "timestamp_ns": timestamp, "episode_id": episode.episode_id, "step": step,
                            "contract_state": summary_state,
                            "l0r_intervened": bool(intervention.get("constraints_active")) and _changed(l0c_command, l0r_command),
                            "l0c_active": bool(l0c.get("l0c_active", False)),
                            "l0c_intervened": l0c.get("scaling_factor", 1.0) < 1.0, "l0c_scaling_factor": l0c.get("scaling_factor", 1.0),
                            "l0c_cap_evidence": _cap(l0c), "energy_ratio": intervention.get("energy_ratio", 0.0), "min_margin_to_L0R": min_margin,
                            "rta_nearmiss": bool(intervention.get("rta_nearmiss", False)), "min_collision_distance": min_collision,
                            "min_keepout_distance": min_keepout,
                            "l0r_evidence": dict(l0r_evidence if l0r_evidence is not None else info["l0r_evidence"])
                            if l0r_evidence is not None or isinstance(info.get("l0r_evidence"), Mapping)
                            else None})
        episode.steps += 1
        episode.min_margin, episode.max_energy, episode.final_state = min(episode.min_margin, float(min_margin)), max(episode.max_energy, float(intervention.get("energy_ratio", 0.0))), summary_state

    def end_episode(self, *, synthetic_revocation: bool = False, synthetic_revocation_params: Mapping[str, Any] | None = None, l1_arm: str = "none", full_head_staleness_episodes: int = 0, fn_count: int = 0, fp_count: int = 0) -> None:
        episode = self._require_episode()
        if episode.steps != episode.planned_steps:
            raise TelemetryError("episode completed with different number of steps than planned")
        self.bundle.append({"event_type": "episode_summary", "episode_id": episode.episode_id, "seed": episode.seed, "contract_hash": self.bundle.contract_hash,
                            "total_steps": episode.steps, "total_interventions": dict(episode.interventions), "state_transitions": episode.transitions,
                            "state_transition_count": len(episode.transitions),
                            "final_state": episode.final_state, "synthetic_revocation": synthetic_revocation,
                            "synthetic_revocation_params": dict(synthetic_revocation_params) if synthetic_revocation_params else None, "l1_arm": l1_arm,
                            "full_head_staleness_episodes": full_head_staleness_episodes, "min_margin_to_L0R": 0.0 if not episode.steps else episode.min_margin,
                            "max_energy_ratio": episode.max_energy, "fn_count": fn_count, "fp_count": fp_count, "episode_crc32": self.bundle.episode_crc32(episode.episode_id)})
        self._episode, self._completed_episodes = None, self._completed_episodes + 1

    def complete(self, *, timestamp_ns: int) -> dict[str, Any]:
        if self._episode is not None:
            raise TelemetryError("cannot complete with active episode")
        if self._completed_episodes != self.total_episodes:
            raise TelemetryError(
                f"run completed {self._completed_episodes} episodes; expected {self.total_episodes}"
            )
        self.bundle.append({"event_type": "run_end", "timestamp_ns": timestamp_ns, "status": "completed", "command_authority": COMMAND_AUTHORITY,
                            "actual_event_count": self.bundle.event_count + 1, "actual_episode_count": self._completed_episodes, "last_event_hash": self.bundle.previous_hash})
        self.bundle.seal_producer()
        return {"bundle": str(self.bundle.path), "event_count": self.bundle.event_count, "episode_count": self._completed_episodes}

    def _require_episode(self) -> "_Episode":
        if self._episode is None:
            raise TelemetryError("no active episode")
        return self._episode


class _Episode:
    def __init__(self, episode_id: str, seed: int, planned_steps: int):
        self.episode_id, self.seed, self.planned_steps, self.steps = episode_id, seed, planned_steps, 0
        self.interventions: Counter[str] = Counter({"L0R": 0, "L0C": 0, "L1": 0, "RTA": 0})
        self.transitions: list[dict[str, Any]] = []
        self.final_state, self.min_margin, self.max_energy = "ACTIVE", float("inf"), 0.0


def _intervention(timestamp: int, episode_id: str, step: int, layer: str, before: list[Any], after: list[Any], source: Mapping[str, Any], truth: Mapping[str, Any], observed: Mapping[str, Any], believed: Mapping[str, Any]) -> dict[str, Any]:
    return {"event_type": "safety_intervention", "timestamp_ns": timestamp, "episode_id": episode_id, "step": step, "layer": layer,
            "qdot_cmd": before, "qdot_exec": after, "constraints_active": list(source.get("constraints_active", [])),
            "margin_to_L0R": dict(source.get("margin_to_L0R", {})) if layer == "L0R" else {}, "energy_ratio": source.get("energy_ratio", 0.0) if layer == "L0R" else 0.0,
            "energy_scaling_factor": source.get("energy_scaling_factor", 1.0) if layer == "L0R" else 1.0,
            "collision_pairs": list(source.get("collision_pairs", [])) if layer == "L0R" else [], "rta_nearmiss": bool(source.get("rta_nearmiss", False)),
            "layer_info": dict(source.get("layer_info", {})) if layer == "RTA" else {"tcp_speed_before": source.get("tcp_speed_before", 0.0), "tcp_speed_after": source.get("tcp_speed_after", 0.0), "scaling_factor": source.get("scaling_factor", 1.0)} if layer == "L0C" else {},
            "truth": dict(truth), "observed": dict(observed), "believed": dict(believed)}


def _cap(l0c: Mapping[str, Any]) -> dict[str, Any] | None:
    required = {"tcp_speed_after", "max_tcp_speed_mps", "tcp_jacobian"}
    return {key: l0c[key] for key in required} if required <= set(l0c) else None


def _changed(before: list[Any], after: list[Any]) -> bool:
    return before != after


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TelemetryError(f"{label} evidence is required")
    return value
