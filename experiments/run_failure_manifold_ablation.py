#!/usr/bin/env python3
"""Run the L1 failure-manifold MVP ablation.

Outputs:
  - metrics.csv
  - summary.json
  - plots/*.svg

Example:
  PYTHONPATH=src python experiments/run_failure_manifold_ablation.py \
    --quick --out experiments/results/quick
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety_kernel.l1.ablations import (
    AblationConfig,
    coerce_float_tuple,
    run_erosion_conformal_ablation_with_diagnostics,
)


DEFAULT_CONTRACT = ROOT / "contracts" / "cell_arm_track_a.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "failure_manifold")
    parser.add_argument("--erosion-scales", nargs="+", default=["0.75", "1.0", "1.25", "1.5"])
    parser.add_argument("--coverages", nargs="+", default=["0.8", "0.9", "0.95"])
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--calibration-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--intervention-tolerance", type=float, default=0.05)
    parser.add_argument("--boundary-margin", type=float, default=0.0)
    parser.add_argument("--calibration-unit", choices=["sample", "thinned", "episode_block"], default="thinned")
    parser.add_argument("--calibration-stride", type=int, default=10)
    parser.add_argument("--min-unsafe-floor", type=int, default=30)
    parser.add_argument("--policy-kind", choices=["boundary", "legacy_velocity"], default="boundary")
    parser.add_argument("--compare-calibration-units", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    parser.add_argument("--no-shifts", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.erosion_scales = ["0.75", "1.0"]
        args.coverages = ["0.8", "0.95"]
        args.train_episodes = 3
        args.calibration_episodes = 2
        args.test_episodes = 2
        args.max_steps = 120

    config = AblationConfig(
        erosion_scales=coerce_float_tuple(args.erosion_scales),
        coverages=coerce_float_tuple(args.coverages),
        train_episodes=args.train_episodes,
        calibration_episodes=args.calibration_episodes,
        test_episodes=args.test_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        contract_path=None if args.no_contract else DEFAULT_CONTRACT,
        use_shift_schedule=not args.no_shifts,
        intervention_tolerance=args.intervention_tolerance,
        boundary_margin=args.boundary_margin,
        calibration_unit=args.calibration_unit,
        calibration_stride=args.calibration_stride,
        min_unsafe_floor=args.min_unsafe_floor,
        policy_kind=args.policy_kind,
    )

    result = run_erosion_conformal_ablation_with_diagnostics(config)
    rows = result.metrics_rows
    args.out.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out / "plots"
    plots_dir.mkdir(exist_ok=True)

    csv_path = args.out / "metrics.csv"
    _write_csv(csv_path, rows)
    diagnostics_path = args.out / "diagnostics.csv"
    _write_csv(diagnostics_path, result.diagnostic_rows)
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": _config_json(config),
                "diagnostic_rows": result.diagnostic_rows,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if args.compare_calibration_units:
        comparison_rows = []
        for unit in ["sample", "thinned", "episode_block"]:
            compare_config = AblationConfig(
                erosion_scales=config.erosion_scales,
                coverages=config.coverages,
                train_episodes=config.train_episodes,
                calibration_episodes=config.calibration_episodes,
                test_episodes=config.test_episodes,
                max_steps=config.max_steps,
                seed=config.seed,
                contract_path=config.contract_path,
                use_shift_schedule=config.use_shift_schedule,
                intervention_tolerance=config.intervention_tolerance,
                boundary_margin=config.boundary_margin,
                calibration_unit=unit,
                calibration_stride=config.calibration_stride,
                min_unsafe_floor=config.min_unsafe_floor,
                policy_kind=config.policy_kind,
            )
            comparison_rows.extend(
                run_erosion_conformal_ablation_with_diagnostics(compare_config).metrics_rows
            )
        comparison_path = args.out / "calibration_unit_comparison.csv"
        _write_csv(comparison_path, comparison_rows)

    for metric in [
        "false_positive_rate",
        "false_negative_rate",
        "reported_false_negative_rate",
        "intervention_rate",
        "empirical_unsafe_coverage",
        "reported_empirical_unsafe_coverage",
        "coverage_gap",
        "reported_coverage_gap",
        "roc_auc",
        "pr_auc",
        "reported_pr_auc",
        "sla_hit_rate",
    ]:
        svg = _line_plot_svg(
            rows,
            x_key="erosion_scale",
            y_key=metric,
            series_key="coverage",
            title=metric.replace("_", " ").title(),
        )
        (plots_dir / f"{metric}.svg").write_text(svg, encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {diagnostics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote plots to {plots_dir}")
    if args.compare_calibration_units:
        print(f"wrote {comparison_path}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _config_json(config: AblationConfig) -> dict:
    return {
        "erosion_scales": list(config.erosion_scales),
        "coverages": list(config.coverages),
        "train_episodes": config.train_episodes,
        "calibration_episodes": config.calibration_episodes,
        "test_episodes": config.test_episodes,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "contract_path": str(config.contract_path) if config.contract_path else None,
        "use_shift_schedule": config.use_shift_schedule,
        "intervention_tolerance": config.intervention_tolerance,
        "boundary_margin": config.boundary_margin,
        "calibration_unit": config.calibration_unit,
        "calibration_stride": config.calibration_stride,
        "min_unsafe_floor": config.min_unsafe_floor,
        "policy_kind": config.policy_kind,
    }


def _line_plot_svg(
    rows: list[dict],
    *,
    x_key: str,
    y_key: str,
    series_key: str,
    title: str,
) -> str:
    width = 760
    height = 460
    left = 74
    right = 26
    top = 54
    bottom = 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({float(r[x_key]) for r in rows})
    ys = [float(r[y_key]) for r in rows]
    ys = [y for y in ys if y == y]
    plot_rows = [r for r in rows if float(r[y_key]) == float(r[y_key])]
    if not plot_rows:
        return _empty_plot_svg(width, height, title, y_key)
    y_min = min(0.0, min(ys) if ys else 0.0)
    y_max = max(1.0, max(ys) if ys else 1.0)
    if y_key in {
        "false_positive_rate",
        "false_negative_rate",
        "reported_false_negative_rate",
        "intervention_rate",
        "empirical_unsafe_coverage",
        "reported_empirical_unsafe_coverage",
        "roc_auc",
        "pr_auc",
        "reported_pr_auc",
        "sla_hit_rate",
    }:
        y_max = 1.0
        y_min = 0.0
    if y_key in {"coverage_gap", "reported_coverage_gap"}:
        y_min = min(-0.25, min(ys) if ys else -0.25)
        y_max = max(0.25, max(ys) if ys else 0.25)
    series = sorted({float(r[series_key]) for r in rows})
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    def px(x: float) -> float:
        if len(xs) == 1:
            return left + plot_w / 2
        return left + (x - min(xs)) / (max(xs) - min(xs)) * plot_w

    def py(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="20" font-weight="700">{_esc(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    if y_key == "coverage_gap":
        ticks = [y_min, 0.0, y_max]
    for tick in ticks:
        y = py(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for x in xs:
        xp = px(x)
        parts.append(f'<line x1="{xp:.2f}" y1="{top + plot_h}" x2="{xp:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>')
        parts.append(f'<text x="{xp:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x:g}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">Erosion scale</text>')
    parts.append(f'<text x="16" y="{top + plot_h / 2}" transform="rotate(-90 16 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{_esc(y_key)}</text>')

    for idx, value in enumerate(series):
        color = palette[idx % len(palette)]
        points = []
        for row in sorted((r for r in plot_rows if float(r[series_key]) == value), key=lambda r: float(r[x_key])):
            points.append((px(float(row[x_key])), py(float(row[y_key]))))
        if points:
            point_s = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{point_s}"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            lx = left + plot_w - 150
            ly = top + 22 + idx * 22
            parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
            parts.append(f'<text x="{lx + 30}" y="{ly + 4}" font-family="Arial" font-size="12">coverage {value:g}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _empty_plot_svg(width: int, height: int, title: str, y_key: str) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="60" y="40" font-family="Arial" font-size="20" font-weight="700">{_esc(title)}</text>',
        f'<text x="60" y="80" font-family="Arial" font-size="13">No statistically valid cells for {_esc(y_key)}.</text>',
        "</svg>",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
