"""
Standalone sweep for network shop quality assumptions.

This script varies the average quality of all shop types across the network
and measures how Primary, Base Case, and Quality Only modes respond in:
  - average cost per completed job
  - average days late for completed jobs
  - overall quality success rate

It can also couple network quality improvements to faster execution by
shifting shop work efficiency and/or worker capacity along with quality.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import SHOP_TYPE_PARAMS
from simulation import SimulationRun
from simulation_secondary import SecondarySimulationRun


# =============================
# Editable Configuration
# =============================

OUTPUT_DIR = Path("output_network_quality_sweep")

# Additive shift applied to each shop type's quality_rate_mean.
# Example: -0.05 reduces every shop type mean quality by 5 percentage points.
QUALITY_MEAN_SHIFTS = [-0.20, -0.16, -0.12, -0.08, -0.04, 0.00, 0.04, 0.08]

# Coupling assumptions: when network quality improves or degrades, timing-related
# shop characteristics can move too. Set these to 0.0 to disable coupling.
WORK_EFFICIENCY_SHIFT_PER_QUALITY_SHIFT = 1.0
WORKER_CAPACITY_SHIFT_PER_QUALITY_SHIFT = 0.0

RUNS = 12
SEED = 42
NUM_SHOPS = 100
NUM_DAYS = 365
JOBS_PER_EVENT = 1

JOB_GENERATION_MODE = "daily"  # start_only | daily | probabilistic | custom_days
JOB_GENERATION_PROBABILITY = 0.10
JOB_GENERATION_DAYS = [0]

NUM_QUALITY_CHECKS = 3
MAX_DELAY_FACTOR = 10.0
FAILURE_PENALTY_RATE = 0.20
BACKUP_SHOP_DEPTH = 3
ALLOCATION_PLANNER_MODE = "fast"  # fast | thorough
ENFORCE_FIRST_LAYER_TIMELINE_FILTER = False

CAPACITY_UTILIZATION_MEAN = None
CAPACITY_UTILIZATION_STD = None

JOB_CONFIG_OVERRIDE = None


@dataclass
class ModeSweepResult:
    mode: str
    quality_mean_shift: float
    network_quality_mean: float
    total_jobs: int
    completed_jobs: int
    avg_completed_jobs_per_run: float
    avg_cost_per_completed_job: float
    avg_days_late_completed: float
    quality_success_rate: float
    timeline_success_rate: float


def _shop_quality_override(quality_mean_shift: float) -> tuple[dict[str, dict], float]:
    override: dict[str, dict] = {}
    weighted_mean = 0.0

    for shop_type, params in SHOP_TYPE_PARAMS.items():
        shifted_mean = float(np.clip(params["quality_rate_mean"] + quality_mean_shift, 0.0, 1.0))
        shifted_efficiency = float(
            np.clip(
                params["work_efficiency_mean"] + quality_mean_shift * WORK_EFFICIENCY_SHIFT_PER_QUALITY_SHIFT,
                0.0,
                1.0,
            )
        )
        shifted_capacity = float(
            max(
                0.1,
                params["worker_capacity_mean"] + quality_mean_shift * WORKER_CAPACITY_SHIFT_PER_QUALITY_SHIFT,
            )
        )
        override[shop_type] = {
            "quality_rate_mean": shifted_mean,
            "work_efficiency_mean": shifted_efficiency,
            "worker_capacity_mean": shifted_capacity,
        }
        weighted_mean += shifted_mean * float(params.get("fraction", 0.0))

    return override, weighted_mean


def _run_primary(quality_mean_shift: float) -> ModeSweepResult:
    jobs_all = []
    completed_per_run = []
    shop_override, network_quality_mean = _shop_quality_override(quality_mean_shift)

    for run_idx in range(RUNS):
        sim = SimulationRun(
            num_shops=NUM_SHOPS,
            num_days=NUM_DAYS,
            jobs_per_day=JOBS_PER_EVENT,
            rng_seed=SEED + run_idx,
            num_quality_checks=NUM_QUALITY_CHECKS,
            failure_penalty_rate=FAILURE_PENALTY_RATE,
            max_delay_factor=MAX_DELAY_FACTOR,
            capacity_utilization_mean=CAPACITY_UTILIZATION_MEAN,
            capacity_utilization_std=CAPACITY_UTILIZATION_STD,
            job_generation_mode=JOB_GENERATION_MODE,
            job_generation_probability=JOB_GENERATION_PROBABILITY,
            job_generation_days=JOB_GENERATION_DAYS,
            shop_type_params_override=shop_override,
            job_config=JOB_CONFIG_OVERRIDE,
            backup_shop_depth=BACKUP_SHOP_DEPTH,
            allocation_planner_mode=ALLOCATION_PLANNER_MODE,
            enforce_first_layer_timeline_filter=ENFORCE_FIRST_LAYER_TIMELINE_FILTER,
        )
        sim.run()
        jobs = list(sim.jobs.values())
        jobs_all.extend(jobs)
        completed_per_run.append(sum(1 for job in jobs if job.completed))

    total_jobs = len(jobs_all)
    completed_jobs = sum(1 for job in jobs_all if job.completed)
    completed_costs = [job.total_cost for job in jobs_all if job.completed]
    days_late = [job.days_late for job in jobs_all if job.completed]
    quality_successes = sum(1 for job in jobs_all if job.completed and job.quality_success)
    timeline_successes = sum(1 for job in jobs_all if job.timeline_success)

    return ModeSweepResult(
        mode="Primary",
        quality_mean_shift=quality_mean_shift,
        network_quality_mean=network_quality_mean,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        avg_completed_jobs_per_run=float(np.mean(completed_per_run)) if completed_per_run else 0.0,
        avg_cost_per_completed_job=float(np.mean(completed_costs)) if completed_costs else 0.0,
        avg_days_late_completed=float(np.mean(days_late)) if days_late else 0.0,
        quality_success_rate=(quality_successes / total_jobs) if total_jobs else 0.0,
        timeline_success_rate=(timeline_successes / total_jobs) if total_jobs else 0.0,
    )


def _run_secondary(mode: str, quality_mean_shift: float) -> ModeSweepResult:
    jobs_all = []
    completed_per_run = []
    shop_override, network_quality_mean = _shop_quality_override(quality_mean_shift)

    for run_idx in range(RUNS):
        method_seed = SEED + 1000 + run_idx + (500 if mode == "quality_top" else 0)
        sim = SecondarySimulationRun(
            num_shops=NUM_SHOPS,
            num_days=NUM_DAYS,
            jobs_per_day=JOBS_PER_EVENT,
            allocation_method=mode,
            rng_seed=method_seed,
            failure_penalty_rate=FAILURE_PENALTY_RATE,
            max_delay_factor=MAX_DELAY_FACTOR,
            capacity_utilization_mean=CAPACITY_UTILIZATION_MEAN,
            capacity_utilization_std=CAPACITY_UTILIZATION_STD,
            job_generation_mode=JOB_GENERATION_MODE,
            job_generation_probability=JOB_GENERATION_PROBABILITY,
            job_generation_days=JOB_GENERATION_DAYS,
            shop_type_params_override=shop_override,
            job_config=JOB_CONFIG_OVERRIDE,
        )
        sim.run()
        jobs = list(sim.jobs.values())
        jobs_all.extend(jobs)
        completed_per_run.append(sum(1 for job in jobs if job.completed))

    total_jobs = len(jobs_all)
    completed_jobs = sum(1 for job in jobs_all if job.completed)
    completed_costs = [job.total_cost for job in jobs_all if job.completed]
    days_late = [job.days_late for job in jobs_all if job.completed]
    quality_successes = sum(1 for job in jobs_all if job.completed and job.quality_success)
    timeline_successes = sum(1 for job in jobs_all if job.timeline_success)

    return ModeSweepResult(
        mode=("Base Case" if mode == "random_cheapest" else "Quality Only"),
        quality_mean_shift=quality_mean_shift,
        network_quality_mean=network_quality_mean,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        avg_completed_jobs_per_run=float(np.mean(completed_per_run)) if completed_per_run else 0.0,
        avg_cost_per_completed_job=float(np.mean(completed_costs)) if completed_costs else 0.0,
        avg_days_late_completed=float(np.mean(days_late)) if days_late else 0.0,
        quality_success_rate=(quality_successes / total_jobs) if total_jobs else 0.0,
        timeline_success_rate=(timeline_successes / total_jobs) if total_jobs else 0.0,
    )


def _save_results(results: list[ModeSweepResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]

    (out_dir / "network_quality_sweep_results.json").write_text(json.dumps(rows, indent=2))

    with (out_dir / "network_quality_sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(results: list[ModeSweepResult], out_dir: Path, metric_key: str, title: str, ylabel: str, filename: str, y_limits: tuple[float, float] | None = None):
    fig, ax = plt.subplots(figsize=(11, 6))
    modes = ["Primary", "Base Case", "Quality Only"]
    colors = {
        "Primary": "#0B4F8C",
        "Base Case": "#888888",
        "Quality Only": "#4FA3E3",
    }

    for mode in modes:
        mode_rows = sorted((row for row in results if row.mode == mode), key=lambda row: row.network_quality_mean)
        x = np.array([row.network_quality_mean for row in mode_rows])
        y = np.array([getattr(row, metric_key) for row in mode_rows])
        ax.plot(x, y, marker="o", linewidth=2.0, color=colors[mode], label=mode)

    ax.set_title(title)
    ax.set_xlabel("Average Network Shop Quality Mean")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def _plot_results(results: list[ModeSweepResult], out_dir: Path):
    _plot_metric(
        results=results,
        out_dir=out_dir,
        metric_key="avg_cost_per_completed_job",
        title="Average Cost per Completed Job vs Network Quality",
        ylabel="Average cost per completed job",
        filename="network_quality_sweep_avg_cost_per_job.png",
    )
    _plot_metric(
        results=results,
        out_dir=out_dir,
        metric_key="avg_days_late_completed",
        title="Average Days Late vs Network Quality",
        ylabel="Average days late",
        filename="network_quality_sweep_avg_days_late.png",
    )
    _plot_metric(
        results=results,
        out_dir=out_dir,
        metric_key="quality_success_rate",
        title="Quality Success Rate vs Network Quality",
        ylabel="Quality success rate",
        filename="network_quality_sweep_quality_success_rate.png",
        y_limits=(0.0, 1.0),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ModeSweepResult] = []
    for quality_shift in QUALITY_MEAN_SHIFTS:
        print(f"Running network quality mean shift={quality_shift:+.2f} ...")
        results.append(_run_primary(quality_shift))
        results.append(_run_secondary("random_cheapest", quality_shift))
        results.append(_run_secondary("quality_top", quality_shift))

    _save_results(results, OUTPUT_DIR)
    _plot_results(results, OUTPUT_DIR)

    print("\nDone. Outputs:")
    print(f"- {OUTPUT_DIR / 'network_quality_sweep_results.json'}")
    print(f"- {OUTPUT_DIR / 'network_quality_sweep_results.csv'}")
    print(f"- {OUTPUT_DIR / 'network_quality_sweep_avg_cost_per_job.png'}")
    print(f"- {OUTPUT_DIR / 'network_quality_sweep_avg_days_late.png'}")
    print(f"- {OUTPUT_DIR / 'network_quality_sweep_quality_success_rate.png'}")


if __name__ == "__main__":
    main()
