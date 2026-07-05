"""
Standalone sweep for reroute rework assumption.

This script varies the fraction of work/material that must be re-done
after a quality failure reroute and measures completion outcomes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation import SimulationRun
from simulation_secondary import SecondarySimulationRun


# =============================
# Editable Configuration
# =============================

# Output
OUTPUT_DIR = Path("output_rework_sweep")

# Sweep settings (fraction of already-performed work/material that must be re-done)
# 0.20 means 20% redo, 1.00 means 100% redo.
REWORK_FRACTIONS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# Core simulation settings
RUNS = 20
SEED = 42
NUM_SHOPS = 100
NUM_DAYS = 365
JOBS_PER_EVENT = 1

# Generation settings
JOB_GENERATION_MODE = "daily"  # start_only | daily | probabilistic | custom_days
JOB_GENERATION_PROBABILITY = 0.10
JOB_GENERATION_DAYS = [0]

# Quality / delay / planner settings
NUM_QUALITY_CHECKS = 3
MAX_DELAY_FACTOR = 10.0
FAILURE_PENALTY_RATE = 0.20
BACKUP_SHOP_DEPTH = 3
ALLOCATION_PLANNER_MODE = "fast"  # fast | thorough
ENFORCE_FIRST_LAYER_TIMELINE_FILTER = False

# Optional capacity overrides (set both to None to use shop defaults)
CAPACITY_UTILIZATION_MEAN = None
CAPACITY_UTILIZATION_STD = None

# Optional overrides (set to None to use defaults)
SHOP_TYPE_PARAMS_OVERRIDE = None
JOB_CONFIG_OVERRIDE = None


@dataclass
class SweepResult:
    rework_fraction: float
    total_jobs: int
    completed_jobs: int
    avg_completed_jobs_per_run: float
    avg_cost_per_completed_job: float
    completion_rate: float
    quality_success_rate: float
    timeline_success_rate: float
    avg_completion_day: float
    avg_days_late_completed: float


@dataclass
class ModeMetrics:
    mode: str
    total_jobs: int
    completed_jobs: int
    avg_completed_jobs_per_run: float
    avg_cost_per_completed_job: float
    completion_rate: float
    quality_success_rate: float
    timeline_success_rate: float
    avg_completion_day: float
    avg_days_late_completed: float


class ReworkSweepSimulationRun(SimulationRun):
    """Primary simulation variant with configurable reroute rework fraction."""

    def __init__(self, rework_fraction: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rework_fraction = float(np.clip(rework_fraction, 0.0, 1.0))

    def _handle_quality_failure(self, comp, shop):
        """On quality failure, optionally reroute with configurable work/material redo."""
        comp.quality_failure_count += 1
        elapsed = comp.deadline_days - comp.days_remaining

        if elapsed <= comp.max_delay and self._can_reroute_now(comp, shop):
            # Redo a configurable fraction of already-performed manhours.
            retain_fraction = 1.0 - self.rework_fraction
            comp.manhours_done *= retain_fraction

            comp.quality_checks_done = 0
            comp.compute_quality_check_thresholds(
                min(
                    comp.max_workers * comp.max_daily_manhours_per_worker,
                    comp.max_workers * shop.worker_capacity,
                ),
                self.num_quality_checks,
            )
            return True

        self._finalize_quality_failure(comp)
        return False


def _run_for_rework_fraction(rework_fraction: float) -> SweepResult:
    jobs_all = []
    completed_per_run = []

    for r in range(RUNS):
        sim = ReworkSweepSimulationRun(
            rework_fraction=rework_fraction,
            num_shops=NUM_SHOPS,
            num_days=NUM_DAYS,
            jobs_per_day=JOBS_PER_EVENT,
            rng_seed=SEED + r,
            num_quality_checks=NUM_QUALITY_CHECKS,
            failure_penalty_rate=FAILURE_PENALTY_RATE,
            max_delay_factor=MAX_DELAY_FACTOR,
            capacity_utilization_mean=CAPACITY_UTILIZATION_MEAN,
            capacity_utilization_std=CAPACITY_UTILIZATION_STD,
            job_generation_mode=JOB_GENERATION_MODE,
            job_generation_probability=JOB_GENERATION_PROBABILITY,
            job_generation_days=JOB_GENERATION_DAYS,
            shop_type_params_override=SHOP_TYPE_PARAMS_OVERRIDE,
            job_config=JOB_CONFIG_OVERRIDE,
            backup_shop_depth=BACKUP_SHOP_DEPTH,
            allocation_planner_mode=ALLOCATION_PLANNER_MODE,
            enforce_first_layer_timeline_filter=ENFORCE_FIRST_LAYER_TIMELINE_FILTER,
        )
        sim.run()
        jobs = list(sim.jobs.values())
        jobs_all.extend(jobs)
        completed_per_run.append(sum(1 for j in jobs if j.completed))

    total_jobs = len(jobs_all)
    completed_jobs = sum(1 for j in jobs_all if j.completed)
    quality_successes = sum(1 for j in jobs_all if j.completed and j.quality_success)
    timeline_successes = sum(1 for j in jobs_all if j.timeline_success)
    completion_days = [j.day_completed for j in jobs_all if j.completed and j.day_completed is not None]
    days_late = [j.days_late for j in jobs_all if j.completed]
    completed_costs = [j.total_cost for j in jobs_all if j.completed]

    return SweepResult(
        rework_fraction=rework_fraction,
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        avg_completed_jobs_per_run=float(np.mean(completed_per_run)) if completed_per_run else 0.0,
        avg_cost_per_completed_job=float(np.mean(completed_costs)) if completed_costs else 0.0,
        completion_rate=(completed_jobs / total_jobs) if total_jobs else 0.0,
        quality_success_rate=(quality_successes / total_jobs) if total_jobs else 0.0,
        timeline_success_rate=(timeline_successes / total_jobs) if total_jobs else 0.0,
        avg_completion_day=float(np.mean(completion_days)) if completion_days else 0.0,
        avg_days_late_completed=float(np.mean(days_late)) if days_late else 0.0,
    )


def _run_secondary_mode(mode: str) -> ModeMetrics:
    jobs_all = []
    completed_per_run = []

    for r in range(RUNS):
        method_seed = SEED + 1000 + r + (500 if mode == "quality_top" else 0)
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
            shop_type_params_override=SHOP_TYPE_PARAMS_OVERRIDE,
            job_config=JOB_CONFIG_OVERRIDE,
        )
        sim.run()
        jobs = list(sim.jobs.values())
        jobs_all.extend(jobs)
        completed_per_run.append(sum(1 for j in jobs if j.completed))

    total_jobs = len(jobs_all)
    completed_jobs = sum(1 for j in jobs_all if j.completed)
    quality_successes = sum(1 for j in jobs_all if j.completed and j.quality_success)
    timeline_successes = sum(1 for j in jobs_all if j.timeline_success)
    completion_days = [j.day_completed for j in jobs_all if j.completed and j.day_completed is not None]
    days_late = [j.days_late for j in jobs_all if j.completed]
    completed_costs = [j.total_cost for j in jobs_all if j.completed]

    return ModeMetrics(
        mode=("Base Case" if mode == "random_cheapest" else "Quality Only"),
        total_jobs=total_jobs,
        completed_jobs=completed_jobs,
        avg_completed_jobs_per_run=float(np.mean(completed_per_run)) if completed_per_run else 0.0,
        avg_cost_per_completed_job=float(np.mean(completed_costs)) if completed_costs else 0.0,
        completion_rate=(completed_jobs / total_jobs) if total_jobs else 0.0,
        quality_success_rate=(quality_successes / total_jobs) if total_jobs else 0.0,
        timeline_success_rate=(timeline_successes / total_jobs) if total_jobs else 0.0,
        avg_completion_day=float(np.mean(completion_days)) if completion_days else 0.0,
        avg_days_late_completed=float(np.mean(days_late)) if days_late else 0.0,
    )


def _save_results(results: list[SweepResult], out_dir: Path, base_case: ModeMetrics, quality_only: ModeMetrics):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        row = asdict(r)
        row["base_case_completion_rate"] = base_case.completion_rate
        row["base_case_quality_success_rate"] = base_case.quality_success_rate
        row["base_case_timeline_success_rate"] = base_case.timeline_success_rate
        row["quality_only_completion_rate"] = quality_only.completion_rate
        row["quality_only_quality_success_rate"] = quality_only.quality_success_rate
        row["quality_only_timeline_success_rate"] = quality_only.timeline_success_rate
        rows.append(row)

    json_path = out_dir / "rework_sweep_results.json"
    json_path.write_text(json.dumps(rows, indent=2))

    csv_path = out_dir / "rework_sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comparison = [asdict(base_case), asdict(quality_only)]
    (out_dir / "rework_sweep_mode_comparison.json").write_text(json.dumps(comparison, indent=2))
    with (out_dir / "rework_sweep_mode_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)


def _plot_results(results: list[SweepResult], out_dir: Path, base_case: ModeMetrics, quality_only: ModeMetrics):
    x = np.array([r.rework_fraction for r in results])
    quality_rate = np.array([r.quality_success_rate for r in results])
    avg_days_late = np.array([r.avg_days_late_completed for r in results])
    avg_cost = np.array([r.avg_cost_per_completed_job for r in results])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, quality_rate, marker="o", linewidth=2.0, label="Primary quality success rate")
    ax.hlines(base_case.quality_success_rate, x.min(), x.max(), colors="#888888", linestyles=":", linewidth=1.8, label="Base quality success rate")
    ax.hlines(quality_only.quality_success_rate, x.min(), x.max(), colors="#4FA3E3", linestyles=":", linewidth=1.8, label="Quality-only quality success rate")
    ax.set_title("Quality Success Rate vs Rework Fraction")
    ax.set_xlabel("Rework Fraction on Quality Reroute")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "rework_sweep_rates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, avg_days_late, marker="o", linewidth=2.0, label="Primary avg days late")
    ax.hlines(base_case.avg_days_late_completed, x.min(), x.max(), colors="#888888", linestyles="--", linewidth=1.8, label="Base avg days late")
    ax.hlines(quality_only.avg_days_late_completed, x.min(), x.max(), colors="#4FA3E3", linestyles="--", linewidth=1.8, label="Quality-only avg days late")
    ax.set_title("Average Days Late vs Rework Fraction")
    ax.set_xlabel("Rework Fraction on Quality Reroute")
    ax.set_ylabel("Average days late")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "rework_sweep_avg_days_late.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, avg_cost, marker="o", linewidth=2.0, color="#0B4F8C", label="Primary avg cost/completed job")
    ax.hlines(base_case.avg_cost_per_completed_job, x.min(), x.max(), colors="#888888", linestyles="--", linewidth=1.8, label="Base avg cost/completed job")
    ax.hlines(quality_only.avg_cost_per_completed_job, x.min(), x.max(), colors="#4FA3E3", linestyles="--", linewidth=1.8, label="Quality-only avg cost/completed job")
    ax.set_title("Average Cost per Completed Job vs Rework Fraction")
    ax.set_xlabel("Rework Fraction on Quality Reroute")
    ax.set_ylabel("Average cost per completed job")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "rework_sweep_avg_cost_per_job.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for frac in REWORK_FRACTIONS:
        print(f"Running rework fraction={frac:.2f} ...")
        results.append(_run_for_rework_fraction(frac))

    print("Running Base Case comparison ...")
    base_case = _run_secondary_mode("random_cheapest")
    print("Running Quality-Only comparison ...")
    quality_only = _run_secondary_mode("quality_top")

    _save_results(results, OUTPUT_DIR, base_case, quality_only)
    _plot_results(results, OUTPUT_DIR, base_case, quality_only)

    print("\nDone. Outputs:")
    print(f"- {OUTPUT_DIR / 'rework_sweep_results.json'}")
    print(f"- {OUTPUT_DIR / 'rework_sweep_results.csv'}")
    print(f"- {OUTPUT_DIR / 'rework_sweep_rates.png'}")
    print(f"- {OUTPUT_DIR / 'rework_sweep_avg_days_late.png'}")
    print(f"- {OUTPUT_DIR / 'rework_sweep_avg_cost_per_job.png'}")
    print(f"- {OUTPUT_DIR / 'rework_sweep_mode_comparison.json'}")
    print(f"- {OUTPUT_DIR / 'rework_sweep_mode_comparison.csv'}")


if __name__ == "__main__":
    main()
