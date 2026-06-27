"""
main.py — Entry point for the shop simulation.

Usage examples
--------------
# Default: 3 runs, primary mode only
python main.py

# Custom: 5 runs, 100 shops, 8 jobs/day, all modes
python main.py --runs 5 --shops 100 --jobs-per-day 8 --mode all --output results/

# Primary mode only, quiet
python main.py --runs 2 --mode primary --output results/primary/

# Secondary mode only (both allocation strategies)
python main.py --runs 2 --mode secondary --output results/secondary/
"""

import argparse
import time
import numpy as np
from typing import Optional

from simulation import SimulationRun
from simulation_secondary import SecondarySimulationRun
from stats import aggregate_runs
from plotting import (
    plot_shop_statistics,
    plot_job_statistics,
    plot_success_rates_vs_targets,
    plot_comparison,
    plot_shop_comparison,
    plot_job_cost_comparison,
)


def run_primary(num_runs: int, num_shops: int, num_days: int, jobs_per_day: int,
                base_seed: int, output_dir: str, num_quality_checks: int = 3,
                failure_penalty_rate: float = 0.20,
                max_delay_factor: float = 1.0,
                capacity_utilization_mean: Optional[float] = None,
                capacity_utilization_std: Optional[float] = None,
                job_generation_mode: str = "start_only",
                job_generation_probability: float = 0.0,
                job_generation_days: Optional[list[int]] = None):
    print(f"\n{'='*60}")
    print(f"  PRIMARY SIMULATION  ({num_runs} runs × {num_days} days)")
    print(f"  Shops: {num_shops}  Jobs/day: {jobs_per_day}")
    print(f"{'='*60}")

    all_stats = []
    for r in range(num_runs):
        t0 = time.perf_counter()
        seed = base_seed + r
        print(f"\n  Run {r+1}/{num_runs}  (seed={seed})")
        sim = SimulationRun(
            num_shops=num_shops,
            num_days=num_days,
            jobs_per_day=jobs_per_day,
            rng_seed=seed,
            num_quality_checks=num_quality_checks,
            failure_penalty_rate=failure_penalty_rate,
            max_delay_factor=max_delay_factor,
            capacity_utilization_mean=capacity_utilization_mean,
            capacity_utilization_std=capacity_utilization_std,
            job_generation_mode=job_generation_mode,
            job_generation_probability=job_generation_probability,
            job_generation_days=job_generation_days,
        )
        sim.run()
        stats = sim.get_statistics()
        all_stats.append(stats)
        n_complete = len(sim.completed_jobs)
        n_total = len(sim.jobs)
        print(f"    Completed jobs: {n_complete}/{n_total}  ({100*n_complete/max(n_total,1):.1f}%)")
        print(f"    Elapsed: {time.perf_counter()-t0:.1f}s")

    print("\n  Aggregating statistics …")
    agg = aggregate_runs(all_stats)

    print("  Plotting …")
    plot_shop_statistics(agg, output_dir=output_dir, label="Primary")
    plot_job_statistics(agg, output_dir=output_dir, label="Primary")
    plot_success_rates_vs_targets(agg, output_dir=output_dir, label="Primary")

    return agg


def run_secondary(num_runs: int, num_shops: int, num_days: int, jobs_per_day: int,
                  base_seed: int, output_dir: str,
                  failure_penalty_rate: float = 0.20,
                  max_delay_factor: float = 1.0,
                  capacity_utilization_mean: Optional[float] = None,
                  capacity_utilization_std: Optional[float] = None,
                  job_generation_mode: str = "start_only",
                  job_generation_probability: float = 0.0,
                  job_generation_days: Optional[list[int]] = None):
    results = {}
    for method in ("random_cheapest", "quality_top"):
        label = "Sec-Random" if method == "random_cheapest" else "Sec-QualTop"
        print(f"\n{'='*60}")
        print(f"  SECONDARY SIMULATION [{method}]  ({num_runs} runs × {num_days} days)")
        print(f"{'='*60}")

        all_stats = []
        for r in range(num_runs):
            t0 = time.perf_counter()
            seed = base_seed + 1000 + r + (500 if method == "quality_top" else 0)
            print(f"\n  Run {r+1}/{num_runs}  (seed={seed})")
            sim = SecondarySimulationRun(
                num_shops=num_shops,
                num_days=num_days,
                jobs_per_day=jobs_per_day,
                allocation_method=method,
                rng_seed=seed,
                failure_penalty_rate=failure_penalty_rate,
                max_delay_factor=max_delay_factor,
                capacity_utilization_mean=capacity_utilization_mean,
                capacity_utilization_std=capacity_utilization_std,
                job_generation_mode=job_generation_mode,
                job_generation_probability=job_generation_probability,
                job_generation_days=job_generation_days,
            )
            sim.run()
            stats = sim.get_statistics()
            all_stats.append(stats)
            n_complete = len(sim.completed_jobs)
            n_total = len(sim.jobs)
            print(f"    Completed jobs: {n_complete}/{n_total}  ({100*n_complete/max(n_total,1):.1f}%)")
            print(f"    Elapsed: {time.perf_counter()-t0:.1f}s")

        print("\n  Aggregating statistics …")
        agg = aggregate_runs(all_stats)
        results[method] = agg

        print("  Plotting …")
        plot_shop_statistics(agg, output_dir=output_dir, label=label)
        plot_job_statistics(agg, output_dir=output_dir, label=label)
        plot_success_rates_vs_targets(agg, output_dir=output_dir, label=label)

    return results


def main():
    parser = argparse.ArgumentParser(description="Shop simulation runner")
    parser.add_argument("--runs",         type=int, default=3,     help="Number of simulation runs (default: 3)")
    parser.add_argument("--shops",        type=int, default=100,   help="Number of shops (default: 100)")
    parser.add_argument("--days",         type=int, default=365,   help="Days per run (default: 365)")
    parser.add_argument("--jobs-per-day", type=int, default=5,     help="Jobs generated per day (default: 5)")
    parser.add_argument("--mode",         type=str, default="all",
                        choices=["primary", "secondary", "all"],
                        help="Simulation mode (default: all)")
    parser.add_argument("--output",       type=str, default="output", help="Output directory for plots")
    parser.add_argument("--seed",         type=int, default=42,    help="Base random seed (default: 42)")
    parser.add_argument("--quality-checks", type=int, default=3,  help="Number of quality checks per component in primary mode (default: 3)")
    parser.add_argument("--failure-penalty-rate", type=float, default=0.20,
                        help="Penalty rate applied to component total cost on timeline/quality failure (default: 0.20)")
    parser.add_argument("--max-delay-factor", type=float, default=1.0,
                        help="Max delay as a multiple of component deadline (default: 1.0)")
    parser.add_argument("--capacity-util-mean", type=float, default=None,
                        help="Override mean external capacity utilization for all shops (default: per shop type)")
    parser.add_argument("--capacity-util-std", type=float, default=None,
                        help="Override std dev external capacity utilization for all shops (default: per shop type)")
    parser.add_argument("--job-generation-mode", type=str, default="start_only",
                        choices=["start_only", "daily", "probabilistic", "custom_days"],
                        help="When jobs are generated (default: start_only)")
    parser.add_argument("--job-generation-probability", type=float, default=0.0,
                        help="Daily probability of a generation event in probabilistic mode (default: 0.0)")
    parser.add_argument("--job-generation-days", type=str, default="0",
                        help="Comma-separated day indices for custom_days mode, e.g. 0,30,60")
    args = parser.parse_args()

    job_generation_days = [int(x.strip()) for x in args.job_generation_days.split(",") if x.strip()]

    agg_primary = None
    agg_secondary = None

    if args.mode in ("primary", "all"):
        agg_primary = run_primary(
            num_runs=args.runs,
            num_shops=args.shops,
            num_days=args.days,
            jobs_per_day=args.jobs_per_day,
            base_seed=args.seed,
            output_dir=args.output,
            num_quality_checks=args.quality_checks,
            failure_penalty_rate=args.failure_penalty_rate,
            max_delay_factor=args.max_delay_factor,
            capacity_utilization_mean=args.capacity_util_mean,
            capacity_utilization_std=args.capacity_util_std,
            job_generation_mode=args.job_generation_mode,
            job_generation_probability=args.job_generation_probability,
            job_generation_days=job_generation_days,
        )

    if args.mode in ("secondary", "all"):
        agg_secondary = run_secondary(
            num_runs=args.runs,
            num_shops=args.shops,
            num_days=args.days,
            jobs_per_day=args.jobs_per_day,
            base_seed=args.seed,
            output_dir=args.output,
            failure_penalty_rate=args.failure_penalty_rate,
            max_delay_factor=args.max_delay_factor,
            capacity_utilization_mean=args.capacity_util_mean,
            capacity_utilization_std=args.capacity_util_std,
            job_generation_mode=args.job_generation_mode,
            job_generation_probability=args.job_generation_probability,
            job_generation_days=job_generation_days,
        )

    # Comparison plot if we have both modes
    if agg_primary is not None and agg_secondary is not None:
        print("\n  Generating comparison plot …")
        plot_comparison(
            agg_primary=agg_primary,
            agg_secondary_rand=agg_secondary["random_cheapest"],
            agg_secondary_qual=agg_secondary["quality_top"],
            output_dir=args.output,
        )

        print("  Generating shop comparison plot …")
        plot_job_cost_comparison(
            agg_primary=agg_primary,
            agg_secondary_rand=agg_secondary["random_cheapest"],
            agg_secondary_qual=agg_secondary["quality_top"],
            output_dir=args.output,
        )
        plot_shop_comparison(
            agg_primary=agg_primary,
            agg_secondary_rand=agg_secondary["random_cheapest"],
            agg_secondary_qual=agg_secondary["quality_top"],
            output_dir=args.output,
        )

    print(f"\n✓ All done. Plots saved to: {args.output}/\n")


if __name__ == "__main__":
    main()
