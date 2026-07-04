"""
stats.py — Aggregate results across multiple simulation runs.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from models import Job


def aggregate_runs(all_run_stats: list[dict]) -> dict:
    """
    Combine statistics from multiple simulation runs into grand averages.

    Parameters
    ----------
    all_run_stats : list of dicts returned by SimulationRun.get_statistics()

    Returns
    -------
    dict with aggregated statistics ready for plotting.
    """
    # ---------- Shop statistics ----------
    # shop_type_capacity[type] -> array shape (num_shops_of_type, num_days)
    # We want: mean capacity fraction per day, per type, averaged across runs

    shop_types = ["Elite", "Strong", "Average", "Risky"]

    # Collect across runs
    cap_by_type: dict[str, list] = defaultdict(list)          # type -> list of (shops, days) arrays
    profit_by_type: dict[str, list] = defaultdict(list)
    busy_by_type: dict[str, list] = defaultdict(list)         # type -> list of (busy_array, num_workers)
    usage_by_type: dict[str, list[float]] = defaultdict(list)
    run_late_means_by_type: list[dict[int, float]] = []

    for run_stats in all_run_stats:
        for st in shop_types:
            cap  = run_stats["shop_type_capacity"].get(st)
            prof = run_stats["shop_type_profit_daily"].get(st)
            busy = run_stats.get("shop_type_busy_workers_daily", {}).get(st)
            usage = run_stats.get("shop_type_assignment_counts", {}).get(st, 0.0)
            if cap is not None and cap.size > 0:
                cap_by_type[st].append(cap)          # shape (n_shops, n_days)
            if prof is not None and prof.size > 0:
                profit_by_type[st].append(prof)
            if busy is not None and busy.size > 0:
                nw = run_stats.get("shop_type_num_workers", {}).get(st, 1)
                busy_by_type[st].append((busy, max(1, int(nw))))
            usage_by_type[st].append(float(usage))

        late_by_type: dict[int, list[float]] = defaultdict(list)
        for job in run_stats.get("all_jobs", []):
            if not job.completed or job.day_completed is None or not job.components:
                continue
            late_days = float(getattr(job, "days_late", 0))
            if late_days <= 0:
                deadline_day = job.day_created + min(c.deadline_days for c in job.components) - 1
                late_days = max(0, job.day_completed - deadline_day)
            late_by_type[job.job_type_index].append(float(late_days))

        run_late_means_by_type.append(
            {
                jt: float(np.mean(vals))
                for jt, vals in late_by_type.items()
                if vals
            }
        )

    # Average capacity fraction across shops then runs
    agg_capacity: dict[str, np.ndarray] = {}
    for st in shop_types:
        if not cap_by_type[st]:
            continue
        per_run_means = [arr.mean(axis=0) for arr in cap_by_type[st]]  # each: (n_days,)
        agg_capacity[st] = np.mean(per_run_means, axis=0)              # (n_days,)

    # Busy-worker fraction (busy_count / num_workers) averaged across shops then runs
    agg_busy_fraction: dict[str, np.ndarray] = {}
    for st in shop_types:
        if not busy_by_type[st]:
            continue
        per_run_means = []
        for arr, nw in busy_by_type[st]:
            per_run_means.append(arr.mean(axis=0) / nw)
        agg_busy_fraction[st] = np.mean(per_run_means, axis=0)

    # Total utilization = assigned fraction + busy fraction
    agg_total_utilization: dict[str, np.ndarray] = {}
    for st in shop_types:
        cap  = agg_capacity.get(st)
        busy = agg_busy_fraction.get(st)
        if cap is not None and busy is not None:
            agg_total_utilization[st] = np.clip(cap + busy, 0, 1)
        elif cap is not None:
            agg_total_utilization[st] = cap

    # Mode-level shop usage frequency (average assignments per run)
    agg_shop_usage_counts: dict[str, float] = {}
    for st in shop_types:
        vals = usage_by_type.get(st, [])
        if vals:
            agg_shop_usage_counts[st] = float(np.mean(vals))

    # Profit: daily mean & variance across shops, then averaged across runs
    agg_profit_mean: dict[str, np.ndarray] = {}
    agg_profit_var: dict[str, np.ndarray] = {}
    agg_profit_total_mean: dict[str, float] = {}
    agg_profit_total_var: dict[str, float] = {}

    for st in shop_types:
        if not profit_by_type[st]:
            continue
        per_run_daily_mean = [arr.mean(axis=0) for arr in profit_by_type[st]]   # each (n_days,)
        per_run_daily_var  = [arr.var(axis=0)  for arr in profit_by_type[st]]

        agg_profit_mean[st] = np.mean(per_run_daily_mean, axis=0)
        agg_profit_var[st]  = np.mean(per_run_daily_var, axis=0)

        # Total profit per shop per run
        total_profits = [arr.sum(axis=1) for arr in profit_by_type[st]]  # each (n_shops,)
        all_totals = np.concatenate(total_profits)
        agg_profit_total_mean[st] = float(all_totals.mean())
        agg_profit_total_var[st]  = float(all_totals.var())

    # ---------- Job statistics ----------
    # Collect all jobs across all runs
    all_jobs: list[Job] = []
    for run_stats in all_run_stats:
        all_jobs.extend(run_stats["all_jobs"])

    # Group by job type
    cost_by_type:         dict[int, list[float]] = defaultdict(list)
    quality_ok_by_type:   dict[int, list[int]]   = defaultdict(list)
    quality_ok_completed_by_type: dict[int, list[int]] = defaultdict(list)
    timeline_ok_by_type:  dict[int, list[int]]   = defaultdict(list)

    # Also group by reliability targets (for success-rate plots)
    # quality_reliability_target and timeline_reliability_target are fixed per job type
    q_target_map:  dict[int, float] = {}
    t_target_map:  dict[int, float] = {}

    for job in all_jobs:
        jt = job.job_type_index
        if job.completed:
            cost_by_type[jt].append(job.total_cost)
            quality_ok_completed_by_type[jt].append(1 if job.quality_success else 0)
        # Quality success metric: passed final quality checks / total cases run.
        # Jobs not completed by horizon do not count as a final quality pass.
        quality_ok_by_type[jt].append(1 if (job.completed and job.quality_success) else 0)
        timeline_ok_by_type[jt].append(1 if job.timeline_success else 0)

        # Set targets (all jobs of the same type share them)
        if jt not in q_target_map and job.components:
            q_target_map[jt] = job.components[0].quality_reliability_target
            t_target_map[jt]  = job.components[0].timeline_reliability_target

    job_types = sorted(set(cost_by_type) | set(quality_ok_by_type) | set(timeline_ok_by_type))
    if not job_types:
        job_types = [1]

    agg_cost_mean: dict[int, float] = {}
    agg_cost_var:  dict[int, float] = {}
    agg_quality_rate:   dict[int, float] = {}
    agg_quality_rate_completed_only: dict[int, float] = {}
    agg_timeline_rate:  dict[int, float] = {}
    agg_days_late_by_type: dict[int, float] = {}

    quality_total_cases = int(len(all_jobs))
    quality_passed_cases = int(sum(1 for job in all_jobs if job.completed and job.quality_success))
    quality_failed_cases = quality_total_cases - quality_passed_cases

    completed_jobs_count = int(sum(1 for job in all_jobs if job.completed))
    failed_final_quality_count = int(sum(1 for job in all_jobs if job.completed and not job.quality_success))

    for jt in job_types:
        costs = cost_by_type.get(jt, [])
        # Include all completed jobs, regardless of whether they were on-time or late.
        agg_cost_mean[jt] = float(np.mean(costs)) if costs else 0.0
        agg_cost_var[jt]  = float(np.var(costs))  if costs else 0.0
        qok  = quality_ok_by_type.get(jt, [])
        qok_completed = quality_ok_completed_by_type.get(jt, [])
        tok  = timeline_ok_by_type.get(jt, [])
        agg_quality_rate[jt]  = float(np.mean(qok))  if qok  else 0.0
        agg_quality_rate_completed_only[jt] = float(np.mean(qok_completed)) if qok_completed else 0.0
        agg_timeline_rate[jt] = float(np.mean(tok))  if tok  else 0.0

        # Per requested definition: compute per-run days-late by job type, then average across runs.
        per_run_vals = [run_map.get(jt, 0.0) for run_map in run_late_means_by_type]
        agg_days_late_by_type[jt] = float(np.mean(per_run_vals)) if per_run_vals else 0.0

    agg_avg_days_late = (
        float(np.mean(list(agg_days_late_by_type.values())))
        if agg_days_late_by_type
        else 0.0
    )

    return dict(
        shop_types=shop_types,
        agg_capacity=agg_capacity,
        agg_busy_fraction=agg_busy_fraction,
        agg_total_utilization=agg_total_utilization,
        agg_profit_mean=agg_profit_mean,
        agg_profit_var=agg_profit_var,
        agg_profit_total_mean=agg_profit_total_mean,
        agg_profit_total_var=agg_profit_total_var,
        agg_shop_usage_counts=agg_shop_usage_counts,
        agg_days_late_by_type=agg_days_late_by_type,
        agg_avg_days_late=agg_avg_days_late,
        agg_cost_mean=agg_cost_mean,
        agg_cost_var=agg_cost_var,
        agg_quality_rate=agg_quality_rate,
        agg_quality_rate_completed_only=agg_quality_rate_completed_only,
        agg_timeline_rate=agg_timeline_rate,
        quality_total_cases=quality_total_cases,
        quality_passed_cases=quality_passed_cases,
        quality_failed_cases=quality_failed_cases,
        completed_jobs_count=completed_jobs_count,
        failed_final_quality_count=failed_final_quality_count,
        q_target_map=q_target_map,
        t_target_map=t_target_map,
        job_type_indices=job_types,
    )
