"""
simulation_secondary.py — Secondary simulation mode.

Two allocation strategies:
  - "random_cheapest": pick 5 random shops with capacity, choose cheapest
  - "quality_top":     rank all shops with capacity by quality_rate, pick top
  
No re-allocation, no quality-check costs, no hub transport.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from models import Shop, Job, JobComponent, generate_shops, generate_jobs

HUB_LOCATION = (0.5, 0.5)
_AVG_BASE_LABOR_RATE = 26.0  # mean of (8 + 4*(i-1)) for job types i=1..10


def distance(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _cost_estimate(comp: JobComponent, shop: Shop) -> float:
    daily_mh = min(
        comp.max_workers * comp.max_daily_manhours_per_worker,
        comp.max_workers * shop.worker_capacity,
    )
    days = max(1, math.ceil(max(0, comp.capacity_needed - comp.manhours_done) / daily_mh)) if daily_mh > 0 else float("inf")
    labor = daily_mh * days * comp.base_labor_rate * shop.labor_cost_multiplier
    dist = distance(shop.location, comp.delivery_location)
    transport = comp.base_transportation_cost * dist
    return labor + transport


class SecondarySimulationRun:
    def __init__(
        self,
        num_shops: int = 100,
        num_days: int = 365,
        jobs_per_day: int = 5,
        allocation_method: str = "random_cheapest",  # or "quality_top"
        rng_seed: Optional[int] = None,
        failure_penalty_rate: float = 0.20,
        max_delay_factor: float = 1.0,
        capacity_utilization_mean: Optional[float] = None,
        capacity_utilization_std: Optional[float] = None,
        job_generation_mode: str = "start_only",
        job_generation_probability: float = 0.0,
        job_generation_days: Optional[list[int]] = None,
        shop_type_params_override: Optional[dict[str, dict]] = None,
        job_config: Optional[dict] = None,
    ):
        assert allocation_method in ("random_cheapest", "quality_top")
        self.num_shops = num_shops
        self.num_days = num_days
        self.jobs_per_day = jobs_per_day
        self.allocation_method = allocation_method
        self.failure_penalty_rate = max(0.0, failure_penalty_rate)
        self.max_delay_factor = max(0.0, max_delay_factor)
        self.capacity_utilization_mean = capacity_utilization_mean
        self.capacity_utilization_std = capacity_utilization_std
        self.job_generation_mode = job_generation_mode
        self.job_generation_probability = float(np.clip(job_generation_probability, 0.0, 1.0))
        self.job_generation_days = set(job_generation_days or [0])
        self.shop_type_params_override = shop_type_params_override
        self.job_config = job_config
        self.rng = np.random.default_rng(rng_seed)

        self.shops: list[Shop] = []
        self.jobs: dict[int, Job] = {}
        self.active_components: list[JobComponent] = []
        self.unassigned_pool: list[JobComponent] = []
        self.completed_jobs: list[Job] = []

        self.shop_capacity_fraction: dict[int, list[float]] = {}
        self.shop_daily_profit: dict[int, list[float]] = {}
        self.shop_busy_workers: dict[int, list[int]] = {}

        self._next_job_id = 0
        self._next_component_id = 0

    def run(self):
        self.shops = generate_shops(
            self.num_shops,
            self.rng,
            capacity_utilization_mean_override=self.capacity_utilization_mean,
            capacity_utilization_std_override=self.capacity_utilization_std,
            shop_type_params_override=self.shop_type_params_override,
        )
        for s in self.shops:
            self.shop_capacity_fraction[s.shop_id] = []
            self.shop_busy_workers[s.shop_id] = []
            s.daily_profit = []

        for day in range(self.num_days):
            self._step(day)

        for job in self.jobs.values():
            if not job.completed:
                job.timeline_success = False
                for comp in job.components:
                    if not comp.completed:
                        comp.timeline_failed = True
                        self._apply_failure_penalty(comp, "timeline")

    def _step(self, day: int):
        # 1. Generate new jobs
        jobs_today = self._jobs_to_generate_today(day)
        if jobs_today > 0:
            new_jobs, new_components, self._next_job_id, self._next_component_id = generate_jobs(
                day=day,
                num_jobs=jobs_today,
                next_job_id=self._next_job_id,
                next_component_id=self._next_component_id,
                rng=self.rng,
                max_delay_factor=self.max_delay_factor,
                job_config=self.job_config,
            )
            for job in new_jobs:
                self.jobs[job.job_id] = job
                for comp in job.components:
                    comp.total_cost += comp.material_cost
            self.unassigned_pool.extend(new_components)

        # 2. Determine which unassigned workers are busy with external work today
        self._update_busy_workers()

        # 3. Allocate
        self._allocate_pool(day)

        # 3. Daily work
        newly_completed = self._execute_daily_work(day)

        # 5. Aggregate costs
        self._aggregate_costs(day)

        # 6. Handle completions
        self._handle_completions(newly_completed, day)

        # 7. Record stats
        self._record_stats(day)

        # 8. Decrement deadlines
        for comp in self.active_components + self.unassigned_pool:
            comp.days_remaining -= 1

    def _update_busy_workers(self):
        for shop in self.shops:
            utilization = float(np.clip(
                self.rng.normal(shop.capacity_utilization_mean, shop.capacity_utilization_std),
                0.0,
                1.0,
            ))
            target_occupied = int(round(utilization * shop.num_workers))
            assigned = len(shop.assigned_components)
            desired_busy = max(0, target_occupied - assigned)
            shop.busy_workers_today = min(shop.free_workers, desired_busy)

    def _jobs_to_generate_today(self, day: int) -> int:
        if self.job_generation_mode == "start_only":
            return self.jobs_per_day if day == 0 else 0
        if self.job_generation_mode == "daily":
            return self.jobs_per_day
        if self.job_generation_mode == "probabilistic":
            return self.jobs_per_day if self.rng.random() < self.job_generation_probability else 0
        if self.job_generation_mode == "custom_days":
            return self.jobs_per_day if day in self.job_generation_days else 0
        return self.jobs_per_day if day == 0 else 0

    def _apply_failure_penalty(self, comp: JobComponent, failure_type: str):
        if self.failure_penalty_rate <= 0:
            return
        if failure_type == "quality":
            if comp.quality_failure_penalty_applied:
                return
            comp.total_cost += comp.total_cost * self.failure_penalty_rate
            comp.quality_failure_penalty_applied = True
            return
        if failure_type == "timeline":
            if comp.timeline_failure_penalty_applied:
                return
            comp.total_cost += comp.total_cost * self.failure_penalty_rate
            comp.timeline_failure_penalty_applied = True

    def _allocate_pool(self, day: int):
        remaining = []
        for comp in self.unassigned_pool:
            if comp.days_remaining <= 0:
                comp.timeline_failed = True
                self._apply_failure_penalty(comp, "timeline")
                comp.completed = True
                if comp.job_id in self.jobs:
                    self.jobs[comp.job_id].timeline_success = False
                continue

            # Find shops with capacity
            available = [s for s in self.shops if s.can_accept(comp)]
            if not available:
                remaining.append(comp)
                continue

            avoid_shop_ids: set[int] = set()
            if comp.job_id in self.jobs:
                for sibling in self.jobs[comp.job_id].components:
                    if sibling is not comp and sibling.assigned_shop is not None:
                        avoid_shop_ids.add(sibling.assigned_shop.shop_id)
            candidate_pool = [s for s in available if s.shop_id not in avoid_shop_ids] or available

            if self.allocation_method == "random_cheapest":
                sample = self.rng.choice(candidate_pool, size=min(5, len(candidate_pool)), replace=False).tolist()
                sample.sort(key=lambda s: _cost_estimate(comp, s))
                chosen = sample[0]
            else:  # quality_top
                candidate_pool.sort(key=lambda s: -s.quality_rate)
                chosen = candidate_pool[0]

            # Allocate (no reallocation in secondary mode)
            comp.assigned_shop = chosen
            chosen.assigned_components.append(comp)
            comp.shop_assignment_history.append(chosen.shop_id)
            comp.allocation_delay = True
            daily_mh = min(
                comp.max_workers * comp.max_daily_manhours_per_worker,
                comp.max_workers * chosen.worker_capacity,
            )
            comp.compute_quality_check_thresholds(daily_mh)
            self.active_components.append(comp)

        self.unassigned_pool = remaining

    def _execute_daily_work(self, day: int) -> list[JobComponent]:
        newly_completed = []
        for comp in list(self.active_components):
            if comp.allocation_delay:
                comp.allocation_delay = False
                comp.days_in_current_shop += 1
                continue

            shop = comp.assigned_shop
            daily_mh = min(
                comp.max_workers * comp.max_daily_manhours_per_worker,
                comp.max_workers * shop.worker_capacity,
            )
            comp.manhours_done += daily_mh * shop.work_efficiency
            comp.days_in_current_shop += 1

            if comp.manhours_done >= comp.capacity_needed:
                comp.manhours_done = comp.capacity_needed
                comp.completed = True
                comp.day_completed = day
                comp._completion_shop = shop
                newly_completed.append(comp)
                self.active_components.remove(comp)
                shop.assigned_components.remove(comp)
                comp.assigned_shop = None

        return newly_completed

    def _aggregate_costs(self, day: int):
        for shop in self.shops:
            day_profit = 0.0
            for comp in shop.assigned_components:
                if comp.allocation_delay:
                    continue
                daily_mh = min(
                    comp.max_workers * comp.max_daily_manhours_per_worker,
                    comp.max_workers * shop.worker_capacity,
                )
                labor_cost = daily_mh * comp.base_labor_rate * shop.labor_cost_multiplier
                comp.total_cost += labor_cost
                day_profit += labor_cost * 0.10
            # Profit from workers busy on external jobs
            if shop.busy_workers_today > 0:
                day_profit += (shop.busy_workers_today * shop.worker_capacity
                               * _AVG_BASE_LABOR_RATE * shop.labor_cost_multiplier * 0.10)
            shop.daily_profit.append(day_profit)

    def _handle_completions(self, newly_completed: list[JobComponent], day: int):
        for comp in newly_completed:
            job = self.jobs[comp.job_id]
            shop = getattr(comp, "_completion_shop", None)

            # Transport directly from shop to delivery location (no hub)
            if shop is not None:
                d = distance(shop.location, comp.delivery_location)
                comp.total_cost += comp.base_transportation_cost * d

            # Final quality check: no cost, no re-routing, but marks failure
            q_pass = shop.quality_rate if shop else 0.9
            comp.quality_checks_performed += 1
            if self.rng.random() > q_pass:
                comp.quality_failed = True
                self._apply_failure_penalty(comp, "quality")
                job.quality_success = False

            all_done = all(c.completed for c in job.components)
            if all_done and not job.completed:
                job.completed = True
                job.day_completed = day
                deadline_day = job.day_created + min(c.deadline_days for c in job.components) - 1
                job.days_late = max(0, day - deadline_day)

                min_remaining = min(c.days_remaining for c in job.components)
                job.timeline_success = (min_remaining >= 2)

                for c in job.components:
                    job.total_cost += c.total_cost

                if job.days_late > 0 and job.late_penalty_per_day > 0:
                    job.total_cost += job.total_cost * job.late_penalty_per_day * job.days_late

                self.completed_jobs.append(job)

    def _record_stats(self, day: int):
        for shop in self.shops:
            total_slots = shop.num_workers
            used_slots = len(shop.assigned_components)
            frac = used_slots / total_slots if total_slots > 0 else 0.0
            self.shop_capacity_fraction[shop.shop_id].append(frac)
            self.shop_busy_workers[shop.shop_id].append(shop.busy_workers_today)

    def get_statistics(self) -> dict:
        from collections import defaultdict
        shop_type_capacity = defaultdict(list)
        shop_type_profit_daily = defaultdict(list)
        shop_type_busy_daily = defaultdict(list)
        shop_type_assignment_counts = defaultdict(int)

        shop_type_by_id = {shop.shop_id: shop.shop_type for shop in self.shops}

        for job in self.jobs.values():
            for comp in job.components:
                for shop_id in comp.shop_assignment_history:
                    shop_type = shop_type_by_id.get(shop_id)
                    if shop_type is not None:
                        shop_type_assignment_counts[shop_type] += 1

        for shop in self.shops:
            fracs = self.shop_capacity_fraction[shop.shop_id]
            profits = shop.daily_profit
            busy = self.shop_busy_workers[shop.shop_id]
            while len(profits) < len(fracs):
                profits.append(0.0)
            shop_type_capacity[shop.shop_type].append(fracs)
            shop_type_profit_daily[shop.shop_type].append(profits)
            shop_type_busy_daily[shop.shop_type].append(busy)

        return {
            "shop_type_capacity": {k: np.array(v) for k, v in shop_type_capacity.items()},
            "shop_type_profit_daily": {k: np.array(v) for k, v in shop_type_profit_daily.items()},
            "shop_type_busy_workers_daily": {k: np.array(v) for k, v in shop_type_busy_daily.items()},
            "shop_type_assignment_counts": dict(shop_type_assignment_counts),
            "shop_type_num_workers": {
                st: int(np.mean([s.num_workers for s in self.shops if s.shop_type == st]))
                for st in set(s.shop_type for s in self.shops)
            },
            "completed_jobs": self.completed_jobs,
            "all_jobs": list(self.jobs.values()),
            "simulation_days": int(self.num_days),
            "avg_days_late": float(
                np.mean(
                    [
                        max(0, job.day_completed - (job.day_created + min(c.deadline_days for c in job.components) - 1))
                        for job in self.completed_jobs
                        if job.day_completed is not None and job.components
                    ]
                )
                if self.completed_jobs else 0.0
            ),
        }
