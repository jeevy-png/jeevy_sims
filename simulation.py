"""
simulation.py — Main simulation loop (full-featured mode).
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


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

def expected_days_to_complete(comp: JobComponent, shop: Shop) -> float:
    """Estimated working days to finish the component at this shop."""
    daily_mh = min(
        comp.max_workers * comp.max_daily_manhours_per_worker,
        comp.max_workers * shop.worker_capacity,
    ) * shop.work_efficiency
    if daily_mh <= 0:
        return float("inf")
    remaining = max(0.0, comp.capacity_needed - comp.manhours_done)
    return remaining / daily_mh


def find_best_shop(
    comp: JobComponent,
    shops: list[Shop],
    rng: np.random.Generator,
) -> Optional[Shop]:
    """
    1. Filter to shops with free capacity.
    2. Filter by expected timeline & quality meeting targets.
    3. Sort by cost (labor + transport), return cheapest.
    """
    candidates = []
    for shop in shops:
        if not shop.can_accept(comp):
            continue

        # --- timeline feasibility: can it finish before deadline? ---
        days_needed = expected_days_to_complete(comp, shop) + 1  # +1 for allocation delay
        if days_needed > comp.days_remaining:
            continue

        # --- threshold by reliability targets ---
        if shop.quality_rate < comp.quality_reliability_target:
            continue
        if shop.work_efficiency < comp.timeline_reliability_target:
            continue

        candidates.append(shop)

    if not candidates:
        return None

    # Sort by estimated cost (labor + transport)
    def cost_estimate(shop: Shop) -> float:
        days = expected_days_to_complete(comp, shop)
        daily_mh = min(
            comp.max_workers * comp.max_daily_manhours_per_worker,
            comp.max_workers * shop.worker_capacity,
        )
        labor = daily_mh * days * comp.base_labor_rate * shop.labor_cost_multiplier
        # transport from shop to delivery location
        dist = distance(shop.location, comp.delivery_location)
        transport = comp.base_transportation_cost * dist
        return labor + transport

    candidates.sort(key=cost_estimate)
    return candidates[0]


def allocate_component(
    comp: JobComponent,
    shop: Shop,
    current_day: int,
    is_reallocation: bool,
    shops: list[Shop],
    num_quality_checks: int = 3,
):
    """Assign component to shop; handle reallocation transport costs."""
    if is_reallocation and comp.prev_shop is not None:
        d = distance(comp.prev_shop.location, shop.location)
        transport_cost = comp.base_transportation_cost * d
        comp.total_cost += transport_cost

    comp.prev_shop = comp.assigned_shop
    if comp.assigned_shop is not None:
        comp.assigned_shop.assigned_components.remove(comp)

    comp.assigned_shop = shop
    shop.assigned_components.append(comp)
    comp.shop_assignment_history.append(shop.shop_id)
    comp.days_in_current_shop = 0
    comp.allocation_delay = True  # 1-day delay before work starts

    daily_mh = min(
        comp.max_workers * comp.max_daily_manhours_per_worker,
        comp.max_workers * shop.worker_capacity,
    )
    comp.compute_quality_check_thresholds(daily_mh, num_quality_checks)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

class SimulationRun:
    def __init__(
        self,
        num_shops: int = 100,
        num_days: int = 365,
        jobs_per_day: int = 5,
        rng_seed: Optional[int] = None,
        num_quality_checks: int = 3,
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
        self.num_shops = num_shops
        self.num_days = num_days
        self.jobs_per_day = jobs_per_day
        self.num_quality_checks = num_quality_checks
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

        # State
        self.shops: list[Shop] = []
        self.jobs: dict[int, Job] = {}
        self.active_components: list[JobComponent] = []   # assigned and working
        self.unassigned_pool: list[JobComponent] = []     # need assignment
        self.completed_jobs: list[Job] = []

        # Statistics
        self.shop_capacity_fraction: dict[int, list[float]] = {}  # shop_id -> [day0, day1, ...]
        self.shop_daily_profit: dict[int, list[float]] = {}
        self.shop_busy_workers: dict[int, list[int]] = {}          # shop_id -> busy count per day

        self._next_job_id = 0
        self._next_component_id = 0

    # ------------------------------------------------------------------
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
            self.shop_daily_profit[s.shop_id] = []
            self.shop_busy_workers[s.shop_id] = []

        for day in range(self.num_days):
            self._step(day)

        # Mark any remaining incomplete jobs
        for job in self.jobs.values():
            if not job.completed:
                for comp in job.components:
                    if not comp.completed:
                        comp.timeline_failed = True
                        self._apply_failure_penalty(comp, "timeline")
                job.timeline_success = False

    # ------------------------------------------------------------------
    def _step(self, day: int):
        # 1. Quality checks
        self._perform_quality_checks(day)

        # 2. Generate new jobs
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
                # Add material costs immediately
                for comp in job.components:
                    comp.total_cost += comp.material_cost
            self.unassigned_pool.extend(new_components)

        # 3. Determine which unassigned workers are busy with external work today
        self._update_busy_workers()

        # 4. Allocate all unassigned components
        self._allocate_pool(day)

        # 4. Daily work
        newly_completed = self._execute_daily_work(day)

        # 6. Aggregate costs (labor, profit)
        self._aggregate_costs(day)

        # 7. Handle completed components / jobs
        self._handle_completions(newly_completed, day)

        # 8. Track statistics
        self._record_stats(day)

        # 9. Decrement deadlines
        for comp in self.active_components + self.unassigned_pool:
            comp.days_remaining -= 1

    # ------------------------------------------------------------------
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

    def _handle_quality_failure(self, comp: JobComponent, shop: Shop) -> bool:
        """Returns True if component should be re-routed, False if it should continue."""
        comp.quality_failure_count += 1
        elapsed = comp.deadline_days - comp.days_remaining
        if elapsed <= comp.max_delay:
            comp.manhours_done *= 0.7
            comp.quality_checks_done = 0
            comp.compute_quality_check_thresholds(
                min(comp.max_workers * comp.max_daily_manhours_per_worker,
                    comp.max_workers * shop.worker_capacity),
                self.num_quality_checks,
            )
            return True

        comp.quality_failed = True
        self._apply_failure_penalty(comp, "quality")
        self.jobs[comp.job_id].quality_success = False
        return False

    # ------------------------------------------------------------------
    def _perform_quality_checks(self, day: int):
        # Only intermediate checks here; the final check (at 100%) fires in _execute_daily_work.
        num_intermediate = self.num_quality_checks - 1
        for comp in list(self.active_components):
            if comp.allocation_delay:
                continue
            if comp.quality_checks_done >= num_intermediate:
                continue
            shop = comp.assigned_shop

            while (
                comp.quality_checks_done < num_intermediate
                and comp.manhours_done >= comp.quality_check_manhour_thresholds[comp.quality_checks_done]
            ):
                comp.total_cost += comp.quality_cost
                comp.quality_checks_performed += 1
                if self.rng.random() > comp.per_check_quality_pass_prob:
                    if self._handle_quality_failure(comp, shop):
                        self.active_components.remove(comp)
                        shop.assigned_components.remove(comp)
                        comp.assigned_shop = None
                        comp.prev_shop = shop
                        self.unassigned_pool.append(comp)
                        break
                else:
                    comp.quality_checks_done += 1

    # ------------------------------------------------------------------
    def _allocate_pool(self, day: int):
        remaining = []
        for comp in self.unassigned_pool:
            if comp.days_remaining <= 0:
                # Deadline passed before allocation
                comp.timeline_failed = True
                self._apply_failure_penalty(comp, "timeline")
                comp.completed = True
                job = self.jobs[comp.job_id]
                job.timeline_success = False
                continue

            is_realloc = comp.prev_shop is not None
            best = find_best_shop(comp, self.shops, self.rng)
            if best is None:
                # No suitable shop found; try again next day
                remaining.append(comp)
            else:
                allocate_component(comp, best, day, is_realloc, self.shops, self.num_quality_checks)
                self.active_components.append(comp)

        self.unassigned_pool = remaining

    # ------------------------------------------------------------------
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
                newly_completed.append(comp)
                self.active_components.remove(comp)
                shop.assigned_components.remove(comp)
                comp.assigned_shop = None

        return newly_completed

    # ------------------------------------------------------------------
    def _aggregate_costs(self, day: int):
        # Ensure each shop has a profit entry for today
        for shop in self.shops:
            while len(shop.daily_profit) <= day:
                shop.daily_profit.append(0.0)

        for comp in self.active_components:
            if comp.allocation_delay:
                continue
            shop = comp.assigned_shop
            if shop is None:
                continue
            daily_mh = min(
                comp.max_workers * comp.max_daily_manhours_per_worker,
                comp.max_workers * shop.worker_capacity,
            )
            labor_cost = daily_mh * comp.base_labor_rate * shop.labor_cost_multiplier
            comp.total_cost += labor_cost
            # Shop profit = 10% of labor costs
            shop.daily_profit[day] += labor_cost * 0.10

        # Profit from workers busy on external jobs
        for shop in self.shops:
            if shop.busy_workers_today > 0:
                busy_profit = (shop.busy_workers_today * shop.worker_capacity
                               * _AVG_BASE_LABOR_RATE * shop.labor_cost_multiplier * 0.10)
                shop.daily_profit[day] += busy_profit

    # ------------------------------------------------------------------
    def _handle_completions(self, newly_completed: list[JobComponent], day: int):
        for comp in newly_completed:
            job = self.jobs[comp.job_id]
            last_shop = comp.prev_shop  # last assigned shop (set when removed)
            # Recover last shop from component history
            # Actually we need to track which shop it just finished at
            # We'll use a small workaround: store shop at completion
            # (We already removed from shop.assigned_components, but comp.prev_shop may be stale)
            # Let's check if we stored it correctly; in _execute_daily_work we set comp.assigned_shop=None
            # but we didn't update prev_shop there. Let's use a completion_shop field we'll add implicitly.
            # We'll rely on comp._completion_shop set in the work step:
            shop = getattr(comp, "_completion_shop", None)
            if shop is None:
                # fallback: use prev_shop
                shop = comp.prev_shop

            # Transport directly from shop to delivery location
            if shop is not None:
                d_to_delivery = distance(shop.location, comp.delivery_location)
                comp.total_cost += comp.base_transportation_cost * d_to_delivery

            # Check if entire job is done
            all_done = all(c.completed for c in job.components)
            if all_done and not job.completed:
                job.completed = True
                job.day_completed = day

                # Timeline success: completed at least 2 days before deadline
                min_remaining = min(c.days_remaining for c in job.components)
                job.timeline_success = (min_remaining >= 2)

                for c in job.components:
                    job.total_cost += c.total_cost

                self.completed_jobs.append(job)

    # ------------------------------------------------------------------
    def _record_stats(self, day: int):
        for shop in self.shops:
            total_slots = shop.num_workers
            used_slots = len(shop.assigned_components)
            frac = used_slots / total_slots if total_slots > 0 else 0.0
            self.shop_capacity_fraction[shop.shop_id].append(frac)
            self.shop_busy_workers[shop.shop_id].append(shop.busy_workers_today)
            # Ensure profit list has an entry for today (in case no labor happened)
            while len(shop.daily_profit) <= day:
                shop.daily_profit.append(0.0)

    # ------------------------------------------------------------------
    def _execute_daily_work(self, day: int) -> list[JobComponent]:
        """Overrides earlier definition to also track completion shop."""
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
                comp.total_cost += comp.quality_cost
                comp.quality_checks_performed += 1
                if self.rng.random() > comp.per_check_quality_pass_prob:
                    if self._handle_quality_failure(comp, shop):
                        self.active_components.remove(comp)
                        shop.assigned_components.remove(comp)
                        comp.assigned_shop = None
                        comp.prev_shop = shop
                        self.unassigned_pool.append(comp)
                    else:
                        comp.completed = True
                        comp.day_completed = day
                        comp._completion_shop = shop
                        newly_completed.append(comp)
                        self.active_components.remove(comp)
                        shop.assigned_components.remove(comp)
                        comp.assigned_shop = None
                else:
                    comp.completed = True
                    comp.day_completed = day
                    comp._completion_shop = shop
                    newly_completed.append(comp)
                    self.active_components.remove(comp)
                    shop.assigned_components.remove(comp)
                    comp.assigned_shop = None

        return newly_completed

    # ------------------------------------------------------------------
    def get_statistics(self) -> dict:
        """Aggregate statistics for analysis."""
        from collections import defaultdict

        # Per shop-type capacity, profit, and busy workers
        shop_type_capacity = defaultdict(list)
        shop_type_profit_daily = defaultdict(list)
        shop_type_busy_daily = defaultdict(list)

        for shop in self.shops:
            fracs = self.shop_capacity_fraction[shop.shop_id]
            profits = shop.daily_profit
            busy = self.shop_busy_workers[shop.shop_id]

            while len(profits) < len(fracs):
                profits.append(0.0)

            shop_type_capacity[shop.shop_type].append(fracs)
            shop_type_profit_daily[shop.shop_type].append(profits)
            shop_type_busy_daily[shop.shop_type].append(busy)

        stats = {
            "shop_type_capacity": {k: np.array(v) for k, v in shop_type_capacity.items()},
            "shop_type_profit_daily": {k: np.array(v) for k, v in shop_type_profit_daily.items()},
            "shop_type_busy_workers_daily": {k: np.array(v) for k, v in shop_type_busy_daily.items()},
            "shop_type_num_workers": {
                st: int(np.mean([s.num_workers for s in self.shops if s.shop_type == st]))
                for st in set(s.shop_type for s in self.shops)
            },
            "completed_jobs": self.completed_jobs,
            "all_jobs": list(self.jobs.values()),
        }
        return stats
