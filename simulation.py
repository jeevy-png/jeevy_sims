"""
simulation.py — Main simulation loop (full-featured mode).
"""

from __future__ import annotations
import math
import numpy as np
from itertools import permutations, product
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


def _attempt_cost_estimate(comp: JobComponent, shop: Shop) -> float:
    days = expected_days_to_complete(comp, shop)
    daily_mh = min(
        comp.max_workers * comp.max_daily_manhours_per_worker,
        comp.max_workers * shop.worker_capacity,
    )
    labor = daily_mh * days * comp.base_labor_rate * shop.labor_cost_multiplier
    dist = distance(shop.location, comp.delivery_location)
    transport = comp.base_transportation_cost * dist
    return labor + transport


def _combined_quality_rate(plan: list[Shop]) -> float:
    fail_all = 1.0
    for shop in plan:
        fail_all *= max(0.0, 1.0 - float(shop.quality_rate))
    return 1.0 - fail_all


def _expected_plan_metrics(comp: JobComponent, plan: list[Shop]) -> tuple[float, float]:
    """Return (expected_cost, expected_timeline_days) for a stacked fallback plan."""
    expected_cost = 0.0
    expected_timeline_days = 0.0
    p_reach_attempt = 1.0

    for idx, shop in enumerate(plan):
        attempt_days = 1.0 + expected_days_to_complete(comp, shop)  # 1-day allocation delay per (re-)allocation
        expected_timeline_days += p_reach_attempt * attempt_days
        expected_cost += p_reach_attempt * _attempt_cost_estimate(comp, shop)

        if idx > 0:
            prev = plan[idx - 1]
            expected_cost += p_reach_attempt * (comp.base_transportation_cost * distance(prev.location, shop.location))

        p_reach_attempt *= max(0.0, 1.0 - float(shop.quality_rate))

    return expected_cost, expected_timeline_days


def _primary_timeline_feasible(comp: JobComponent, shop: Shop, max_delay_factor: float) -> bool:
    timeline_limit = max(1.0, float(comp.days_remaining) * max(0.0, float(max_delay_factor)))
    primary_days = 1.0 + expected_days_to_complete(comp, shop)
    return primary_days <= timeline_limit


def _component_plan_options(
    comp: JobComponent,
    primary: Shop,
    candidate_pool: list[Shop],
    backup_shop_depth: int,
    max_plan_options: int = 120,
) -> list[tuple[list[Shop], float, float]]:
    """
    Enumerate bounded backup plans for one component with fixed primary shop.
    Returns tuples of (plan, expected_cost, expected_timeline_days).
    """
    depth = max(1, int(backup_shop_depth))
    others = [s for s in candidate_pool if s.shop_id != primary.shop_id]
    # Prune backup search to the most promising candidates for speed.
    others.sort(key=lambda s: (-s.quality_rate, _attempt_cost_estimate(comp, s)))
    others = others[: min(6, len(others))]
    max_backups = min(depth - 1, len(others))

    options: list[tuple[list[Shop], float, float]] = []
    for r in range(0, max_backups + 1):
        for backup_perm in permutations(others, r):
            if len(options) >= max_plan_options:
                break
            plan = [primary, *backup_perm]
            combined_quality = _combined_quality_rate(plan)
            if combined_quality < comp.quality_reliability_target:
                continue
            expected_cost, expected_timeline_days = _expected_plan_metrics(comp, plan)
            options.append((plan, expected_cost, expected_timeline_days))
        if len(options) >= max_plan_options:
            break

    # Keep bounded best options by expected timeline then expected cost.
    options.sort(key=lambda x: (x[2], x[1]))
    return options[:max_plan_options]


def find_best_shop(
    comp: JobComponent,
    shops: list[Shop],
    rng: np.random.Generator,
    avoid_shop_ids: Optional[set[int]] = None,
    backup_shop_depth: int = 3,
    max_delay_factor: float = 1.0,
    enforce_first_layer_timeline_filter: bool = False,
) -> Optional[Shop]:
    """
    1. Filter to shops with free capacity.
    2. Filter by expected timeline & quality meeting targets.
    3. Sort by cost (labor + transport), return cheapest.
    """
    available = [s for s in shops if s.can_accept(comp)]
    if not available:
        return None

    # Keep a bounded candidate pool for plan generation.
    available.sort(key=lambda s: _attempt_cost_estimate(comp, s))
    candidate_pool = available[: min(16, len(available))]

    depth = max(1, int(backup_shop_depth))
    timeline_limit = max(1.0, float(comp.days_remaining) * max(0.0, float(max_delay_factor)))

    avoid = avoid_shop_ids or set()
    best_shop = None
    best_score = float("inf")

    for primary in candidate_pool:
        if primary.work_efficiency < comp.timeline_reliability_target:
            continue
        if enforce_first_layer_timeline_filter and not _primary_timeline_feasible(comp, primary, max_delay_factor):
            continue

        plan = [primary]
        remaining = [s for s in candidate_pool if s.shop_id != primary.shop_id]
        remaining.sort(key=lambda s: (-s.quality_rate, _attempt_cost_estimate(comp, s)))

        while len(plan) < depth and _combined_quality_rate(plan) < comp.quality_reliability_target:
            if not remaining:
                break
            plan.append(remaining.pop(0))

        combined_quality = _combined_quality_rate(plan)
        if combined_quality < comp.quality_reliability_target:
            continue

        expected_cost, expected_timeline_days = _expected_plan_metrics(comp, plan)
        if expected_timeline_days > timeline_limit:
            continue

        avoid_penalty = 1e9 if primary.shop_id in avoid else 0.0
        score = expected_cost + avoid_penalty
        if score < best_score:
            best_score = score
            best_shop = primary

    if best_shop is not None:
        return best_shop

    # Sort by estimated cost (labor + transport)
    def cost_estimate(shop: Shop) -> float:
        return _attempt_cost_estimate(comp, shop)

    fallback = [s for s in candidate_pool if s.work_efficiency >= comp.timeline_reliability_target]
    if not fallback:
        return None
    fallback.sort(key=lambda s: ((s.shop_id in avoid), cost_estimate(s)))
    return fallback[0]


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
        backup_shop_depth: int = 3,
        allocation_planner_mode: str = "fast",
        enforce_first_layer_timeline_filter: bool = False,
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
        self.backup_shop_depth = max(1, int(backup_shop_depth))
        self.allocation_planner_mode = allocation_planner_mode if allocation_planner_mode in ("fast", "thorough") else "fast"
        self.enforce_first_layer_timeline_filter = bool(enforce_first_layer_timeline_filter)
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

    def _can_reroute_now(self, comp: JobComponent, current_shop: Optional[Shop]) -> bool:
        if comp.days_remaining <= 0:
            return False

        avoid = set()
        if current_shop is not None:
            avoid.add(current_shop.shop_id)

        candidate = find_best_shop(
            comp,
            self.shops,
            self.rng,
            avoid_shop_ids=avoid,
            backup_shop_depth=self.backup_shop_depth,
            max_delay_factor=self.max_delay_factor,
            enforce_first_layer_timeline_filter=self.enforce_first_layer_timeline_filter,
        )
        return candidate is not None

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

        # Only mark final quality failure when max-delay window is exceeded.
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
        remaining: list[JobComponent] = []

        # Planner caps: thorough explores more combinations at higher runtime cost.
        planner_caps = {
            "fast": {
                "candidate_pool_base": 6,
                "primary_assignments_base": 240,
                "per_component_plan_options": 18,
                "combo_eval_base": 900,
            },
            "thorough": {
                "candidate_pool_base": 10,
                "primary_assignments_base": 1200,
                "per_component_plan_options": 40,
                "combo_eval_base": 4000,
            },
        }
        caps = planner_caps[self.allocation_planner_mode]

        # 1) Handle hard deadline misses immediately.
        valid_pool: list[JobComponent] = []
        for comp in self.unassigned_pool:
            if comp.days_remaining <= 0:
                comp.timeline_failed = True
                self._apply_failure_penalty(comp, "timeline")
                comp.completed = True
                job = self.jobs[comp.job_id]
                job.timeline_success = False
            else:
                valid_pool.append(comp)

        # 2) Re-allocations still use per-component planner.
        reallocation_pool = [c for c in valid_pool if c.prev_shop is not None]
        initial_pool = [c for c in valid_pool if c.prev_shop is None]

        # 3) Initial allocations: evaluate bounded split/shop/backup combinations per job.
        by_job: dict[int, list[JobComponent]] = {}
        for comp in initial_pool:
            by_job.setdefault(comp.job_id, []).append(comp)

        for job_id, comps in by_job.items():
            comps = sorted(comps, key=lambda c: c.component_id)
            if not comps:
                continue

            # Available capacity slots today.
            slots_by_shop: dict[int, int] = {}
            for shop in self.shops:
                slots_by_shop[shop.shop_id] = max(0, shop.free_workers - shop.busy_workers_today)

            if sum(slots_by_shop.values()) < len(comps):
                remaining.extend(comps)
                continue

            # Bounded candidate pool for combinatorial planning.
            baseline_comp = comps[0]
            available_shops = [s for s in self.shops if slots_by_shop.get(s.shop_id, 0) > 0 and s.can_accept(baseline_comp)]
            if self.enforce_first_layer_timeline_filter:
                available_shops = [
                    s for s in available_shops
                    if _primary_timeline_feasible(baseline_comp, s, self.max_delay_factor)
                ]
            if not available_shops:
                remaining.extend(comps)
                continue

            available_shops.sort(key=lambda s: _attempt_cost_estimate(baseline_comp, s))
            # Keep combinatorics manageable while preserving diversity.
            candidate_pool_size = min(len(available_shops), max(4, min(caps["candidate_pool_base"], len(comps) + 2)))
            candidate_pool = available_shops[:candidate_pool_size]

            # Enumerate bounded primary assignments with capacity feasibility.
            primary_assignments: list[tuple[Shop, ...]] = []
            max_primary_assignments = min(caps["primary_assignments_base"], max(60, 40 * len(comps)))

            def _dfs_primary(idx: int, cur: list[Shop], local_slots: dict[int, int]):
                if len(primary_assignments) >= max_primary_assignments:
                    return
                if idx == len(comps):
                    primary_assignments.append(tuple(cur))
                    return
                for shop in candidate_pool:
                    sid = shop.shop_id
                    if local_slots.get(sid, 0) <= 0:
                        continue
                    local_slots[sid] -= 1
                    cur.append(shop)
                    _dfs_primary(idx + 1, cur, local_slots)
                    cur.pop()
                    local_slots[sid] += 1

            _dfs_primary(0, [], dict(slots_by_shop))
            if not primary_assignments:
                remaining.extend(comps)
                continue

            best_choice = None
            best_key = (float("inf"), float("inf"))
            min_days_remaining = float(min(c.days_remaining for c in comps))

            for assignment in primary_assignments:
                per_comp_options: list[list[tuple[list[Shop], float, float]]] = []
                invalid = False
                for comp, primary in zip(comps, assignment):
                    if self.enforce_first_layer_timeline_filter and not _primary_timeline_feasible(comp, primary, self.max_delay_factor):
                        invalid = True
                        break
                    opts = _component_plan_options(
                        comp=comp,
                        primary=primary,
                        candidate_pool=candidate_pool,
                        backup_shop_depth=self.backup_shop_depth,
                        max_plan_options=caps["per_component_plan_options"],
                    )
                    if not opts:
                        invalid = True
                        break
                    per_comp_options.append(opts)
                if invalid:
                    continue

                # Cartesian product across component plan options, bounded.
                max_combo_eval = min(caps["combo_eval_base"], max(180, 120 * len(comps)))
                evaluated = 0
                for combo in product(*per_comp_options):
                    evaluated += 1
                    if evaluated > max_combo_eval:
                        break

                    expected_job_days = max(opt[2] for opt in combo)
                    expected_days_late = max(0.0, expected_job_days - min_days_remaining)

                    # Delay-centric objective requested by user.
                    cost_of_being_late = 0.0
                    total_expected_cost = 0.0
                    for comp, opt in zip(comps, combo):
                        expected_cost = opt[1]
                        total_expected_cost += expected_cost
                        cost_of_being_late += expected_cost * max(0.0, comp.late_penalty_per_day)

                    delay_cost = expected_days_late * cost_of_being_late
                    key = (delay_cost, total_expected_cost)
                    if key < best_key:
                        best_key = key
                        best_choice = (assignment, combo)

            if best_choice is None:
                remaining.extend(comps)
                continue

            assignment, combo = best_choice
            for comp, primary, opt in zip(comps, assignment, combo):
                allocate_component(comp, primary, day, False, self.shops, self.num_quality_checks)
                # Keep selected plan for debugging/inspection.
                comp.selected_backup_plan = [s.shop_id for s in opt[0]]
                self.active_components.append(comp)

        # 4) Re-allocation path (single component at a time).
        for comp in reallocation_pool:
            best = find_best_shop(
                comp,
                self.shops,
                self.rng,
                avoid_shop_ids={comp.prev_shop.shop_id} if comp.prev_shop is not None else set(),
                backup_shop_depth=self.backup_shop_depth,
                max_delay_factor=self.max_delay_factor,
                enforce_first_layer_timeline_filter=self.enforce_first_layer_timeline_filter,
            )
            if best is None:
                remaining.append(comp)
            else:
                allocate_component(comp, best, day, True, self.shops, self.num_quality_checks)
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
                deadline_day = job.day_created + min(c.deadline_days for c in job.components) - 1
                job.days_late = max(0, day - deadline_day)

                # Timeline success: completed at least 2 days before deadline.
                days_early = deadline_day - day
                job.timeline_success = (days_early >= 2)

                for c in job.components:
                    job.total_cost += c.total_cost

                if job.days_late > 0 and job.late_penalty_per_day > 0:
                    job.total_cost += job.total_cost * job.late_penalty_per_day * job.days_late

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

        stats = {
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
        return stats
