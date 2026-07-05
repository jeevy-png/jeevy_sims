"""
models.py — Data classes for shops, jobs, and job components.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------

SHOP_TYPE_PARAMS = {
    "Elite": dict(
        num_workers=9,
        quality_rate_mean=0.995, quality_rate_std=0.003,
        work_efficiency_mean=0.98,  work_efficiency_std=0.01,
        worker_capacity_mean=10,  worker_capacity_std=1.0,
        labor_cost_multiplier_mean=1.25, labor_cost_multiplier_std=0.05,
        capacity_utilization_mean=0.95, capacity_utilization_std=0.10,
        fraction=0.10,
    ),
    "Strong": dict(
        num_workers=6,
        quality_rate_mean=0.985, quality_rate_std=0.01,
        work_efficiency_mean=0.96,  work_efficiency_std=0.02,
        worker_capacity_mean=10,  worker_capacity_std=1.22,
        labor_cost_multiplier_mean=1.0,  labor_cost_multiplier_std=0.1,
        capacity_utilization_mean=0.90, capacity_utilization_std=0.10,
        fraction=0.30,
    ),
    "Average": dict(
        num_workers=5,
        quality_rate_mean=0.94,  quality_rate_std=0.03,
        work_efficiency_mean=0.92,  work_efficiency_std=0.03,
        worker_capacity_mean=10,  worker_capacity_std=1.79,
        labor_cost_multiplier_mean=0.9,  labor_cost_multiplier_std=0.15,
        capacity_utilization_mean=0.65, capacity_utilization_std=0.10,
        fraction=0.40,
    ),
    "Risky": dict(
        num_workers=5,
        quality_rate_mean=0.88,  quality_rate_std=0.05,
        work_efficiency_mean=0.85,  work_efficiency_std=0.05,
        worker_capacity_mean=10,  worker_capacity_std=1.79,
        labor_cost_multiplier_mean=0.8,  labor_cost_multiplier_std=0.2,
        capacity_utilization_mean=0.50, capacity_utilization_std=0.10,
        fraction=0.20,
    ),
}


DEFAULT_JOB_CONFIG = {
    "job_type_min_index": 1,
    "job_type_max_index": 10,
    "job_type_indices": [],
    "job_type_overrides": {},
    "num_components_divisor": 5,
    "deadline_base": 6,
    "deadline_step": 7,
    "capacity_base": 200,
    "capacity_step": 111,
    "max_workers": 2,
    "quality_cost_base": 800,
    "quality_cost_step": 44,
    "material_cost_base": 8000,
    "material_cost_step": 4444,
    "max_daily_manhours_per_worker": 10,
    "base_labor_rate_base": 8,
    "base_labor_rate_step": 4,
    "base_transportation_cost_base": 2000,
    "base_transportation_cost_step": 1111,
    "late_penalty_per_day_base": 0.01,
    "late_penalty_per_day_step": 0.0,
    "timeline_reliability_target_base": 0.8,
    "timeline_reliability_target_step": 0.01,
    "quality_reliability_target_base": 0.94,
    "quality_reliability_target_step": 0.005,
}


@dataclass
class Shop:
    shop_id: int
    shop_type: str
    num_workers: int
    quality_rate: float
    work_efficiency: float       # fraction of man-hours that count as useful progress
    worker_capacity: float       # man-hours per worker per day
    labor_cost_multiplier: float
    capacity_utilization_mean: float  # mean external utilization fraction
    capacity_utilization_std: float   # std dev external utilization fraction
    location: tuple              # (x, y) in [0,1]^2

    # runtime state
    assigned_components: list = field(default_factory=list)  # list of JobComponent
    daily_profit: list = field(default_factory=list)         # profit per day
    busy_workers_today: int = 0  # unassigned workers busy with external work today (set each step)

    @property
    def free_workers(self) -> int:
        return self.num_workers - len(self.assigned_components)

    @property
    def daily_capacity(self) -> float:
        """Total man-hours this shop can deliver per day across all workers."""
        return self.num_workers * self.worker_capacity

    def can_accept(self, component: "JobComponent") -> bool:
        """True if at least one unassigned worker is available (not busy with external work)."""
        return (self.free_workers - self.busy_workers_today) >= 1

    def expected_quality_rate_for_component(self, component: "JobComponent") -> float:
        return self.quality_rate


@dataclass
class JobComponent:
    component_id: int
    job_id: int
    job_type_index: int          # 1..10

    deadline_days: int
    max_delay: int
    max_workers: int
    capacity_needed: float       # man-hours to complete
    quality_cost: float
    material_cost: float
    base_labor_rate: float
    base_transportation_cost: float
    late_penalty_per_day: float
    max_daily_manhours_per_worker: float
    timeline_reliability_target: float
    quality_reliability_target: float
    delivery_location: tuple     # (x, y)

    # runtime state
    manhours_done: float = 0.0
    days_remaining: int = 0      # set when component enters simulation (= deadline_days)
    assigned_shop: Optional[Shop] = None
    prev_shop: Optional[Shop] = None   # last shop before re-allocation
    days_in_current_shop: int = 0      # days since last allocation (includes 1-day delay)

    # tracking
    total_cost: float = 0.0
    material_cost_total: float = 0.0
    labor_cost_total: float = 0.0
    transport_cost_total: float = 0.0
    quality_cost_total: float = 0.0
    failure_penalty_cost_total: float = 0.0
    completed: bool = False
    quality_failed: bool = False       # failed final hub check
    timeline_failed: bool = False
    day_completed: Optional[int] = None

    # quality check bookkeeping
    quality_check_manhour_thresholds: list = field(default_factory=list)
    quality_checks_done: int = 0
    quality_checks_total: int = 3       # configurable; set by compute_quality_check_thresholds
    quality_failure_count: int = 0      # number of quality check failures so far
    quality_checks_performed: int = 0   # total number of quality checks run on this component
    shop_assignment_history: list[int] = field(default_factory=list)
    quality_failure_penalty_applied: bool = False
    timeline_failure_penalty_applied: bool = False

    # flag: waiting 1-day before work starts after (re-)allocation
    allocation_delay: bool = True

    def is_feasible(self) -> bool:
        """
        A component is feasible if, in theory, max_workers working at
        max_daily_manhours_per_worker every day can finish it by deadline.
        """
        max_possible = self.max_workers * self.max_daily_manhours_per_worker * self.deadline_days
        return max_possible >= self.capacity_needed

    def compute_quality_check_thresholds(self, daily_work_rate: float, num_checks: int = 3):
        """
        Place num_checks evenly at k/num_checks * capacity for k=1..num_checks.
        The last threshold always lands at capacity_needed (handled at completion).
        """
        self.quality_checks_total = num_checks
        self.quality_check_manhour_thresholds = [
            (k / num_checks) * self.capacity_needed
            for k in range(1, num_checks + 1)
        ]
        self.quality_checks_done = 0

    @property
    def per_check_quality_pass_prob(self) -> float:
        """P(pass one check) such that P(all num_checks pass) = shop.quality_rate."""
        if self.assigned_shop is None:
            return 1.0
        return self.assigned_shop.quality_rate ** (1.0 / max(1, self.quality_checks_total))

    def workers_assigned(self) -> int:
        """Number of workers from the shop working on this component (always 1 slot = 1 worker)."""
        return min(self.max_workers, 1)   # each component occupies 1 worker slot


@dataclass
class Job:
    job_id: int
    job_type_index: int          # 1..10
    components: list             # list of JobComponent
    day_created: int

    total_cost: float = 0.0
    material_cost_total: float = 0.0
    labor_cost_total: float = 0.0
    transport_cost_total: float = 0.0
    quality_cost_total: float = 0.0
    failure_penalty_cost_total: float = 0.0
    late_penalty_cost_total: float = 0.0
    completed: bool = False
    quality_success: bool = True
    timeline_success: bool = False
    day_completed: Optional[int] = None
    late_penalty_per_day: float = 0.01
    days_late: int = 0


# ---------------------------------------------------------------------------
# Job generation
# ---------------------------------------------------------------------------

def _resolved_job_config(job_config: Optional[dict] = None) -> dict:
    cfg = dict(DEFAULT_JOB_CONFIG)
    if job_config:
        cfg.update(job_config)
    if not isinstance(cfg.get("job_type_indices", []), list):
        cfg["job_type_indices"] = []
    if not isinstance(cfg.get("job_type_overrides", {}), dict):
        cfg["job_type_overrides"] = {}
    return cfg


def _job_type_indices(job_config: Optional[dict] = None) -> list[int]:
    cfg = _resolved_job_config(job_config)
    explicit = [int(x) for x in cfg.get("job_type_indices", []) if int(x) >= 1]
    if explicit:
        return sorted(set(explicit))

    min_idx = max(1, int(cfg.get("job_type_min_index", 1)))
    max_idx = max(min_idx, int(cfg.get("job_type_max_index", 10)))
    return list(range(min_idx, max_idx + 1))


def _job_type_override_for(i: int, job_config: Optional[dict] = None) -> dict:
    cfg = _resolved_job_config(job_config)
    overrides = cfg.get("job_type_overrides", {})
    return overrides.get(str(i), overrides.get(i, {}))


def _average_shop_profile() -> dict[str, float]:
    total_fraction = sum(v.get("fraction", 0.0) for v in SHOP_TYPE_PARAMS.values())
    total_fraction = total_fraction if total_fraction > 0 else 1.0

    def wavg(key: str) -> float:
        return sum(v.get(key, 0.0) * v.get("fraction", 0.0) for v in SHOP_TYPE_PARAMS.values()) / total_fraction

    return {
        "worker_capacity": wavg("worker_capacity_mean"),
        "work_efficiency": wavg("work_efficiency_mean"),
        "labor_cost_multiplier": wavg("labor_cost_multiplier_mean"),
    }


def _timeline_success_proxy(estimated_days_to_finish: float, deadline_days: float) -> float:
    slack = float(deadline_days) - float(estimated_days_to_finish)
    return 1.0 / (1.0 + math.exp(-slack))


def _predicted_split_objective(
    total_capacity: float,
    deadline_days: int,
    max_workers: int,
    max_daily_manhours_per_worker: float,
    base_labor_rate: float,
    base_transportation_cost: float,
    late_penalty_per_day: float,
    num_components: int,
) -> tuple[float, float]:
    """
    Predict weighted split objective as:
    expected_total_cost + expected_late_penalty_cost,
    while accounting for extra shipping introduced by splitting.
    """
    profile = _average_shop_profile()
    num_components = max(1, int(num_components))

    cap_per_component = total_capacity / num_components
    daily_mh = (
        min(max_workers * max_daily_manhours_per_worker, max_workers * profile["worker_capacity"])
        * profile["work_efficiency"]
    )
    if daily_mh <= 0:
        return float("inf"), 0.0

    # Components can run in parallel across different shops; each component has its own 1-day allocation delay.
    estimated_days = 1.0 + (cap_per_component / daily_mh)
    p_timeline = _timeline_success_proxy(estimated_days, deadline_days)
    expected_late_days = max(0.0, estimated_days - deadline_days)

    labor_cost = total_capacity * base_labor_rate * profile["labor_cost_multiplier"]

    # Extra shipping grows with split count (one delivery leg per component).
    expected_distance = 0.52  # expected Euclidean distance in unit square (approx)
    shipping_cost = num_components * base_transportation_cost * expected_distance

    base_cost = labor_cost + shipping_cost
    expected_late_cost = base_cost * late_penalty_per_day * expected_late_days

    # Weighted prediction balances expected lateness exposure and shipping overhead.
    weighted_objective = base_cost + expected_late_cost
    return weighted_objective, p_timeline


def _select_num_components(i: int, cfg: dict, override: dict) -> int:
    if "num_components" in override:
        return max(1, int(override["num_components"]))

    divisor = max(1, int(cfg["num_components_divisor"]))
    if divisor <= 1:
        return 1

    total_capacity = float(cfg["capacity_base"] + cfg["capacity_step"] * (i - 1))
    deadline_days = max(1, int(math.ceil(cfg["deadline_base"] + i * cfg["deadline_step"])))
    base_labor_rate = float(cfg["base_labor_rate_base"] + cfg["base_labor_rate_step"] * (i - 1))
    base_transportation_cost = float(cfg["base_transportation_cost_base"] + cfg["base_transportation_cost_step"] * (i - 1))
    late_penalty_per_day = float(cfg.get("late_penalty_per_day_base", 0.01) + cfg.get("late_penalty_per_day_step", 0.0) * (i - 1))

    max_workers = int(cfg["max_workers"])
    max_daily_manhours_per_worker = float(cfg["max_daily_manhours_per_worker"])

    max_candidates = max(2, min(divisor, 12))
    candidates = list(range(1, max_candidates + 1))

    baseline_objective, baseline_p = _predicted_split_objective(
        total_capacity,
        deadline_days,
        max_workers,
        max_daily_manhours_per_worker,
        base_labor_rate,
        base_transportation_cost,
        late_penalty_per_day,
        1,
    )
    best_components = 1
    best_objective = baseline_objective

    for n in candidates[1:]:
        objective, p_timeline = _predicted_split_objective(
            total_capacity,
            deadline_days,
            max_workers,
            max_daily_manhours_per_worker,
            base_labor_rate,
            base_transportation_cost,
            late_penalty_per_day,
            n,
        )
        if p_timeline >= baseline_p and objective < best_objective:
            best_objective = objective
            best_components = n

    return best_components


def job_type_params(i: int, rng: np.random.Generator, job_config: Optional[dict] = None) -> dict:
    """Return raw parameters for job type i (1-indexed)."""
    cfg = _resolved_job_config(job_config)
    override = _job_type_override_for(i, job_config)
    num_components = _select_num_components(i, cfg, override)
    total_capacity = cfg["capacity_base"] + cfg["capacity_step"] * (i - 1)
    cap_per_component = total_capacity / num_components
    deadline_days = math.ceil(cfg["deadline_base"] + i * cfg["deadline_step"])

    params = dict(
        job_type_index=i,
        num_components=num_components,
        deadline_days=deadline_days,
        capacity_needed=cap_per_component,
        max_workers=int(cfg["max_workers"]),
        quality_cost=cfg["quality_cost_base"] + cfg["quality_cost_step"] * (i - 1),
        material_cost=cfg["material_cost_base"] + cfg["material_cost_step"] * (i - 1),
        late_penalty_per_day=cfg.get("late_penalty_per_day_base", 0.01) + cfg.get("late_penalty_per_day_step", 0.0) * (i - 1),
        max_daily_manhours_per_worker=cfg["max_daily_manhours_per_worker"],
        base_labor_rate=cfg["base_labor_rate_base"] + cfg["base_labor_rate_step"] * (i - 1),
        base_transportation_cost=cfg["base_transportation_cost_base"] + cfg["base_transportation_cost_step"] * (i - 1),
        timeline_reliability_target=cfg["timeline_reliability_target_base"] + cfg["timeline_reliability_target_step"] * i,
        quality_reliability_target=cfg["quality_reliability_target_base"] + cfg["quality_reliability_target_step"] * i,
    )
    params.update(override)
    return params


def _job_type_weights(job_config: Optional[dict] = None) -> np.ndarray:
    cfg = _resolved_job_config(job_config)
    indices = _job_type_indices(job_config)
    weights = np.array([cfg["capacity_base"] + cfg["capacity_step"] * (i - 1) for i in indices], dtype=float)
    if np.sum(weights) <= 0:
        return np.ones(len(indices)) / max(1, len(indices))
    return weights / np.sum(weights)


def generate_jobs(
    day: int,
    num_jobs: int,
    next_job_id: int,
    next_component_id: int,
    rng: np.random.Generator,
    max_delay_factor: float = 1.0,
    job_config: Optional[dict] = None,
) -> tuple[list[Job], list[JobComponent], int, int]:
    """
    Generate `num_jobs` jobs for the given day.
    Returns (jobs, skipped_infeasible_components, next_job_id, next_component_id).
    """
    jobs: list[Job] = []
    all_components: list[JobComponent] = []

    allowed_indices = np.array(_job_type_indices(job_config), dtype=int)
    if allowed_indices.size == 0:
        allowed_indices = np.array([1], dtype=int)
    job_type_indices = rng.choice(allowed_indices, size=num_jobs, p=_job_type_weights(job_config))

    for jt_idx in job_type_indices:
        params = job_type_params(int(jt_idx), rng, job_config=job_config)
        delivery_loc = (rng.uniform(), rng.uniform())

        components = []
        feasible = True
        for _ in range(params["num_components"]):
            comp = JobComponent(
                component_id=next_component_id,
                job_id=next_job_id,
                job_type_index=params["job_type_index"],
                deadline_days=params["deadline_days"],
                max_delay=max(1, int(math.ceil(params["deadline_days"] * max_delay_factor))),
                max_workers=params["max_workers"],
                capacity_needed=params["capacity_needed"],
                quality_cost=params["quality_cost"],
                material_cost=params["material_cost"],
                base_labor_rate=params["base_labor_rate"],
                base_transportation_cost=params["base_transportation_cost"],
                late_penalty_per_day=float(params.get("late_penalty_per_day", 0.01)),
                max_daily_manhours_per_worker=params["max_daily_manhours_per_worker"],
                timeline_reliability_target=params["timeline_reliability_target"],
                quality_reliability_target=params["quality_reliability_target"],
                delivery_location=delivery_loc,
            )
            comp.days_remaining = params["deadline_days"]

            if not comp.is_feasible():
                print(
                    f"[ERROR] Day {day}: Job type {jt_idx} component is infeasible "
                    f"(capacity_needed={comp.capacity_needed:.1f}, "
                    f"max_possible={comp.max_workers * comp.max_daily_manhours_per_worker * comp.deadline_days:.1f}). "
                    "Skipping job."
                )
                feasible = False
                break

            components.append(comp)
            next_component_id += 1

        if not feasible:
            continue

        job = Job(
            job_id=next_job_id,
            job_type_index=params["job_type_index"],
            components=components,
            day_created=day,
            late_penalty_per_day=float(params.get("late_penalty_per_day", 0.01)),
        )
        jobs.append(job)
        all_components.extend(components)
        next_job_id += 1

    return jobs, all_components, next_job_id, next_component_id


# ---------------------------------------------------------------------------
# Shop generation
# ---------------------------------------------------------------------------

def generate_shops(
    num_shops: int,
    rng: np.random.Generator,
    capacity_utilization_mean_override: Optional[float] = None,
    capacity_utilization_std_override: Optional[float] = None,
    shop_type_params_override: Optional[dict[str, dict]] = None,
) -> list[Shop]:
    shop_params = dict(SHOP_TYPE_PARAMS)
    if shop_type_params_override:
        for type_name, overrides in shop_type_params_override.items():
            if type_name not in shop_params:
                continue
            merged = dict(shop_params[type_name])
            merged.update(overrides)
            shop_params[type_name] = merged

    type_names = list(shop_params.keys())
    fractions = [shop_params[t]["fraction"] for t in type_names]
    counts = _fractional_counts(num_shops, fractions)

    shops: list[Shop] = []
    shop_id = 0
    for t_name, count in zip(type_names, counts):
        p = shop_params[t_name]
        for _ in range(count):
            qr  = float(np.clip(rng.normal(p["quality_rate_mean"], p["quality_rate_std"]), 0, 1))
            we  = float(np.clip(rng.normal(p["work_efficiency_mean"], p["work_efficiency_std"]), 0, 1))
            wc  = float(max(0.1, rng.normal(p["worker_capacity_mean"], p["worker_capacity_std"])))
            lm  = float(max(0.01, rng.normal(p["labor_cost_multiplier_mean"], p["labor_cost_multiplier_std"])))
            cu_mean = p["capacity_utilization_mean"] if capacity_utilization_mean_override is None else capacity_utilization_mean_override
            cu_std = p["capacity_utilization_std"] if capacity_utilization_std_override is None else capacity_utilization_std_override
            loc = (float(rng.uniform()), float(rng.uniform()))

            shops.append(Shop(
                shop_id=shop_id,
                shop_type=t_name,
                num_workers=p["num_workers"],
                quality_rate=qr,
                work_efficiency=we,
                worker_capacity=wc,
                labor_cost_multiplier=lm,
                capacity_utilization_mean=float(np.clip(cu_mean, 0.0, 1.0)),
                capacity_utilization_std=float(max(0.0, cu_std)),
                location=loc,
            ))
            shop_id += 1
    return shops


def _fractional_counts(total: int, fractions: list) -> list[int]:
    """Distribute `total` into buckets according to `fractions`, rounding carefully."""
    counts = [int(f * total) for f in fractions]
    remainder = total - sum(counts)
    # Give remainder to the largest fraction groups
    order = sorted(range(len(fractions)), key=lambda i: fractions[i] * total - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i]] += 1
    return counts
