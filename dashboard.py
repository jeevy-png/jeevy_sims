from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import streamlit as st

from plotting import (
    plot_comparison,
    plot_job_cost_comparison,
    plot_job_statistics,
    plot_shop_comparison,
    plot_shop_statistics,
    plot_success_rates_vs_targets,
)
from simulation import SimulationRun
from simulation_secondary import SecondarySimulationRun
from stats import aggregate_runs
from models import DEFAULT_JOB_CONFIG, SHOP_TYPE_PARAMS

HUB_LOCATION = (0.5, 0.5)


def _plot_network_map(sim, title: str, output_dir: Path, filename: str) -> Path:
    shop_by_id = {s.shop_id: s for s in sim.shops}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#111315")
    ax.set_facecolor("#171b1f")

    # Plot shops by type.
    type_colors = {
        "Elite": "#30a2ff",
        "Strong": "#48c774",
        "Average": "#ffb347",
        "Risky": "#ff5c5c",
    }
    for shop_type, color in type_colors.items():
        xs = [s.location[0] for s in sim.shops if s.shop_type == shop_type]
        ys = [s.location[1] for s in sim.shops if s.shop_type == shop_type]
        if xs:
            ax.scatter(xs, ys, s=30, c=color, alpha=0.8, label=f"{shop_type} shops")

    # Plot job delivery points and travel paths.
    delivery_x, delivery_y = [], []
    for job in sim.jobs.values():
        for comp in job.components:
            delivery_x.append(comp.delivery_location[0])
            delivery_y.append(comp.delivery_location[1])

            history = [shop_by_id[sid] for sid in comp.shop_assignment_history if sid in shop_by_id]
            if history:
                # Path through network shops (including re-allocations).
                for a, b in zip(history[:-1], history[1:]):
                    ax.plot(
                        [a.location[0], b.location[0]],
                        [a.location[1], b.location[1]],
                        color="#ff8c24",
                        linewidth=1.1,
                        alpha=0.65,
                    )

                # Last hop to delivery.
                last = history[-1]
                ax.plot(
                    [last.location[0], comp.delivery_location[0]],
                    [last.location[1], comp.delivery_location[1]],
                    color="#8be9fd",
                    linewidth=0.9,
                    alpha=0.5,
                    linestyle="--",
                )

                # Quality checks shown as links to hub.
                if comp.quality_checks_performed > 0:
                    unique_shops = list({s.shop_id: s for s in history}.values())
                    checks_per_shop = max(1, int(round(comp.quality_checks_performed / max(1, len(unique_shops)))))
                    for s in unique_shops:
                        ax.plot(
                            [s.location[0], HUB_LOCATION[0]],
                            [s.location[1], HUB_LOCATION[1]],
                            color="#ffd166",
                            linewidth=0.4 + 0.25 * checks_per_shop,
                            alpha=0.25,
                            linestyle=":",
                        )

    if delivery_x:
        ax.scatter(delivery_x, delivery_y, s=24, c="#f7f7f7", alpha=0.65, marker="x", label="Job deliveries")

    # Hub marker.
    ax.scatter([HUB_LOCATION[0]], [HUB_LOCATION[1]], s=120, c="#ffd166", marker="*", label="Quality hub")

    ax.set_title(title, color="#f2f4f5", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("X Location", color="#d7dde2")
    ax.set_ylabel("Y Location", color="#d7dde2")
    ax.grid(color="#313942", alpha=0.3)
    ax.tick_params(colors="#d7dde2")
    for spine in ax.spines.values():
        spine.set_color("#39424c")
    leg = ax.legend(facecolor="#171b1f", edgecolor="#39424c", fontsize=8)
    for text in leg.get_texts():
        text.set_color("#f2f4f5")

    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_days_list(raw: str) -> list[int]:
    if not raw.strip():
        return [0]
    days = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("Custom generation days must be non-negative.")
        days.append(value)
    return sorted(set(days)) if days else [0]


def _normalize_shop_fractions(shop_cfg: dict[str, dict]) -> dict[str, dict]:
    total = sum(max(0.0, float(v.get("fraction", 0.0))) for v in shop_cfg.values())
    if total <= 0:
        n = max(1, len(shop_cfg))
        for name in shop_cfg:
            shop_cfg[name]["fraction"] = 1.0 / n
        return shop_cfg
    for name in shop_cfg:
        shop_cfg[name]["fraction"] = max(0.0, float(shop_cfg[name].get("fraction", 0.0))) / total
    return shop_cfg


def _shop_params_controls() -> dict[str, dict]:
    st.subheader("Shop Parameters")
    shop_cfg: dict[str, dict] = {}
    for shop_type, defaults in SHOP_TYPE_PARAMS.items():
        with st.expander(f"{shop_type} shop type", expanded=False):
            shop_cfg[shop_type] = {
                "num_workers": int(st.number_input(f"{shop_type} num_workers", min_value=1, max_value=100, value=int(defaults["num_workers"]), step=1)),
                "quality_rate_mean": float(st.slider(f"{shop_type} quality_rate_mean", min_value=0.0, max_value=1.0, value=float(defaults["quality_rate_mean"]), step=0.001)),
                "quality_rate_std": float(st.slider(f"{shop_type} quality_rate_std", min_value=0.0, max_value=0.5, value=float(defaults["quality_rate_std"]), step=0.001)),
                "work_efficiency_mean": float(st.slider(f"{shop_type} work_efficiency_mean", min_value=0.0, max_value=1.0, value=float(defaults["work_efficiency_mean"]), step=0.001)),
                "work_efficiency_std": float(st.slider(f"{shop_type} work_efficiency_std", min_value=0.0, max_value=0.5, value=float(defaults["work_efficiency_std"]), step=0.001)),
                "worker_capacity_mean": float(st.number_input(f"{shop_type} worker_capacity_mean", min_value=0.1, max_value=100.0, value=float(defaults["worker_capacity_mean"]), step=0.1)),
                "worker_capacity_std": float(st.number_input(f"{shop_type} worker_capacity_std", min_value=0.0, max_value=100.0, value=float(defaults["worker_capacity_std"]), step=0.1)),
                "labor_cost_multiplier_mean": float(st.number_input(f"{shop_type} labor_cost_multiplier_mean", min_value=0.01, max_value=10.0, value=float(defaults["labor_cost_multiplier_mean"]), step=0.01)),
                "labor_cost_multiplier_std": float(st.number_input(f"{shop_type} labor_cost_multiplier_std", min_value=0.0, max_value=10.0, value=float(defaults["labor_cost_multiplier_std"]), step=0.01)),
                "capacity_utilization_mean": float(st.slider(f"{shop_type} capacity_utilization_mean", min_value=0.0, max_value=1.0, value=float(defaults["capacity_utilization_mean"]), step=0.01)),
                "capacity_utilization_std": float(st.slider(f"{shop_type} capacity_utilization_std", min_value=0.0, max_value=1.0, value=float(defaults["capacity_utilization_std"]), step=0.01)),
                "fraction": float(st.slider(f"{shop_type} fraction", min_value=0.0, max_value=1.0, value=float(defaults["fraction"]), step=0.01)),
            }
    return _normalize_shop_fractions(shop_cfg)


def _job_config_controls(max_days: int) -> tuple[dict, list[int]]:
    st.subheader("Job Parameters")
    cfg = dict(DEFAULT_JOB_CONFIG)

    selection_mode = st.selectbox("Job type selection mode", ["range", "explicit"], index=0)
    if selection_mode == "range":
        min_idx = int(st.number_input("job_type_min_index", min_value=1, max_value=50, value=int(cfg["job_type_min_index"]), step=1))
        max_idx = int(st.number_input("job_type_max_index", min_value=min_idx, max_value=50, value=int(cfg["job_type_max_index"]), step=1))
        cfg["job_type_min_index"] = min_idx
        cfg["job_type_max_index"] = max_idx
        selected_types = list(range(min_idx, max_idx + 1))
        cfg["job_type_indices"] = []
    else:
        selected_types = st.multiselect("job_type_indices", options=list(range(1, 51)), default=list(range(1, 11)))
        if not selected_types:
            selected_types = [1]
        cfg["job_type_indices"] = selected_types

    cfg["num_components_divisor"] = int(st.number_input("num_components_divisor", min_value=1, max_value=50, value=int(cfg["num_components_divisor"]), step=1))
    cfg["deadline_base"] = float(st.number_input("deadline_base", min_value=0.0, max_value=1000.0, value=float(cfg["deadline_base"]), step=1.0))
    cfg["deadline_step"] = float(st.number_input("deadline_step", min_value=0.0, max_value=1000.0, value=float(cfg["deadline_step"]), step=1.0))
    cfg["capacity_base"] = float(st.number_input("capacity_base", min_value=0.0, max_value=1_000_000.0, value=float(cfg["capacity_base"]), step=1.0))
    cfg["capacity_step"] = float(st.number_input("capacity_step", min_value=0.0, max_value=1_000_000.0, value=float(cfg["capacity_step"]), step=1.0))
    cfg["max_workers"] = int(st.number_input("max_workers", min_value=1, max_value=100, value=int(cfg["max_workers"]), step=1))
    cfg["quality_cost_base"] = float(st.number_input("quality_cost_base", min_value=0.0, max_value=1_000_000.0, value=float(cfg["quality_cost_base"]), step=1.0))
    cfg["quality_cost_step"] = float(st.number_input("quality_cost_step", min_value=0.0, max_value=1_000_000.0, value=float(cfg["quality_cost_step"]), step=1.0))
    cfg["material_cost_base"] = float(st.number_input("material_cost_base", min_value=0.0, max_value=10_000_000.0, value=float(cfg["material_cost_base"]), step=1.0))
    cfg["material_cost_step"] = float(st.number_input("material_cost_step", min_value=0.0, max_value=10_000_000.0, value=float(cfg["material_cost_step"]), step=1.0))
    cfg["max_daily_manhours_per_worker"] = float(st.number_input("max_daily_manhours_per_worker", min_value=0.1, max_value=100.0, value=float(cfg["max_daily_manhours_per_worker"]), step=0.1))
    cfg["base_labor_rate_base"] = float(st.number_input("base_labor_rate_base", min_value=0.0, max_value=10_000.0, value=float(cfg["base_labor_rate_base"]), step=1.0))
    cfg["base_labor_rate_step"] = float(st.number_input("base_labor_rate_step", min_value=0.0, max_value=10_000.0, value=float(cfg["base_labor_rate_step"]), step=1.0))
    cfg["base_transportation_cost_base"] = float(st.number_input("base_transportation_cost_base", min_value=0.0, max_value=10_000_000.0, value=float(cfg["base_transportation_cost_base"]), step=1.0))
    cfg["base_transportation_cost_step"] = float(st.number_input("base_transportation_cost_step", min_value=0.0, max_value=10_000_000.0, value=float(cfg["base_transportation_cost_step"]), step=1.0))
    cfg["timeline_reliability_target_base"] = float(st.slider("timeline_reliability_target_base", min_value=0.0, max_value=1.0, value=float(cfg["timeline_reliability_target_base"]), step=0.001))
    cfg["timeline_reliability_target_step"] = float(st.slider("timeline_reliability_target_step", min_value=0.0, max_value=1.0, value=float(cfg["timeline_reliability_target_step"]), step=0.001))
    cfg["quality_reliability_target_base"] = float(st.slider("quality_reliability_target_base", min_value=0.0, max_value=1.0, value=float(cfg["quality_reliability_target_base"]), step=0.001))
    cfg["quality_reliability_target_step"] = float(st.slider("quality_reliability_target_step", min_value=0.0, max_value=1.0, value=float(cfg["quality_reliability_target_step"]), step=0.001))

    st.subheader("Per-Type Hardcoded Job Overrides")
    override_types = st.multiselect("Override specific job types", options=selected_types, default=[])
    overrides: dict[str, dict] = {}
    for jt in override_types:
        with st.expander(f"Type {jt} overrides", expanded=False):
            overrides[str(jt)] = {
                "num_components": int(st.number_input(f"T{jt} num_components", min_value=1, max_value=100, value=max(1, int((jt + cfg["num_components_divisor"] - 1) // cfg["num_components_divisor"])), step=1)),
                "deadline_days": int(st.number_input(f"T{jt} deadline_days", min_value=1, max_value=max(1, max_days * 2), value=max(1, int((cfg["deadline_base"] + jt * cfg["deadline_step"]) / max(1, int((jt + cfg["num_components_divisor"] - 1) // cfg["num_components_divisor"])))), step=1)),
                "capacity_needed": float(st.number_input(f"T{jt} capacity_needed", min_value=0.0, max_value=10_000_000.0, value=float(cfg["capacity_base"] + cfg["capacity_step"] * (jt - 1)), step=1.0)),
                "max_workers": int(st.number_input(f"T{jt} max_workers", min_value=1, max_value=100, value=int(cfg["max_workers"]), step=1)),
                "quality_cost": float(st.number_input(f"T{jt} quality_cost", min_value=0.0, max_value=1_000_000.0, value=float(cfg["quality_cost_base"] + cfg["quality_cost_step"] * (jt - 1)), step=1.0)),
                "material_cost": float(st.number_input(f"T{jt} material_cost", min_value=0.0, max_value=10_000_000.0, value=float(cfg["material_cost_base"] + cfg["material_cost_step"] * (jt - 1)), step=1.0)),
                "max_daily_manhours_per_worker": float(st.number_input(f"T{jt} max_daily_manhours_per_worker", min_value=0.1, max_value=100.0, value=float(cfg["max_daily_manhours_per_worker"]), step=0.1)),
                "base_labor_rate": float(st.number_input(f"T{jt} base_labor_rate", min_value=0.0, max_value=10_000.0, value=float(cfg["base_labor_rate_base"] + cfg["base_labor_rate_step"] * (jt - 1)), step=1.0)),
                "base_transportation_cost": float(st.number_input(f"T{jt} base_transportation_cost", min_value=0.0, max_value=10_000_000.0, value=float(cfg["base_transportation_cost_base"] + cfg["base_transportation_cost_step"] * (jt - 1)), step=1.0)),
                "timeline_reliability_target": float(st.slider(f"T{jt} timeline_reliability_target", min_value=0.0, max_value=1.0, value=float(cfg["timeline_reliability_target_base"] + cfg["timeline_reliability_target_step"] * jt), step=0.001)),
                "quality_reliability_target": float(st.slider(f"T{jt} quality_reliability_target", min_value=0.0, max_value=1.0, value=float(cfg["quality_reliability_target_base"] + cfg["quality_reliability_target_step"] * jt), step=0.001)),
            }
    cfg["job_type_overrides"] = overrides

    return cfg, selected_types


def _run_primary(
    runs: int,
    shops: int,
    days: int,
    jobs_per_day: int,
    seed: int,
    quality_checks: int,
    failure_penalty_rate: float,
    max_delay_factor: float,
    capacity_utilization_mean: Optional[float],
    capacity_utilization_std: Optional[float],
    job_generation_mode: str,
    job_generation_probability: float,
    job_generation_days: list[int],
    shop_type_params_override: Optional[dict[str, dict]],
    job_config: Optional[dict],
    output_dir: str,
) -> tuple[dict, SimulationRun]:
    all_stats = []
    exemplar_sim: Optional[SimulationRun] = None
    for r in range(runs):
        sim = SimulationRun(
            num_shops=shops,
            num_days=days,
            jobs_per_day=jobs_per_day,
            rng_seed=seed + r,
            num_quality_checks=quality_checks,
            failure_penalty_rate=failure_penalty_rate,
            max_delay_factor=max_delay_factor,
            capacity_utilization_mean=capacity_utilization_mean,
            capacity_utilization_std=capacity_utilization_std,
            job_generation_mode=job_generation_mode,
            job_generation_probability=job_generation_probability,
            job_generation_days=job_generation_days,
            shop_type_params_override=shop_type_params_override,
            job_config=job_config,
        )
        sim.run()
        all_stats.append(sim.get_statistics())
        if exemplar_sim is None:
            exemplar_sim = sim

    agg = aggregate_runs(all_stats)
    plot_shop_statistics(agg, output_dir=output_dir, label="Primary")
    plot_job_statistics(agg, output_dir=output_dir, label="Primary")
    plot_success_rates_vs_targets(agg, output_dir=output_dir, label="Primary")
    return agg, exemplar_sim


def _run_secondary(
    runs: int,
    shops: int,
    days: int,
    jobs_per_day: int,
    seed: int,
    failure_penalty_rate: float,
    max_delay_factor: float,
    capacity_utilization_mean: Optional[float],
    capacity_utilization_std: Optional[float],
    job_generation_mode: str,
    job_generation_probability: float,
    job_generation_days: list[int],
    shop_type_params_override: Optional[dict[str, dict]],
    job_config: Optional[dict],
    output_dir: str,
    methods: tuple[str, ...] = ("random_cheapest", "quality_top"),
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for method in methods:
        all_stats = []
        exemplar_sim: Optional[SecondarySimulationRun] = None
        for r in range(runs):
            method_seed = seed + 1000 + r + (500 if method == "quality_top" else 0)
            sim = SecondarySimulationRun(
                num_shops=shops,
                num_days=days,
                jobs_per_day=jobs_per_day,
                allocation_method=method,
                rng_seed=method_seed,
                failure_penalty_rate=failure_penalty_rate,
                max_delay_factor=max_delay_factor,
                capacity_utilization_mean=capacity_utilization_mean,
                capacity_utilization_std=capacity_utilization_std,
                job_generation_mode=job_generation_mode,
                job_generation_probability=job_generation_probability,
                job_generation_days=job_generation_days,
                shop_type_params_override=shop_type_params_override,
                job_config=job_config,
            )
            sim.run()
            all_stats.append(sim.get_statistics())
            if exemplar_sim is None:
                exemplar_sim = sim

        agg = aggregate_runs(all_stats)
        results[method] = {"agg": agg, "sim": exemplar_sim}
        label = "BaseCase" if method == "random_cheapest" else "PartialNetwork"
        plot_shop_statistics(agg, output_dir=output_dir, label=label)
        plot_job_statistics(agg, output_dir=output_dir, label=label)
        plot_success_rates_vs_targets(agg, output_dir=output_dir, label=label)

    return results


def _render_images(output_dir: Path):
    images = sorted(output_dir.glob("*.png"))
    if not images:
        st.warning("No plots were generated.")
        return

    for image_path in images:
        st.image(str(image_path), caption=image_path.name, use_container_width=True)


def main():
    st.set_page_config(page_title="Jeevy Simulation Dashboard", layout="wide")
    st.title("Jeevy Simulation Dashboard")
    st.write("Configure parameters on the left, then run simulations and view plots on the right.")

    with st.sidebar:
        st.header("Run Controls")
        case_mode = st.selectbox(
            "Case",
            ["base case", "partial network", "full network", "all cases"],
            index=0,
        )
        runs = st.number_input("Runs", min_value=1, max_value=200, value=3, step=1)
        shops = st.number_input("Shops", min_value=1, max_value=2000, value=100, step=1)
        days = st.number_input("Days", min_value=1, max_value=3650, value=365, step=1)
        jobs_per_day = st.number_input("Jobs Generated Per Event", min_value=1, max_value=1000, value=5, step=1)
        seed = st.number_input("Base Seed", min_value=0, max_value=2_000_000_000, value=42, step=1)

        st.subheader("Quality / Delay")
        quality_checks = st.number_input("Quality Checks (Primary)", min_value=1, max_value=20, value=3, step=1)
        max_delay_factor = st.number_input("Max Delay Factor", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        failure_penalty_rate = st.number_input("Failure Penalty Rate", min_value=0.0, max_value=5.0, value=0.20, step=0.01)

        st.subheader("Capacity Utilization")
        use_util_overrides = st.checkbox("Override Utilization Mean/Std for all shop types", value=False)
        capacity_utilization_mean = None
        capacity_utilization_std = None
        if use_util_overrides:
            capacity_utilization_mean = st.number_input("Capacity Utilization Mean", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
            capacity_utilization_std = st.number_input("Capacity Utilization Std Dev", min_value=0.0, max_value=1.0, value=0.02, step=0.01)

        st.subheader("Job Generation")
        job_generation_mode = st.selectbox(
            "Job Generation Mode",
            ["start_only", "daily", "probabilistic", "custom_days"],
            index=0,
        )
        job_generation_probability = 0.0
        custom_days_selection = [0]
        if job_generation_mode == "probabilistic":
            job_generation_probability = st.number_input(
                "Generation Probability Per Day",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
            )
        if job_generation_mode == "custom_days":
            day_options = list(range(int(days)))
            default_days = [0] if int(days) > 0 else []
            custom_days_selection = st.multiselect("Custom generation days", options=day_options, default=default_days)
            if not custom_days_selection:
                custom_days_selection = [0]

        output_base = st.text_input("Output Base Directory", value="dashboard_runs")

        shop_type_params_override = _shop_params_controls()
        job_config_override, selected_job_types = _job_config_controls(max_days=int(days))

        run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("Current Configuration")
        st.write(
            {
                "case": case_mode,
                "runs": int(runs),
                "shops": int(shops),
                "days": int(days),
                "jobs_per_generation_event": int(jobs_per_day),
                "seed": int(seed),
                "quality_checks": int(quality_checks),
                "max_delay_factor": float(max_delay_factor),
                "failure_penalty_rate": float(failure_penalty_rate),
                "capacity_utilization_mean_override": capacity_utilization_mean,
                "capacity_utilization_std_override": capacity_utilization_std,
                "job_generation_mode": job_generation_mode,
                "job_generation_probability": float(job_generation_probability),
                "generated_job_types": selected_job_types,
                "shop_params_customized": True,
                "job_params_customized": True,
            }
        )

    with right_col:
        st.subheader("Plots")

        if run_clicked:
            try:
                if job_generation_mode == "custom_days":
                    job_generation_days = sorted(set(int(d) for d in custom_days_selection))
                else:
                    job_generation_days = [0]

                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(output_base) / f"run_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)

                with st.spinner("Running simulation..."):
                    agg_primary = None
                    agg_secondary = None
                    map_paths: list[Path] = []

                    if case_mode in ("full network", "all cases"):
                        agg_primary, primary_sim = _run_primary(
                            runs=int(runs),
                            shops=int(shops),
                            days=int(days),
                            jobs_per_day=int(jobs_per_day),
                            seed=int(seed),
                            quality_checks=int(quality_checks),
                            failure_penalty_rate=float(failure_penalty_rate),
                            max_delay_factor=float(max_delay_factor),
                            capacity_utilization_mean=capacity_utilization_mean,
                            capacity_utilization_std=capacity_utilization_std,
                            job_generation_mode=job_generation_mode,
                            job_generation_probability=float(job_generation_probability),
                            job_generation_days=job_generation_days,
                            shop_type_params_override=shop_type_params_override,
                            job_config=job_config_override,
                            output_dir=str(output_dir),
                        )
                        # Re-label primary figures to match dashboard terminology.
                        plot_shop_statistics(agg_primary, output_dir=str(output_dir), label="FullNetwork")
                        plot_job_statistics(agg_primary, output_dir=str(output_dir), label="FullNetwork")
                        plot_success_rates_vs_targets(agg_primary, output_dir=str(output_dir), label="FullNetwork")
                        if primary_sim is not None:
                            map_paths.append(
                                _plot_network_map(
                                    primary_sim,
                                    "Full Network Flow Map",
                                    output_dir,
                                    "network_map_full_network.png",
                                )
                            )

                    if case_mode in ("base case", "partial network", "all cases"):
                        if case_mode == "base case":
                            methods = ("random_cheapest",)
                        elif case_mode == "partial network":
                            methods = ("quality_top",)
                        else:
                            methods = ("random_cheapest", "quality_top")
                        agg_secondary = _run_secondary(
                            runs=int(runs),
                            shops=int(shops),
                            days=int(days),
                            jobs_per_day=int(jobs_per_day),
                            seed=int(seed),
                            failure_penalty_rate=float(failure_penalty_rate),
                            max_delay_factor=float(max_delay_factor),
                            capacity_utilization_mean=capacity_utilization_mean,
                            capacity_utilization_std=capacity_utilization_std,
                            job_generation_mode=job_generation_mode,
                            job_generation_probability=float(job_generation_probability),
                            job_generation_days=job_generation_days,
                            shop_type_params_override=shop_type_params_override,
                            job_config=job_config_override,
                            output_dir=str(output_dir),
                            methods=methods,
                        )

                        if "random_cheapest" in agg_secondary:
                            sec_rand_sim = agg_secondary["random_cheapest"].get("sim")
                            if sec_rand_sim is not None:
                                map_paths.append(
                                    _plot_network_map(
                                        sec_rand_sim,
                                        "Base Case Flow Map",
                                        output_dir,
                                        "network_map_base_case.png",
                                    )
                                )

                        if "quality_top" in agg_secondary:
                            sec_qual_sim = agg_secondary["quality_top"].get("sim")
                            if sec_qual_sim is not None:
                                map_paths.append(
                                    _plot_network_map(
                                        sec_qual_sim,
                                        "Partial Network Flow Map",
                                        output_dir,
                                        "network_map_partial_network.png",
                                    )
                                )

                    if (
                        case_mode == "all cases"
                        and agg_primary is not None
                        and agg_secondary is not None
                        and "random_cheapest" in agg_secondary
                        and "quality_top" in agg_secondary
                    ):
                        plot_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                        )
                        plot_job_cost_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                        )
                        plot_shop_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                        )

                st.session_state["last_output_dir"] = str(output_dir)
                st.session_state["last_map_paths"] = [str(p) for p in map_paths]
                st.success(f"Run complete. Plots saved to: {output_dir}")
            except Exception as exc:
                st.error(f"Run failed: {exc}")

        map_paths = st.session_state.get("last_map_paths", [])
        if map_paths:
            st.subheader("Network Travel Maps")
            for p in map_paths:
                st.image(p, caption=Path(p).name, use_container_width=True)

        last_output_dir = st.session_state.get("last_output_dir")
        if last_output_dir:
            _render_images(Path(last_output_dir))
        else:
            st.info("Run a simulation to generate and display plots.")


if __name__ == "__main__":
    main()
