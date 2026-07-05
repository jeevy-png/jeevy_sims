from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from plotting import (
    plot_primary_quality_check_sweep,
    plot_completed_only_quality_rate,
    plot_completed_only_quality_comparison,
    plot_comparison,
    plot_job_cost_comparison,
    plot_job_statistics,
    plot_mode_operational_comparison,
    plot_shop_comparison,
    plot_shop_statistics,
    plot_success_rates_vs_targets,
)
from simulation import SimulationRun
from simulation_secondary import SecondarySimulationRun
from stats import aggregate_runs
from models import DEFAULT_JOB_CONFIG, SHOP_TYPE_PARAMS

HUB_LOCATION = (0.5, 0.5)


def _plot_network_map(
    sim,
    title: str,
    output_dir: Path,
    filename: str,
    text_overrides: Optional[dict[str, str]] = None,
) -> Path:
    shop_by_id = {s.shop_id: s for s in sim.shops}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F6FAFE")

    # Plot shops by type.
    type_colors = {
        "Elite": "#0B4F8C",
        "Strong": "#1F77B4",
        "Average": "#4FA3E3",
        "Risky": "#9BCBF3",
    }
    for shop_type, color in type_colors.items():
        xs = [s.location[0] for s in sim.shops if s.shop_type == shop_type]
        ys = [s.location[1] for s in sim.shops if s.shop_type == shop_type]
        if xs:
            ax.scatter(xs, ys, s=34, c=color, alpha=0.9, edgecolors="#FFFFFF", linewidths=0.5, label=f"{shop_type} shops")

    # Plot job delivery points and travel paths.
    delivery_x, delivery_y = [], []
    for job in sim.jobs.values():
        for comp in job.components:
            delivery_x.append(comp.delivery_location[0])
            delivery_y.append(comp.delivery_location[1])

            history = [shop_by_id[sid] for sid in comp.shop_assignment_history if sid in shop_by_id]
            if history:
                # Path through network shops (including re-allocations) with arrows.
                for a, b in zip(history[:-1], history[1:]):
                    ax.annotate(
                        "",
                        xy=(b.location[0], b.location[1]),
                        xytext=(a.location[0], a.location[1]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="#1F77B4",
                            lw=1.8,
                            alpha=0.7,
                            mutation_scale=15,
                            shrinkA=2,
                            shrinkB=2,
                        ),
                    )

                # Last hop to delivery with arrow.
                last = history[-1]
                ax.annotate(
                    "",
                    xy=(comp.delivery_location[0], comp.delivery_location[1]),
                    xytext=(last.location[0], last.location[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#0B4F8C",
                        lw=1.4,
                        alpha=0.6,
                        mutation_scale=13,
                        linestyle="dashed",
                        shrinkA=2,
                        shrinkB=2,
                    ),
                )

                # Quality checks shown as links to hub.
                if comp.quality_checks_performed > 0:
                    unique_shops = list({s.shop_id: s for s in history}.values())
                    checks_per_shop = max(1, int(round(comp.quality_checks_performed / max(1, len(unique_shops)))))
                    for s in unique_shops:
                        ax.plot(
                            [s.location[0], HUB_LOCATION[0]],
                            [s.location[1], HUB_LOCATION[1]],
                            color="#6AAFE6",
                            linewidth=0.35 + 0.18 * checks_per_shop,
                            alpha=0.2,
                            linestyle=":",
                        )

    if delivery_x:
        ax.scatter(delivery_x, delivery_y, s=26, c="#0B2A44", alpha=0.8, marker="x", label="Job deliveries")

    # Hub marker.
    ax.scatter([HUB_LOCATION[0]], [HUB_LOCATION[1]], s=135, c="#0B4F8C", marker="*", label="Quality hub")

    ax.set_title(text_overrides.get(title, title) if text_overrides else title, color="#0B2A44", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_label = "X Location"
    y_label = "Y Location"
    if text_overrides:
        x_label = text_overrides.get(x_label, x_label)
        y_label = text_overrides.get(y_label, y_label)
    ax.set_xlabel(x_label, color="#0B2A44", fontsize=13)
    ax.set_ylabel(y_label, color="#0B2A44", fontsize=13)
    ax.tick_params(colors="#16324A", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D4DDE6")
    ax.spines["bottom"].set_color("#D4DDE6")
    ax.grid(False)
    leg = ax.legend(facecolor="#FFFFFF", edgecolor="#FFFFFF", fontsize=10, framealpha=0.95)
    for text in leg.get_texts():
        text.set_color("#0B2A44")

    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _settings_dir() -> Path:
    p = Path("saved_settings")
    p.mkdir(exist_ok=True)
    return p


def _collect_settings(case_mode, runs, shops, days, jobs_per_day, seed,
                       quality_checks, max_delay_factor, failure_penalty_rate, backup_shop_depth,
                       allocation_planner_mode,
                       enforce_first_layer_timeline_filter,
                       enable_quality_check_sweep, sweep_quality_checks_min, sweep_quality_checks_max,
                       use_util_overrides, capacity_utilization_mean, capacity_utilization_std,
                       job_generation_mode, job_generation_probability, job_generation_days,
                       output_base, shop_type_params_override, job_config_override,
                       plot_text_overrides) -> dict:
    return {
        "case_mode": case_mode,
        "runs": int(runs),
        "shops": int(shops),
        "days": int(days),
        "jobs_per_day": int(jobs_per_day),
        "seed": int(seed),
        "quality_checks": int(quality_checks),
        "max_delay_factor": float(max_delay_factor),
        "failure_penalty_rate": float(failure_penalty_rate),
        "backup_shop_depth": int(backup_shop_depth),
        "allocation_planner_mode": allocation_planner_mode,
        "enforce_first_layer_timeline_filter": bool(enforce_first_layer_timeline_filter),
        "enable_quality_check_sweep": bool(enable_quality_check_sweep),
        "sweep_quality_checks_min": int(sweep_quality_checks_min),
        "sweep_quality_checks_max": int(sweep_quality_checks_max),
        "use_util_overrides": bool(use_util_overrides),
        "capacity_utilization_mean": capacity_utilization_mean,
        "capacity_utilization_std": capacity_utilization_std,
        "job_generation_mode": job_generation_mode,
        "job_generation_probability": float(job_generation_probability),
        "job_generation_days": [int(d) for d in (job_generation_days or [0])],
        "output_base": output_base,
        "shop_type_params_override": shop_type_params_override,
        "job_config_override": job_config_override,
        "plot_text_overrides": plot_text_overrides,
    }


def _apply_loaded_settings_to_session_state(loaded: dict):
    st.session_state["_loaded_settings"] = loaded

    key_map = {
        "case_mode": "case_mode",
        "runs": "runs",
        "shops": "shops",
        "days": "days",
        "jobs_per_day": "jobs_per_day",
        "seed": "seed",
        "quality_checks": "quality_checks",
        "max_delay_factor": "max_delay_factor",
        "failure_penalty_rate": "failure_penalty_rate",
        "backup_shop_depth": "backup_shop_depth",
        "allocation_planner_mode": "allocation_planner_mode",
        "enforce_first_layer_timeline_filter": "enforce_first_layer_timeline_filter",
        "enable_quality_check_sweep": "enable_quality_check_sweep",
        "sweep_quality_checks_min": "sweep_quality_checks_min",
        "sweep_quality_checks_max": "sweep_quality_checks_max",
        "use_util_overrides": "use_util_overrides",
        "capacity_utilization_mean": "capacity_utilization_mean",
        "capacity_utilization_std": "capacity_utilization_std",
        "job_generation_mode": "job_generation_mode",
        "job_generation_probability": "job_generation_probability",
        "output_base": "output_base",
    }
    for src, dst in key_map.items():
        if src in loaded:
            st.session_state[dst] = loaded[src]

    if "job_generation_days" in loaded:
        st.session_state["custom_generation_days"] = [int(d) for d in loaded.get("job_generation_days", [0])]

    if "plot_text_overrides" in loaded:
        st.session_state["plot_text_overrides_raw"] = json.dumps(loaded.get("plot_text_overrides", {}), indent=2)

    shop_overrides = loaded.get("shop_type_params_override", {}) or {}
    for shop_type, params in shop_overrides.items():
        if not isinstance(params, dict):
            continue
        for param_name, param_value in params.items():
            st.session_state[f"{shop_type}_{param_name}"] = param_value

    job_cfg = loaded.get("job_config_override", {}) or {}
    if isinstance(job_cfg, dict):
        scalar_job_keys = [
            "job_type_min_index", "job_type_max_index", "num_components_divisor",
            "deadline_base", "deadline_step", "capacity_base", "capacity_step",
            "max_workers", "quality_cost_base", "quality_cost_step",
            "material_cost_base", "material_cost_step", "max_daily_manhours_per_worker",
            "base_labor_rate_base", "base_labor_rate_step",
            "base_transportation_cost_base", "base_transportation_cost_step",
            "late_penalty_per_day_base", "late_penalty_per_day_step",
            "timeline_reliability_target_base", "timeline_reliability_target_step",
            "quality_reliability_target_base", "quality_reliability_target_step",
        ]
        for key in scalar_job_keys:
            if key in job_cfg:
                st.session_state[key] = job_cfg[key]

        indices = job_cfg.get("job_type_indices", [])
        if indices:
            st.session_state["job_type_selection_mode"] = "explicit"
            st.session_state["job_type_indices"] = [int(x) for x in indices]
        else:
            st.session_state["job_type_selection_mode"] = "range"

        overrides = job_cfg.get("job_type_overrides", {}) or {}
        if isinstance(overrides, dict):
            override_types = sorted(int(k) for k in overrides.keys())
            st.session_state["job_override_types"] = override_types
            for jt_str, params in overrides.items():
                if not isinstance(params, dict):
                    continue
                jt = int(jt_str)
                for param_name, param_value in params.items():
                    st.session_state[f"job_{jt}_{param_name}"] = param_value


def _parse_plot_text_overrides(raw_text: str) -> dict[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Plot text overrides must be a JSON object mapping existing text to replacement text.")
    return {str(k): str(v) for k, v in parsed.items()}


def _d(key: str, default):
    """Read a top-level value from loaded settings, with a default fallback."""
    return st.session_state.get("_loaded_settings", {}).get(key, default)


def _ds(shop_type: str, param: str, default):
    """Read a shop-type param from loaded settings."""
    return (
        st.session_state.get("_loaded_settings", {})
        .get("shop_type_params_override", {})
        .get(shop_type, {})
        .get(param, default)
    )


def _dj(param: str, default):
    """Read a job config param from loaded settings."""
    jcfg = st.session_state.get("_loaded_settings", {}).get("job_config_override") or {}
    return jcfg.get(param, default)


def _shop_assumptions_df(shop_cfg: dict) -> pd.DataFrame:
    rows = []
    for shop_type, params in shop_cfg.items():
        row = {"shop_type": shop_type}
        row.update(params)
        rows.append(row)
    return pd.DataFrame(rows).set_index("shop_type")


def _job_assumptions_df(job_cfg: dict, selected_types: list) -> pd.DataFrame:
    import math
    import numpy as np
    from models import job_type_params as _jtp, SHOP_TYPE_PARAMS
    rng = np.random.default_rng(0)
    # Best-case efficiency is the highest work_efficiency_mean across shop types.
    best_efficiency = max(v["work_efficiency_mean"] for v in SHOP_TYPE_PARAMS.values())
    rows = []
    for i in selected_types:
        p = _jtp(i, rng, job_config=job_cfg)
        mw = p["max_workers"]
        mh = p["max_daily_manhours_per_worker"]
        cap = p["capacity_needed"]
        daily_raw = mw * mh
        # Ideal: perfect efficiency, no delay
        min_days_ideal = math.ceil(cap / max(daily_raw, 0.01))
        # Practical: best shop efficiency + 1-day allocation delay
        min_days_practical = math.ceil(cap / max(daily_raw * best_efficiency, 0.01)) + 1
        rows.append({
            "type": f"T{i}",
            "components": p["num_components"],
            "deadline_days": p["deadline_days"],
            "min_days_ideal": min_days_ideal,
            "min_days_practical": min_days_practical,
            "capacity_needed": round(p["capacity_needed"], 1),
            "max_workers": p["max_workers"],
            "quality_cost": p["quality_cost"],
            "material_cost": p["material_cost"],
            "late_penalty_per_day": round(p.get("late_penalty_per_day", 0.01), 4),
            "base_labor_rate": p["base_labor_rate"],
            "transport_base": p["base_transportation_cost"],
            "timeline_target": round(p["timeline_reliability_target"], 3),
            "quality_target": round(p["quality_reliability_target"], 3),
        })
    return pd.DataFrame(rows).set_index("type")


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
                "num_workers": int(st.number_input(f"{shop_type} num_workers", min_value=1, max_value=100, value=int(_ds(shop_type, "num_workers", defaults["num_workers"])), step=1, key=f"{shop_type}_num_workers")),
                "quality_rate_mean": float(st.slider(f"{shop_type} quality_rate_mean", min_value=0.0, max_value=1.0, value=float(_ds(shop_type, "quality_rate_mean", defaults["quality_rate_mean"])), step=0.001, key=f"{shop_type}_quality_rate_mean")),
                "quality_rate_std": float(st.slider(f"{shop_type} quality_rate_std", min_value=0.0, max_value=0.5, value=float(_ds(shop_type, "quality_rate_std", defaults["quality_rate_std"])), step=0.001, key=f"{shop_type}_quality_rate_std")),
                "work_efficiency_mean": float(st.slider(f"{shop_type} work_efficiency_mean", min_value=0.0, max_value=1.0, value=float(_ds(shop_type, "work_efficiency_mean", defaults["work_efficiency_mean"])), step=0.001, key=f"{shop_type}_work_efficiency_mean")),
                "work_efficiency_std": float(st.slider(f"{shop_type} work_efficiency_std", min_value=0.0, max_value=0.5, value=float(_ds(shop_type, "work_efficiency_std", defaults["work_efficiency_std"])), step=0.001, key=f"{shop_type}_work_efficiency_std")),
                "worker_capacity_mean": float(st.number_input(f"{shop_type} worker_capacity_mean", min_value=0.1, max_value=100.0, value=float(_ds(shop_type, "worker_capacity_mean", defaults["worker_capacity_mean"])), step=0.1, key=f"{shop_type}_worker_capacity_mean")),
                "worker_capacity_std": float(st.number_input(f"{shop_type} worker_capacity_std", min_value=0.0, max_value=100.0, value=float(_ds(shop_type, "worker_capacity_std", defaults["worker_capacity_std"])), step=0.1, key=f"{shop_type}_worker_capacity_std")),
                "labor_cost_multiplier_mean": float(st.number_input(f"{shop_type} labor_cost_multiplier_mean", min_value=0.01, max_value=10.0, value=float(_ds(shop_type, "labor_cost_multiplier_mean", defaults["labor_cost_multiplier_mean"])), step=0.01, key=f"{shop_type}_labor_cost_multiplier_mean")),
                "labor_cost_multiplier_std": float(st.number_input(f"{shop_type} labor_cost_multiplier_std", min_value=0.0, max_value=10.0, value=float(_ds(shop_type, "labor_cost_multiplier_std", defaults["labor_cost_multiplier_std"])), step=0.01, key=f"{shop_type}_labor_cost_multiplier_std")),
                "capacity_utilization_mean": float(st.slider(f"{shop_type} capacity_utilization_mean", min_value=0.0, max_value=1.0, value=float(_ds(shop_type, "capacity_utilization_mean", defaults["capacity_utilization_mean"])), step=0.01, key=f"{shop_type}_capacity_utilization_mean")),
                "capacity_utilization_std": float(st.slider(f"{shop_type} capacity_utilization_std", min_value=0.0, max_value=1.0, value=float(_ds(shop_type, "capacity_utilization_std", defaults["capacity_utilization_std"])), step=0.01, key=f"{shop_type}_capacity_utilization_std")),
                "fraction": float(st.slider(f"{shop_type} fraction", min_value=0.0, max_value=1.0, value=float(_ds(shop_type, "fraction", defaults["fraction"])), step=0.01, key=f"{shop_type}_fraction")),
            }
    return _normalize_shop_fractions(shop_cfg)


def _job_config_controls(max_days: int) -> tuple[dict, list[int]]:
    st.subheader("Job Parameters")
    cfg = dict(DEFAULT_JOB_CONFIG)

    selection_mode = st.selectbox("Job type selection mode", ["range", "explicit"], index=0, key="job_type_selection_mode")
    if selection_mode == "range":
        min_idx = int(st.number_input("job_type_min_index", min_value=1, max_value=50, value=int(_dj("job_type_min_index", cfg["job_type_min_index"])), step=1, key="job_type_min_index"))
        max_idx = int(st.number_input("job_type_max_index", min_value=min_idx, max_value=50, value=int(_dj("job_type_max_index", cfg["job_type_max_index"])), step=1, key="job_type_max_index"))
        cfg["job_type_min_index"] = min_idx
        cfg["job_type_max_index"] = max_idx
        selected_types = list(range(min_idx, max_idx + 1))
        cfg["job_type_indices"] = []
    else:
        selected_types = st.multiselect("job_type_indices", options=list(range(1, 51)), default=_dj("job_type_indices", list(range(1, 11))), key="job_type_indices")
        if not selected_types:
            selected_types = [1]
        cfg["job_type_indices"] = selected_types

    cfg["num_components_divisor"] = int(st.number_input("num_components_divisor", min_value=1, max_value=50, value=int(_dj("num_components_divisor", cfg["num_components_divisor"])), step=1, key="num_components_divisor"))
    cfg["deadline_base"] = float(st.number_input("deadline_base", min_value=0.0, max_value=1000.0, value=float(_dj("deadline_base", cfg["deadline_base"])), step=1.0, key="deadline_base"))
    cfg["deadline_step"] = float(st.number_input("deadline_step", min_value=0.0, max_value=1000.0, value=float(_dj("deadline_step", cfg["deadline_step"])), step=1.0, key="deadline_step"))
    cfg["capacity_base"] = float(st.number_input("capacity_base", min_value=0.0, max_value=1_000_000.0, value=float(_dj("capacity_base", cfg["capacity_base"])), step=1.0, key="capacity_base"))
    cfg["capacity_step"] = float(st.number_input("capacity_step", min_value=0.0, max_value=1_000_000.0, value=float(_dj("capacity_step", cfg["capacity_step"])), step=1.0, key="capacity_step"))
    cfg["max_workers"] = int(st.number_input("max_workers", min_value=1, max_value=100, value=int(_dj("max_workers", cfg["max_workers"])), step=1, key="max_workers"))
    cfg["quality_cost_base"] = float(st.number_input("quality_cost_base", min_value=0.0, max_value=1_000_000.0, value=float(_dj("quality_cost_base", cfg["quality_cost_base"])), step=1.0, key="quality_cost_base"))
    cfg["quality_cost_step"] = float(st.number_input("quality_cost_step", min_value=0.0, max_value=1_000_000.0, value=float(_dj("quality_cost_step", cfg["quality_cost_step"])), step=1.0, key="quality_cost_step"))
    cfg["material_cost_base"] = float(st.number_input("material_cost_base", min_value=0.0, max_value=10_000_000.0, value=float(_dj("material_cost_base", cfg["material_cost_base"])), step=1.0, key="material_cost_base"))
    cfg["material_cost_step"] = float(st.number_input("material_cost_step", min_value=0.0, max_value=10_000_000.0, value=float(_dj("material_cost_step", cfg["material_cost_step"])), step=1.0, key="material_cost_step"))
    cfg["max_daily_manhours_per_worker"] = float(st.number_input("max_daily_manhours_per_worker", min_value=0.1, max_value=100.0, value=float(_dj("max_daily_manhours_per_worker", cfg["max_daily_manhours_per_worker"])), step=0.1, key="max_daily_manhours_per_worker"))
    cfg["base_labor_rate_base"] = float(st.number_input("base_labor_rate_base", min_value=0.0, max_value=10_000.0, value=float(_dj("base_labor_rate_base", cfg["base_labor_rate_base"])), step=1.0, key="base_labor_rate_base"))
    cfg["base_labor_rate_step"] = float(st.number_input("base_labor_rate_step", min_value=0.0, max_value=10_000.0, value=float(_dj("base_labor_rate_step", cfg["base_labor_rate_step"])), step=1.0, key="base_labor_rate_step"))
    cfg["base_transportation_cost_base"] = float(st.number_input("base_transportation_cost_base", min_value=0.0, max_value=10_000_000.0, value=float(_dj("base_transportation_cost_base", cfg["base_transportation_cost_base"])), step=1.0, key="base_transportation_cost_base"))
    cfg["base_transportation_cost_step"] = float(st.number_input("base_transportation_cost_step", min_value=0.0, max_value=10_000_000.0, value=float(_dj("base_transportation_cost_step", cfg["base_transportation_cost_step"])), step=1.0, key="base_transportation_cost_step"))
    cfg["late_penalty_per_day_base"] = float(st.number_input("late_penalty_per_day_base", min_value=0.0, max_value=1.0, value=float(_dj("late_penalty_per_day_base", cfg["late_penalty_per_day_base"])), step=0.001, key="late_penalty_per_day_base"))
    cfg["late_penalty_per_day_step"] = float(st.number_input("late_penalty_per_day_step", min_value=0.0, max_value=1.0, value=float(_dj("late_penalty_per_day_step", cfg["late_penalty_per_day_step"])), step=0.001, key="late_penalty_per_day_step"))
    cfg["timeline_reliability_target_base"] = float(st.slider("timeline_reliability_target_base", min_value=0.0, max_value=1.0, value=float(_dj("timeline_reliability_target_base", cfg["timeline_reliability_target_base"])), step=0.001, key="timeline_reliability_target_base"))
    cfg["timeline_reliability_target_step"] = float(st.slider("timeline_reliability_target_step", min_value=0.0, max_value=1.0, value=float(_dj("timeline_reliability_target_step", cfg["timeline_reliability_target_step"])), step=0.001, key="timeline_reliability_target_step"))
    cfg["quality_reliability_target_base"] = float(st.slider("quality_reliability_target_base", min_value=0.0, max_value=1.0, value=float(_dj("quality_reliability_target_base", cfg["quality_reliability_target_base"])), step=0.001, key="quality_reliability_target_base"))
    cfg["quality_reliability_target_step"] = float(st.slider("quality_reliability_target_step", min_value=0.0, max_value=1.0, value=float(_dj("quality_reliability_target_step", cfg["quality_reliability_target_step"])), step=0.001, key="quality_reliability_target_step"))

    st.subheader("Per-Type Hardcoded Job Overrides")
    override_types = st.multiselect("Override specific job types", options=selected_types, default=_dj("job_override_types", []), key="job_override_types")
    overrides: dict[str, dict] = {}
    for jt in override_types:
        with st.expander(f"Type {jt} overrides", expanded=False):
            overrides[str(jt)] = {
                "num_components": int(st.number_input(f"T{jt} num_components", min_value=1, max_value=100, value=max(1, int(cfg["num_components_divisor"])), step=1, key=f"job_{jt}_num_components")),
                "deadline_days": int(st.number_input(f"T{jt} deadline_days", min_value=1, max_value=max(1, max_days * 2), value=max(1, int(cfg["deadline_base"] + jt * cfg["deadline_step"])), step=1, key=f"job_{jt}_deadline_days")),
                "capacity_needed": float(st.number_input(f"T{jt} capacity_needed", min_value=0.0, max_value=10_000_000.0, value=float(cfg["capacity_base"] + cfg["capacity_step"] * (jt - 1)), step=1.0, key=f"job_{jt}_capacity_needed")),
                "max_workers": int(st.number_input(f"T{jt} max_workers", min_value=1, max_value=100, value=int(cfg["max_workers"]), step=1, key=f"job_{jt}_max_workers")),
                "quality_cost": float(st.number_input(f"T{jt} quality_cost", min_value=0.0, max_value=1_000_000.0, value=float(cfg["quality_cost_base"] + cfg["quality_cost_step"] * (jt - 1)), step=1.0, key=f"job_{jt}_quality_cost")),
                "material_cost": float(st.number_input(f"T{jt} material_cost", min_value=0.0, max_value=10_000_000.0, value=float(cfg["material_cost_base"] + cfg["material_cost_step"] * (jt - 1)), step=1.0, key=f"job_{jt}_material_cost")),
                "late_penalty_per_day": float(st.number_input(f"T{jt} late_penalty_per_day", min_value=0.0, max_value=1.0, value=float(cfg["late_penalty_per_day_base"] + cfg["late_penalty_per_day_step"] * (jt - 1)), step=0.001, key=f"job_{jt}_late_penalty_per_day")),
                "max_daily_manhours_per_worker": float(st.number_input(f"T{jt} max_daily_manhours_per_worker", min_value=0.1, max_value=100.0, value=float(cfg["max_daily_manhours_per_worker"]), step=0.1, key=f"job_{jt}_max_daily_manhours_per_worker")),
                "base_labor_rate": float(st.number_input(f"T{jt} base_labor_rate", min_value=0.0, max_value=10_000.0, value=float(cfg["base_labor_rate_base"] + cfg["base_labor_rate_step"] * (jt - 1)), step=1.0, key=f"job_{jt}_base_labor_rate")),
                "base_transportation_cost": float(st.number_input(f"T{jt} base_transportation_cost", min_value=0.0, max_value=10_000_000.0, value=float(cfg["base_transportation_cost_base"] + cfg["base_transportation_cost_step"] * (jt - 1)), step=1.0, key=f"job_{jt}_base_transportation_cost")),
                "timeline_reliability_target": float(st.slider(f"T{jt} timeline_reliability_target", min_value=0.0, max_value=1.0, value=float(cfg["timeline_reliability_target_base"] + cfg["timeline_reliability_target_step"] * jt), step=0.001, key=f"job_{jt}_timeline_reliability_target")),
                "quality_reliability_target": float(st.slider(f"T{jt} quality_reliability_target", min_value=0.0, max_value=1.0, value=float(cfg["quality_reliability_target_base"] + cfg["quality_reliability_target_step"] * jt), step=0.001, key=f"job_{jt}_quality_reliability_target")),
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
    backup_shop_depth: int,
    capacity_utilization_mean: Optional[float],
    capacity_utilization_std: Optional[float],
    job_generation_mode: str,
    job_generation_probability: float,
    job_generation_days: list[int],
    shop_type_params_override: Optional[dict[str, dict]],
    job_config: Optional[dict],
    output_dir: str,
    allocation_planner_mode: str = "fast",
    enforce_first_layer_timeline_filter: bool = False,
    generate_plots: bool = True,
    text_overrides: Optional[dict[str, str]] = None,
) -> tuple[dict, list[SimulationRun]]:
    all_stats = []
    run_sims: list[SimulationRun] = []
    simulation_init_params = inspect.signature(SimulationRun.__init__).parameters
    supports_allocation_planner_mode = "allocation_planner_mode" in simulation_init_params
    supports_first_layer_timeline_filter = "enforce_first_layer_timeline_filter" in simulation_init_params

    for r in range(runs):
        sim_kwargs = dict(
            num_shops=shops,
            num_days=days,
            jobs_per_day=jobs_per_day,
            rng_seed=seed + r,
            num_quality_checks=quality_checks,
            failure_penalty_rate=failure_penalty_rate,
            max_delay_factor=max_delay_factor,
            backup_shop_depth=backup_shop_depth,
            capacity_utilization_mean=capacity_utilization_mean,
            capacity_utilization_std=capacity_utilization_std,
            job_generation_mode=job_generation_mode,
            job_generation_probability=job_generation_probability,
            job_generation_days=job_generation_days,
            shop_type_params_override=shop_type_params_override,
            job_config=job_config,
        )

        # Backward compatibility for hosted environments temporarily running older simulation.py.
        if supports_allocation_planner_mode:
            sim_kwargs["allocation_planner_mode"] = allocation_planner_mode
        if supports_first_layer_timeline_filter:
            sim_kwargs["enforce_first_layer_timeline_filter"] = enforce_first_layer_timeline_filter

        sim = SimulationRun(**sim_kwargs)
        sim.run()
        all_stats.append(sim.get_statistics())
        run_sims.append(sim)

    agg = aggregate_runs(all_stats)
    if generate_plots:
        plot_shop_statistics(agg, output_dir=output_dir, label="Primary", text_overrides=text_overrides)
        plot_job_statistics(agg, output_dir=output_dir, label="Primary", text_overrides=text_overrides)
        plot_success_rates_vs_targets(agg, output_dir=output_dir, label="Primary", text_overrides=text_overrides)
        plot_completed_only_quality_rate(agg, output_dir=output_dir, label="Primary", text_overrides=text_overrides)
    return agg, run_sims


def _mean_metric(values_by_type: dict) -> float:
    vals = list(values_by_type.values()) if values_by_type else []
    return float(sum(vals) / len(vals)) if vals else 0.0


def _run_primary_quality_check_sweep(
    checks_min: int,
    checks_max: int,
    runs: int,
    shops: int,
    days: int,
    jobs_per_day: int,
    seed: int,
    failure_penalty_rate: float,
    max_delay_factor: float,
    backup_shop_depth: int,
    allocation_planner_mode: str,
    enforce_first_layer_timeline_filter: bool,
    capacity_utilization_mean: Optional[float],
    capacity_utilization_std: Optional[float],
    job_generation_mode: str,
    job_generation_probability: float,
    job_generation_days: list[int],
    shop_type_params_override: Optional[dict[str, dict]],
    job_config: Optional[dict],
    output_dir: str,
    text_overrides: Optional[dict[str, str]] = None,
):
    rows = []
    for qc in range(int(checks_min), int(checks_max) + 1):
        agg, _ = _run_primary(
            runs=runs,
            shops=shops,
            days=days,
            jobs_per_day=jobs_per_day,
            seed=seed,
            quality_checks=qc,
            failure_penalty_rate=failure_penalty_rate,
            max_delay_factor=max_delay_factor,
            backup_shop_depth=backup_shop_depth,
            capacity_utilization_mean=capacity_utilization_mean,
            capacity_utilization_std=capacity_utilization_std,
            job_generation_mode=job_generation_mode,
            job_generation_probability=job_generation_probability,
            job_generation_days=job_generation_days,
            shop_type_params_override=shop_type_params_override,
            job_config=job_config,
            output_dir=output_dir,
            allocation_planner_mode=allocation_planner_mode,
            enforce_first_layer_timeline_filter=enforce_first_layer_timeline_filter,
            generate_plots=False,
            text_overrides=text_overrides,
        )
        total = int(agg.get("quality_total_cases", 0))
        passed = int(agg.get("quality_passed_cases", 0))
        rows.append(
            {
                "quality_checks": qc,
                "mean_cost": _mean_metric(agg.get("agg_cost_mean", {})),
                "quality_success_rate": (passed / total) if total > 0 else 0.0,
                "timeline_success_rate": _mean_metric(agg.get("agg_timeline_rate", {})),
            }
        )

    plot_primary_quality_check_sweep(rows, output_dir=output_dir, label="Primary", text_overrides=text_overrides)
    return rows


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
    text_overrides: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for method in methods:
        all_stats = []
        all_jobs = []
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
            all_jobs.extend(sim.jobs.values())
            if exemplar_sim is None:
                exemplar_sim = sim

        agg = aggregate_runs(all_stats)
        results[method] = {"agg": agg, "sim": exemplar_sim, "jobs": all_jobs}
        label = "BaseCase" if method == "random_cheapest" else "PartialNetwork"
        plot_shop_statistics(agg, output_dir=output_dir, label=label, text_overrides=text_overrides)
        plot_job_statistics(agg, output_dir=output_dir, label=label, text_overrides=text_overrides)
        plot_success_rates_vs_targets(agg, output_dir=output_dir, label=label, text_overrides=text_overrides)
        plot_completed_only_quality_rate(agg, output_dir=output_dir, label=label, text_overrides=text_overrides)

    return results


def _render_images(output_dir: Path):
    # Network maps are rendered separately via the run slider.
    images = [p for p in sorted(output_dir.glob("*.png")) if not p.name.startswith("network_map_")]
    if not images:
        st.warning("No plots were generated.")
        return

    for image_path in images:
        st.image(str(image_path), caption=image_path.name, width="stretch")


def _quality_audit_rows(label: str, agg: Optional[dict]) -> list[dict]:
    if not agg:
        return []
    total = int(agg.get("quality_total_cases", 0))
    passed = int(agg.get("quality_passed_cases", 0))
    failed = int(agg.get("quality_failed_cases", 0))
    completed = int(agg.get("completed_jobs_count", 0))
    failed_final_quality = int(agg.get("failed_final_quality_count", 0))
    rate = (passed / total) if total > 0 else 0.0
    return [
        {
            "mode": label,
            "total_jobs_run": total,
            "completed_jobs": completed,
            "quality_passed_final": passed,
            "quality_failed_final_or_unfinished": failed,
            "failed_final_quality_completed": failed_final_quality,
            "quality_success_rate": round(rate, 4),
        }
    ]


def _mean_mode_cost(agg: Optional[dict]) -> Optional[float]:
    if not agg:
        return None
    vals = list((agg.get("agg_cost_mean") or {}).values())
    if not vals:
        return None
    return float(sum(vals) / max(1, len(vals)))


def _main_timeline_failure_cause(jobs: list) -> str:
    causes: dict[str, int] = {}
    for job in jobs:
        if getattr(job, "timeline_success", False):
            continue
        if not getattr(job, "completed", False):
            key = "Unfinished by simulation horizon"
        elif getattr(job, "days_late", 0) > 0:
            key = "Completed after deadline"
        else:
            key = "Completed within deadline but missed 2-day timeline buffer"
        causes[key] = causes.get(key, 0) + 1
    if not causes:
        return "No timeline failures observed"
    return max(causes.items(), key=lambda x: x[1])[0]


def _main_quality_failure_cause(jobs: list) -> str:
    causes: dict[str, int] = {}
    for job in jobs:
        if getattr(job, "completed", False) and getattr(job, "quality_success", False):
            continue
        if not getattr(job, "completed", False):
            key = "Unfinished by simulation horizon"
        elif any(getattr(comp, "quality_failed", False) for comp in getattr(job, "components", [])):
            key = "Exceeded max-delay window after repeated quality reroutes"
        else:
            key = "Final quality check failure"
        causes[key] = causes.get(key, 0) + 1
    if not causes:
        return "No quality failures observed"
    return max(causes.items(), key=lambda x: x[1])[0]


def _build_run_summary_text(
    agg_primary: Optional[dict],
    primary_sims: Optional[list[SimulationRun]],
    agg_secondary: Optional[dict],
) -> str:
    mode_costs: list[tuple[str, float]] = []
    all_jobs = []

    total_primary_jobs = 0
    rerouted_jobs = 0  # unique jobs with at least one reroute
    reroute_events = 0  # total reroute events across all components/jobs
    if primary_sims:
        for sim in primary_sims:
            for job in sim.jobs.values():
                total_primary_jobs += 1
                all_jobs.append(job)
                job_rerouted = False
                for comp in job.components:
                    events = max(0, len(getattr(comp, "shop_assignment_history", [])) - 1)
                    reroute_events += events
                    if events > 0:
                        job_rerouted = True
                if job_rerouted:
                    rerouted_jobs += 1

    reroute_pct = (100.0 * rerouted_jobs / total_primary_jobs) if total_primary_jobs > 0 else 0.0
    avg_reroutes_per_job = (reroute_events / total_primary_jobs) if total_primary_jobs > 0 else 0.0
    avg_reroutes_per_rerouted_job = (reroute_events / rerouted_jobs) if rerouted_jobs > 0 else 0.0

    c = _mean_mode_cost(agg_primary)
    if c is not None:
        mode_costs.append(("Full Network", c))

    if agg_secondary:
        base = (agg_secondary.get("random_cheapest") or {}).get("agg")
        partial = (agg_secondary.get("quality_top") or {}).get("agg")
        base_jobs = (agg_secondary.get("random_cheapest") or {}).get("jobs") or []
        partial_jobs = (agg_secondary.get("quality_top") or {}).get("jobs") or []
        all_jobs.extend(base_jobs)
        all_jobs.extend(partial_jobs)
        bc = _mean_mode_cost(base)
        pc = _mean_mode_cost(partial)
        if bc is not None:
            mode_costs.append(("Base Case", bc))
        if pc is not None:
            mode_costs.append(("Partial Network", pc))

    if mode_costs:
        cheapest_mode = min(mode_costs, key=lambda x: x[1])
        expensive_mode = max(mode_costs, key=lambda x: x[1])
        cheapest_line = f"{cheapest_mode[0]} (mean cost ${cheapest_mode[1]:,.0f})"
        expensive_line = f"{expensive_mode[0]} (mean cost ${expensive_mode[1]:,.0f})"
    else:
        cheapest_line = "N/A"
        expensive_line = "N/A"

    timeline_cause = _main_timeline_failure_cause(all_jobs)
    quality_cause = _main_quality_failure_cause(all_jobs)

    return (
        f"Primary jobs (total): {total_primary_jobs}\n"
        f"Primary jobs rerouted (unique): {rerouted_jobs}\n"
        f"Primary reroute events (total): {reroute_events}\n"
        f"Primary rerouted jobs (%): {reroute_pct:.1f}%\n"
        f"Primary avg reroutes per job: {avg_reroutes_per_job:.2f}\n"
        f"Primary avg reroutes per rerouted job: {avg_reroutes_per_rerouted_job:.2f}\n"
        f"Cheapest mode: {cheapest_line}\n"
        f"Most expensive mode: {expensive_line}\n"
        f"Main cause of timeline failures: {timeline_cause}\n"
        f"Main cause of quality failures: {quality_cause}"
    )


def main():
    st.set_page_config(page_title="Jeevy Simulation Dashboard", layout="wide")

    pending = st.session_state.pop("_pending_loaded_settings", None)
    if pending is not None:
        _apply_loaded_settings_to_session_state(pending)

    st.title("Jeevy Simulation Dashboard")
    st.write("Configure parameters on the left, then run simulations and view plots on the right.")

    with st.sidebar:
        st.header("Run Controls")
        case_mode = st.selectbox(
            "Case",
            ["base case", "partial network", "full network", "all cases"],
            index=["base case", "partial network", "full network", "all cases"].index(_d("case_mode", "base case")) if _d("case_mode", "base case") in ["base case", "partial network", "full network", "all cases"] else 0,
            key="case_mode",
        )
        runs = st.number_input("Runs", min_value=1, max_value=200, value=int(_d("runs", 3)), step=1, key="runs")
        shops = st.number_input("Shops", min_value=1, max_value=2000, value=int(_d("shops", 100)), step=1, key="shops")
        days = st.number_input("Days", min_value=1, max_value=3650, value=int(_d("days", 365)), step=1, key="days")
        jobs_per_day = st.number_input("Jobs Generated Per Event", min_value=1, max_value=1000, value=int(_d("jobs_per_day", 5)), step=1, key="jobs_per_day")
        seed = st.number_input("Base Seed", min_value=0, max_value=2_000_000_000, value=int(_d("seed", 42)), step=1, key="seed")

        st.subheader("Quality / Delay")
        quality_checks = st.number_input("Quality Checks (Primary)", min_value=1, max_value=20, value=int(_d("quality_checks", 3)), step=1, key="quality_checks")
        max_delay_factor = st.number_input("Max Delay Factor", min_value=0.0, max_value=10.0, value=float(_d("max_delay_factor", 10.0)), step=0.1, key="max_delay_factor")
        failure_penalty_rate = st.number_input("Failure Penalty Rate", min_value=0.0, max_value=5.0, value=float(_d("failure_penalty_rate", 0.20)), step=0.01, key="failure_penalty_rate")
        backup_shop_depth = st.number_input("Primary Backup Shop Depth", min_value=1, max_value=6, value=int(_d("backup_shop_depth", 3)), step=1, key="backup_shop_depth")
        allocation_planner_mode = st.selectbox(
            "Primary Allocation Planner Mode",
            ["fast", "thorough"],
            index=0 if _d("allocation_planner_mode", "fast") == "fast" else 1,
            key="allocation_planner_mode",
            help="fast = quicker bounded search; thorough = deeper combo search, slower.",
        )
        enforce_first_layer_timeline_filter = st.checkbox(
            "Filter out first-layer shops that miss timeline requirement",
            value=bool(_d("enforce_first_layer_timeline_filter", False)),
            key="enforce_first_layer_timeline_filter",
            help="When enabled, primary shops that cannot meet the timeline window on their own are excluded before combo evaluation.",
        )
        enable_quality_check_sweep = st.checkbox(
            "Sweep quality checks for primary",
            value=bool(_d("enable_quality_check_sweep", False)),
            key="enable_quality_check_sweep",
            help="Run primary case repeatedly across a quality-check range and plot cost/quality/timeline outcomes.",
        )
        sweep_quality_checks_min = int(_d("sweep_quality_checks_min", 1))
        sweep_quality_checks_max = int(_d("sweep_quality_checks_max", 6))
        if enable_quality_check_sweep:
            sweep_quality_checks_min = int(st.number_input("Sweep min quality checks", min_value=1, max_value=20, value=sweep_quality_checks_min, step=1, key="sweep_quality_checks_min"))
            sweep_quality_checks_max = int(st.number_input("Sweep max quality checks", min_value=sweep_quality_checks_min, max_value=20, value=max(sweep_quality_checks_min, sweep_quality_checks_max), step=1, key="sweep_quality_checks_max"))

        st.subheader("Capacity Utilization")
        use_util_overrides = st.checkbox("Override Utilization Mean/Std for all shop types", value=bool(_d("use_util_overrides", False)), key="use_util_overrides")
        capacity_utilization_mean = None
        capacity_utilization_std = None
        if use_util_overrides:
            capacity_utilization_mean = st.number_input("Capacity Utilization Mean", min_value=0.0, max_value=1.0, value=float(_d("capacity_utilization_mean", 0.10)), step=0.01, key="capacity_utilization_mean")
            capacity_utilization_std = st.number_input("Capacity Utilization Std Dev", min_value=0.0, max_value=1.0, value=float(_d("capacity_utilization_std", 0.02)), step=0.01, key="capacity_utilization_std")

        st.subheader("Job Generation")
        job_generation_mode = st.selectbox(
            "Job Generation Mode",
            ["start_only", "daily", "probabilistic", "custom_days"],
            index=["start_only", "daily", "probabilistic", "custom_days"].index(_d("job_generation_mode", "start_only")) if _d("job_generation_mode", "start_only") in ["start_only", "daily", "probabilistic", "custom_days"] else 0,
            key="job_generation_mode",
        )
        job_generation_probability = 0.0
        custom_days_selection = [0]
        if job_generation_mode == "probabilistic":
            job_generation_probability = st.number_input(
                "Generation Probability Per Day",
                min_value=0.0,
                max_value=1.0,
                value=float(_d("job_generation_probability", 0.10)),
                step=0.01,
                key="job_generation_probability",
            )
        if job_generation_mode == "custom_days":
            day_options = list(range(int(days)))
            loaded_days = _d("job_generation_days", [0])
            default_days = [int(d) for d in loaded_days if 0 <= int(d) < int(days)] if int(days) > 0 else []
            if not default_days and int(days) > 0:
                default_days = [0]
            custom_days_selection = st.multiselect("Custom generation days", options=day_options, default=default_days, key="custom_generation_days")
            if not custom_days_selection:
                custom_days_selection = [0]

        output_base = st.text_input("Output Base Directory", value=str(_d("output_base", "dashboard_runs")), key="output_base")

        st.subheader("Plot Text")
        default_plot_text = _d("plot_text_overrides", {})
        default_plot_text_json = json.dumps(default_plot_text, indent=2) if isinstance(default_plot_text, dict) else "{}"
        plot_text_overrides_raw = st.text_area(
            "Title/Axis Label Overrides (JSON)",
            value=default_plot_text_json,
            height=160,
            key="plot_text_overrides_raw",
            help=(
                "Map existing chart text to replacement text. Example:\n"
                "{\n"
                "  \"Quality Success Rate\": \"Quality Rate\",\n"
                "  \"Rate\": \"Probability\",\n"
                "  \"X Location\": \"Longitude\"\n"
                "}"
            ),
        )
        plot_text_overrides: dict[str, str] = {}
        try:
            plot_text_overrides = _parse_plot_text_overrides(plot_text_overrides_raw)
        except Exception as exc:
            st.warning(f"Invalid plot text override JSON. Using defaults. ({exc})")

        shop_type_params_override = _shop_params_controls()
        job_config_override, selected_job_types = _job_config_controls(max_days=int(days))

        run_clicked = st.button("Run Simulation", type="primary", width="stretch")

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("Assumptions")

        st.markdown("**Shop Parameters**")
        shop_df = _shop_assumptions_df(shop_type_params_override)
        st.dataframe(shop_df, width="stretch")

        st.markdown("**Job Parameters**")
        job_df = _job_assumptions_df(job_config_override, selected_job_types)
        # Highlight rows where deadline < min_days_practical
        def _style_row(row):
            flag = row.get("deadline_days", 999) < row.get("min_days_practical", 0)
            return ["background-color: #ffcccc" if flag else "" for _ in row]
        st.dataframe(
            job_df.style.apply(_style_row, axis=1),
            width="stretch",
        )
        st.caption(
            "min_days_ideal = ceil(capacity / (max_workers × mh/worker)) — no delay, perfect efficiency.  \n"
            "min_days_practical = ideal ÷ best shop efficiency + 1 allocation day.  \n"
            "Rows highlighted red have deadline < min_days_practical."
        )

        st.subheader("Save / Export / Import Settings")
        settings_name = st.text_input("Settings name", value="my_settings", key="settings_name_input")
        save_col, export_col = st.columns(2)
        with save_col:
            if st.button("💾 Save settings", width="stretch"):
                snap = _collect_settings(
                    case_mode, runs, shops, days, jobs_per_day, seed,
                    quality_checks, max_delay_factor, failure_penalty_rate, backup_shop_depth,
                    allocation_planner_mode,
                    enforce_first_layer_timeline_filter,
                    enable_quality_check_sweep, sweep_quality_checks_min, sweep_quality_checks_max,
                    use_util_overrides, capacity_utilization_mean, capacity_utilization_std,
                    job_generation_mode, job_generation_probability, custom_days_selection,
                    output_base, shop_type_params_override, job_config_override,
                    plot_text_overrides,
                )
                path = _settings_dir() / f"{settings_name.strip() or 'settings'}.json"
                path.write_text(json.dumps(snap, indent=2))
                st.success(f"Saved to {path}")
        with export_col:
            snap = _collect_settings(
                case_mode, runs, shops, days, jobs_per_day, seed,
                quality_checks, max_delay_factor, failure_penalty_rate, backup_shop_depth,
                allocation_planner_mode,
                enforce_first_layer_timeline_filter,
                enable_quality_check_sweep, sweep_quality_checks_min, sweep_quality_checks_max,
                use_util_overrides, capacity_utilization_mean, capacity_utilization_std,
                job_generation_mode, job_generation_probability, custom_days_selection,
                output_base, shop_type_params_override, job_config_override,
                plot_text_overrides,
            )
            st.download_button(
                "📥 Export JSON",
                data=json.dumps(snap, indent=2),
                file_name=f"{settings_name.strip() or 'settings'}.json",
                mime="application/json",
                width="stretch",
            )

        uploaded = st.file_uploader("📤 Import settings JSON", type="json", key="settings_uploader")
        if uploaded is not None:
            payload = uploaded.getvalue()
            payload_sig = hash(payload)
            if st.session_state.get("_last_import_payload_sig") != payload_sig:
                loaded = json.loads(payload.decode("utf-8"))
                st.session_state["_pending_loaded_settings"] = loaded
                st.session_state["_last_import_payload_sig"] = payload_sig
                st.rerun()

        saved_files = sorted(_settings_dir().glob("*.json"))
        if saved_files:
            chosen = st.selectbox(
                "Load saved settings",
                options=["— select —"] + [f.stem for f in saved_files],
                key="load_saved_select",
            )
            if chosen != "— select —":
                if st.session_state.get("_last_loaded_settings_name") != chosen:
                    loaded = json.loads((_settings_dir() / f"{chosen}.json").read_text())
                    st.session_state["_pending_loaded_settings"] = loaded
                    st.session_state["_last_loaded_settings_name"] = chosen
                    st.rerun()

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
                    primary_sims: list[SimulationRun] = []
                    map_paths: list[Path] = []

                    if case_mode in ("full network", "all cases"):
                        agg_primary, primary_sims = _run_primary(
                            runs=int(runs),
                            shops=int(shops),
                            days=int(days),
                            jobs_per_day=int(jobs_per_day),
                            seed=int(seed),
                            quality_checks=int(quality_checks),
                            failure_penalty_rate=float(failure_penalty_rate),
                            max_delay_factor=float(max_delay_factor),
                            backup_shop_depth=int(backup_shop_depth),
                            capacity_utilization_mean=capacity_utilization_mean,
                            capacity_utilization_std=capacity_utilization_std,
                            job_generation_mode=job_generation_mode,
                            job_generation_probability=float(job_generation_probability),
                            job_generation_days=job_generation_days,
                            shop_type_params_override=shop_type_params_override,
                            job_config=job_config_override,
                            output_dir=str(output_dir),
                            allocation_planner_mode=allocation_planner_mode,
                            enforce_first_layer_timeline_filter=enforce_first_layer_timeline_filter,
                            text_overrides=plot_text_overrides,
                        )
                        if enable_quality_check_sweep:
                            _run_primary_quality_check_sweep(
                                checks_min=sweep_quality_checks_min,
                                checks_max=sweep_quality_checks_max,
                                runs=int(runs),
                                shops=int(shops),
                                days=int(days),
                                jobs_per_day=int(jobs_per_day),
                                seed=int(seed),
                                failure_penalty_rate=float(failure_penalty_rate),
                                max_delay_factor=float(max_delay_factor),
                                backup_shop_depth=int(backup_shop_depth),
                                allocation_planner_mode=allocation_planner_mode,
                                enforce_first_layer_timeline_filter=enforce_first_layer_timeline_filter,
                                capacity_utilization_mean=capacity_utilization_mean,
                                capacity_utilization_std=capacity_utilization_std,
                                job_generation_mode=job_generation_mode,
                                job_generation_probability=float(job_generation_probability),
                                job_generation_days=job_generation_days,
                                shop_type_params_override=shop_type_params_override,
                                job_config=job_config_override,
                                output_dir=str(output_dir),
                                text_overrides=plot_text_overrides,
                            )
                        # Re-label primary figures to match dashboard terminology.
                        plot_shop_statistics(agg_primary, output_dir=str(output_dir), label="FullNetwork", text_overrides=plot_text_overrides)
                        plot_job_statistics(agg_primary, output_dir=str(output_dir), label="FullNetwork", text_overrides=plot_text_overrides)
                        plot_success_rates_vs_targets(agg_primary, output_dir=str(output_dir), label="FullNetwork", text_overrides=plot_text_overrides)
                        plot_completed_only_quality_rate(agg_primary, output_dir=str(output_dir), label="FullNetwork", text_overrides=plot_text_overrides)
                        for idx, sim in enumerate(primary_sims, start=1):
                            map_paths.append(
                                _plot_network_map(
                                    sim,
                                    f"Full Network Flow Map (run {idx})",
                                    output_dir,
                                    f"network_map_full_network_run_{idx}.png",
                                    text_overrides=plot_text_overrides,
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
                            text_overrides=plot_text_overrides,
                        )
                        # Generate plots for secondary simulation results
                        if case_mode == "base case":
                            base_agg = agg_secondary.get("random_cheapest", {}).get("agg")
                            if base_agg:
                                plot_shop_statistics(base_agg, output_dir=str(output_dir), label="BaseCase", text_overrides=plot_text_overrides)
                                plot_job_statistics(base_agg, output_dir=str(output_dir), label="BaseCase", text_overrides=plot_text_overrides)
                                plot_success_rates_vs_targets(base_agg, output_dir=str(output_dir), label="BaseCase", text_overrides=plot_text_overrides)
                                plot_completed_only_quality_rate(base_agg, output_dir=str(output_dir), label="BaseCase", text_overrides=plot_text_overrides)
                        elif case_mode == "partial network":
                            partial_agg = agg_secondary.get("quality_top", {}).get("agg")
                            if partial_agg:
                                plot_shop_statistics(partial_agg, output_dir=str(output_dir), label="PartialNetwork", text_overrides=plot_text_overrides)
                                plot_job_statistics(partial_agg, output_dir=str(output_dir), label="PartialNetwork", text_overrides=plot_text_overrides)
                                plot_success_rates_vs_targets(partial_agg, output_dir=str(output_dir), label="PartialNetwork", text_overrides=plot_text_overrides)
                                plot_completed_only_quality_rate(partial_agg, output_dir=str(output_dir), label="PartialNetwork", text_overrides=plot_text_overrides)

                    if enable_quality_check_sweep and case_mode not in ("full network", "all cases"):
                        st.warning("Quality-check sweep runs only for primary/full-network mode. Switch Case to 'full network' or 'all cases'.")

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
                            text_overrides=plot_text_overrides,
                        )
                        plot_completed_only_quality_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                            text_overrides=plot_text_overrides,
                        )
                        plot_job_cost_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                            text_overrides=plot_text_overrides,
                        )
                        plot_shop_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                            text_overrides=plot_text_overrides,
                        )
                        plot_mode_operational_comparison(
                            agg_primary=agg_primary,
                            agg_secondary_rand=agg_secondary["random_cheapest"]["agg"],
                            agg_secondary_qual=agg_secondary["quality_top"]["agg"],
                            output_dir=str(output_dir),
                            text_overrides=plot_text_overrides,
                        )

                st.session_state["last_output_dir"] = str(output_dir)
                st.session_state["last_map_paths"] = [str(p) for p in map_paths]
                st.session_state["last_quality_audit"] = {
                    "primary": agg_primary,
                    "secondary": agg_secondary,
                }
                st.session_state["last_summary_text"] = _build_run_summary_text(
                    agg_primary=agg_primary,
                    primary_sims=primary_sims,
                    agg_secondary=agg_secondary,
                )
                st.success(f"Run complete. Plots saved to: {output_dir}")
            except Exception as exc:
                st.error(f"Run failed: {exc}")

        summary_text = st.session_state.get("last_summary_text")
        if summary_text:
            st.subheader("Run Summary")
            st.text_area("High-level run insights", value=summary_text, height=170, disabled=True)

        quality_audit = st.session_state.get("last_quality_audit")
        if quality_audit:
            rows = []
            rows += _quality_audit_rows("Full Network", quality_audit.get("primary"))
            secondary = quality_audit.get("secondary") or {}
            rows += _quality_audit_rows("Base Case", secondary.get("random_cheapest", {}).get("agg"))
            rows += _quality_audit_rows("Partial Network", secondary.get("quality_top", {}).get("agg"))
            if rows:
                st.subheader("Quality Audit")
                st.dataframe(pd.DataFrame(rows), width="stretch")
                st.caption("quality_success_rate = quality_passed_final / total_jobs_run")

        map_paths = st.session_state.get("last_map_paths", [])
        if map_paths:
            st.subheader("Network Travel Maps")
            if len(map_paths) > 1:
                run_index = st.slider("Network map run", min_value=1, max_value=len(map_paths), value=len(map_paths), step=1)
                selected = map_paths[run_index - 1]
                st.image(selected, caption=f"Run {run_index}: {Path(selected).name}", width="stretch")
            else:
                st.image(map_paths[0], caption=f"Run 1: {Path(map_paths[0]).name}", width="stretch")

        last_output_dir = st.session_state.get("last_output_dir")
        if last_output_dir:
            _render_images(Path(last_output_dir))
        else:
            st.info("Run a simulation to generate and display plots.")


if __name__ == "__main__":
    main()
