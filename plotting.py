"""
plotting.py — All plots for the simulation results.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SHOP_COLORS = {
    "Elite":   "#0B4F8C",
    "Strong":  "#1F77B4",
    "Average": "#4FA3E3",
    "Risky":   "#9BCBF3",
}

MODE_COLORS = {
    "primary": "#0B4F8C",
    "secondary_random": "#2B7DBD",
    "secondary_quality": "#6AAFE6",
    "target": "#C8E1F6",
}


def _base_style(fig):
    fig.patch.set_facecolor("#FFFFFF")


def _style_axis(ax, grid_y: bool = False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D4DDE6")
    ax.spines["bottom"].set_color("#D4DDE6")
    ax.tick_params(labelsize=11, colors="#16324A")
    ax.grid(False)


def _smooth(arr: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _apply_text_overrides(fig, text_overrides: dict[str, str] | None = None):
    if not text_overrides:
        return

    for ax in fig.axes:
        title = ax.get_title()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if title in text_overrides:
            ax.set_title(text_overrides[title])
        if xlabel in text_overrides:
            ax.set_xlabel(text_overrides[xlabel])
        if ylabel in text_overrides:
            ax.set_ylabel(text_overrides[ylabel])

    # Includes figure-level text such as suptitle.
    for txt in fig.texts:
        original = txt.get_text()
        if original in text_overrides:
            txt.set_text(text_overrides[original])


def plot_shop_statistics(agg: dict, output_dir: str = ".", label: str = "", text_overrides: dict[str, str] | None = None):
    """
    Four subplots:
      1. Mean daily capacity fraction per shop type
      2. Mean daily profit per shop type
      3. Total profit mean per shop type (bar chart)
      4. Total profit variance per shop type (bar chart)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shop_types = agg["shop_types"]
    days = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _base_style(fig)
    fig.suptitle(f"Shop Statistics{' — ' + label if label else ''}", fontsize=18, fontweight="bold", color="#0B2A44")

    # --- 1. Capacity fraction over time (assigned=solid, total incl. busy=dashed) ---
    ax = axes[0, 0]
    for st in shop_types:
        data  = agg["agg_capacity"].get(st)
        total = agg.get("agg_total_utilization", {}).get(st)
        if data is None:
            continue
        if days is None:
            days = np.arange(len(data))
        color = SHOP_COLORS[st]
        ax.plot(days, _smooth(data),  label=f"{st}",       color=color, linewidth=1.8)
        if total is not None:
            ax.plot(days, _smooth(total), linestyle="--", color=color, linewidth=1.2, alpha=0.7)
    ax.set_title("Mean Daily Worker Utilisation\n(solid = assigned to jobs, dashed = total incl. busy)", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Day", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Fraction of workers occupied", fontsize=13, color="#0B2A44")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=False)

    # --- 2. Daily profit per shop type ---
    ax = axes[0, 1]
    for st in shop_types:
        data = agg["agg_profit_mean"].get(st)
        if data is None:
            continue
        d = np.arange(len(data))
        ax.plot(d, _smooth(data), label=st, color=SHOP_COLORS[st], linewidth=1.8)
    ax.set_title("Mean Daily Profit per Shop (by type)", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Day", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Profit ($)", fontsize=13, color="#0B2A44")
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=False)

    # --- 3. Total profit mean ---
    ax = axes[1, 0]
    st_present = [st for st in shop_types if st in agg["agg_profit_total_mean"]]
    vals = [agg["agg_profit_total_mean"][st] for st in st_present]
    colors = [SHOP_COLORS[st] for st in st_present]
    bars = ax.bar(st_present, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Total Annual Profit — Mean per Shop (by type)", fontsize=14, color="#0B2A44")
    ax.set_ylabel("Total Profit ($)", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"${v:,.0f}",
                ha="center", va="bottom", fontsize=9)

    # --- 4. Total profit variance ---
    ax = axes[1, 1]
    vals_var = [agg["agg_profit_total_var"].get(st, 0) for st in st_present]
    bars = ax.bar(st_present, vals_var, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Total Annual Profit — Variance per Shop (by type)", fontsize=14, color="#0B2A44")
    ax.set_ylabel("Profit Variance ($ squared)", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / f"shop_statistics{'_' + label if label else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_job_statistics(agg: dict, output_dir: str = ".", label: str = "", text_overrides: dict[str, str] | None = None):
    """
    Four subplots:
      1. Mean total cost per job type
      2. Cost variance per job type
      3. Quality success rate vs. target per job type
      4. Timeline success rate vs. target per job type
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    job_types = agg.get("job_type_indices") or sorted(agg.get("agg_cost_mean", {}).keys())
    if not job_types:
        job_types = [1]

    cost_means  = [agg["agg_cost_mean"].get(jt, 0)  for jt in job_types]
    cost_vars   = [agg["agg_cost_var"].get(jt, 0)   for jt in job_types]
    qual_rates  = [agg["agg_quality_rate"].get(jt, 0)  for jt in job_types]
    time_rates  = [agg["agg_timeline_rate"].get(jt, 0) for jt in job_types]
    q_targets   = [agg["q_target_map"].get(jt, 0)  for jt in job_types]
    t_targets   = [agg["t_target_map"].get(jt, 0)  for jt in job_types]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _base_style(fig)
    fig.suptitle(f"Job Statistics{' — ' + label if label else ''}", fontsize=18, fontweight="bold", color="#0B2A44")

    x = np.arange(len(job_types))
    bar_width = 0.6

    # --- 1. Cost means ---
    ax = axes[0, 0]
    ax.bar(x, cost_means, bar_width, color=MODE_COLORS["primary"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_title("Mean Total Cost per Job Type", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Cost ($)", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)

    # --- 2. Cost variance ---
    ax = axes[0, 1]
    ax.bar(x, cost_vars, bar_width, color=MODE_COLORS["secondary_random"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_title("Cost Variance per Job Type", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Variance ($ squared)", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)

    # --- 3. Quality success rate vs. target ---
    ax = axes[1, 0]
    ax.bar(x - 0.18, qual_rates,  0.35, label="Actual",  color=MODE_COLORS["primary"], edgecolor="white")
    ax.bar(x + 0.18, q_targets,   0.35, label="Target",  color=MODE_COLORS["target"], edgecolor="white", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_title("Quality Success Rate vs. Target", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Rate", fontsize=13, color="#0B2A44")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=True)

    # --- 4. Timeline success rate vs. target ---
    ax = axes[1, 1]
    ax.bar(x - 0.18, time_rates, 0.35, label="Actual",  color=MODE_COLORS["secondary_random"], edgecolor="white")
    ax.bar(x + 0.18, t_targets,  0.35, label="Target",  color=MODE_COLORS["target"], edgecolor="white", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_title("Timeline Success Rate vs. Target", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Rate", fontsize=13, color="#0B2A44")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / f"job_statistics{'_' + label if label else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_success_rates_vs_targets(agg: dict, output_dir: str = ".", label: str = "", text_overrides: dict[str, str] | None = None):
    """
    Scatter: actual success rate vs. required target, for quality and timeline.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    job_types = agg.get("job_type_indices") or sorted(agg.get("agg_quality_rate", {}).keys())
    if not job_types:
        job_types = [1]
    q_targets  = np.array([agg["q_target_map"].get(jt, 0)  for jt in job_types])
    t_targets  = np.array([agg["t_target_map"].get(jt, 0)  for jt in job_types])
    qual_rates = np.array([agg["agg_quality_rate"].get(jt, 0)  for jt in job_types])
    time_rates = np.array([agg["agg_timeline_rate"].get(jt, 0) for jt in job_types])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _base_style(fig)
    fig.suptitle(f"Success Rates vs. Required Targets{' — ' + label if label else ''}", fontsize=17, fontweight="bold", color="#0B2A44")

    for ax, actual, target, title, color in [
        (axes[0], qual_rates,  q_targets, "Quality",  MODE_COLORS["primary"]),
        (axes[1], time_rates,  t_targets, "Timeline", MODE_COLORS["secondary_random"]),
    ]:
        ax.plot([0, 1], [0, 1], "--", linewidth=1.4, color="#A3B8CC", label="Perfect")
        sc = ax.scatter(target, actual, c=[i + 1 for i in range(len(job_types))],
                        cmap="tab10", s=90, zorder=5)
        for i, jt in enumerate(job_types):
            ax.annotate(f"T{jt}", (target[i], actual[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Required Target Rate", fontsize=13, color="#0B2A44")
        ax.set_ylabel("Actual Rate", fontsize=13, color="#0B2A44")
        ax.set_title(f"{title} Success Rate", fontsize=14, color="#0B2A44")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, frameon=False)
        _style_axis(ax, grid_y=False)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / f"success_vs_targets{'_' + label if label else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_completed_only_quality_rate(agg: dict, output_dir: str = ".", label: str = "", text_overrides: dict[str, str] | None = None):
    """Bar chart: completed-only quality success rate vs target per job type."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    job_types = agg.get("job_type_indices") or sorted(agg.get("agg_quality_rate_completed_only", {}).keys())
    if not job_types:
        job_types = [1]

    completed_only_rates = [agg.get("agg_quality_rate_completed_only", {}).get(jt, 0.0) for jt in job_types]
    q_targets = [agg.get("q_target_map", {}).get(jt, 0.0) for jt in job_types]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    _base_style(fig)
    fig.suptitle(f"Completed-Only Quality Success Rate{' — ' + label if label else ''}", fontsize=17, fontweight="bold", color="#0B2A44")

    x = np.arange(len(job_types))
    ax.bar(x - 0.18, completed_only_rates, 0.35, label="Completed-only actual", color=MODE_COLORS["primary"], edgecolor="white")
    ax.bar(x + 0.18, q_targets, 0.35, label="Target", color=MODE_COLORS["target"], edgecolor="white", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Rate", fontsize=13, color="#0B2A44")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / f"completed_only_quality_rate{'_' + label if label else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_primary_quality_check_sweep(
    sweep_rows: list[dict],
    output_dir: str = ".",
    label: str = "Primary",
    text_overrides: dict[str, str] | None = None,
):
    """Plot cost, quality success, and timeline success across quality-check sweep values."""
    if not sweep_rows:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = sorted(sweep_rows, key=lambda r: int(r["quality_checks"]))
    x = np.array([int(r["quality_checks"]) for r in rows])
    cost = np.array([float(r.get("mean_cost", 0.0)) for r in rows])
    quality = np.array([float(r.get("quality_success_rate", 0.0)) for r in rows])
    timeline = np.array([float(r.get("timeline_success_rate", 0.0)) for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    _base_style(fig)
    fig.suptitle(f"Primary Sweep: Quality Checks vs Outcomes{' — ' + label if label else ''}", fontsize=17, fontweight="bold", color="#0B2A44")

    ax = axes[0]
    ax.plot(x, cost, color=MODE_COLORS["primary"], linewidth=2.0, marker="o")
    ax.set_ylabel("Mean Cost ($)", fontsize=13, color="#0B2A44")
    ax.set_title("Cost", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)

    ax = axes[1]
    ax.plot(x, quality, color=MODE_COLORS["secondary_random"], linewidth=2.0, marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate", fontsize=13, color="#0B2A44")
    ax.set_title("Quality Success Rate", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)

    ax = axes[2]
    ax.plot(x, timeline, color=MODE_COLORS["secondary_quality"], linewidth=2.0, marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate", fontsize=13, color="#0B2A44")
    ax.set_xlabel("Number of Quality Checks", fontsize=13, color="#0B2A44")
    ax.set_title("Timeline Success Rate", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / f"primary_quality_check_sweep{'_' + label if label else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_comparison(agg_primary: dict, agg_secondary_rand: dict, agg_secondary_qual: dict,
                    output_dir: str = ".", text_overrides: dict[str, str] | None = None):
    """Side-by-side quality and timeline success rate comparison across three modes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    job_types = sorted(
        set(agg_primary.get("job_type_indices", []))
        | set(agg_secondary_rand.get("job_type_indices", []))
        | set(agg_secondary_qual.get("job_type_indices", []))
    )
    if not job_types:
        job_types = [1]
    x = np.arange(len(job_types))
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _base_style(fig)
    fig.suptitle("Mode Comparison: Quality & Timeline Success Rates", fontsize=17, fontweight="bold", color="#0B2A44")

    for ax, key, title in [
        (axes[0], "agg_quality_rate",  "Quality Success Rate"),
        (axes[1], "agg_timeline_rate", "Timeline Success Rate"),
    ]:
        p  = [agg_primary.get(key, {}).get(jt, 0)          for jt in job_types]
        sr = [agg_secondary_rand.get(key, {}).get(jt, 0)   for jt in job_types]
        sq = [agg_secondary_qual.get(key, {}).get(jt, 0)   for jt in job_types]

        ax.bar(x - w, p,  w, label="Primary",          color=MODE_COLORS["primary"], edgecolor="white")
        ax.bar(x,     sr, w, label="Sec: Random",      color=MODE_COLORS["secondary_random"], edgecolor="white")
        ax.bar(x + w, sq, w, label="Sec: Quality-top", color=MODE_COLORS["secondary_quality"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{jt}" for jt in job_types])
        ax.set_title(title, fontsize=14, color="#0B2A44")
        ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
        ax.set_ylabel("Rate", fontsize=13, color="#0B2A44")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, frameon=False)
        _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / "mode_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_shop_comparison(agg_primary: dict, agg_secondary_rand: dict, agg_secondary_qual: dict,
                         output_dir: str = ".", text_overrides: dict[str, str] | None = None):
    """
    Combined 4x3 grid comparing shop statistics across all three modes.
    Rows: capacity fraction | daily profit | total profit mean | total profit variance
    Columns: Primary | Sec-Random | Sec-QualTop
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shop_types = ["Elite", "Strong", "Average", "Risky"]
    modes = [
        (agg_primary,        "Primary"),
        (agg_secondary_rand, "Secondary: Random-Cheapest"),
        (agg_secondary_qual, "Secondary: Quality-Top"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    _base_style(fig)
    fig.suptitle("Shop Statistics — All Modes Compared", fontsize=18, fontweight="bold", color="#0B2A44")

    row_titles = [
        "Mean Daily Capacity Utilisation",
        "Mean Daily Profit per Shop ($)",
        "Total Annual Profit — Mean per Shop ($)",
        "Total Annual Profit — Variance per Shop",
    ]

    for col, (agg, mode_label) in enumerate(modes):
        axes[0, col].set_title(mode_label, fontsize=11, fontweight="bold", pad=8)

        # Row 0: Capacity fraction over time (solid=assigned, dashed=total incl. busy)
        ax = axes[0, col]
        for st in shop_types:
            data  = agg["agg_capacity"].get(st)
            total = agg.get("agg_total_utilization", {}).get(st)
            if data is None:
                continue
            days = np.arange(len(data))
            color = SHOP_COLORS[st]
            ax.plot(days, _smooth(data),  label=st, color=color, linewidth=1.6)
            if total is not None:
                ax.plot(days, _smooth(total), linestyle="--", color=color, linewidth=1.0, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Day")
        if col == 0:
            ax.set_ylabel(row_titles[0], fontsize=12, color="#0B2A44")
            ax.legend(fontsize=9, frameon=False)
        _style_axis(ax, grid_y=False)

        # Row 1: Daily profit
        ax = axes[1, col]
        for st in shop_types:
            data = agg["agg_profit_mean"].get(st)
            if data is None:
                continue
            d = np.arange(len(data))
            ax.plot(d, _smooth(data), label=st, color=SHOP_COLORS[st], linewidth=1.6)
        ax.set_xlabel("Day")
        if col == 0:
            ax.set_ylabel(row_titles[1], fontsize=12, color="#0B2A44")
            ax.legend(fontsize=9, frameon=False)
        _style_axis(ax, grid_y=False)

        # Row 2: Total profit mean (bar)
        ax = axes[2, col]
        st_present = [st for st in shop_types if st in agg.get("agg_profit_total_mean", {})]
        vals = [agg["agg_profit_total_mean"][st] for st in st_present]
        colors = [SHOP_COLORS[st] for st in st_present]
        bars = ax.bar(st_present, vals, color=colors, edgecolor="white")
        if col == 0:
            ax.set_ylabel(row_titles[2], fontsize=12, color="#0B2A44")
        _style_axis(ax, grid_y=True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"${v:,.0f}", ha="center", va="bottom", fontsize=7)

        # Row 3: Total profit variance (bar)
        ax = axes[3, col]
        vals_var = [agg.get("agg_profit_total_var", {}).get(st, 0) for st in st_present]
        bars = ax.bar(st_present, vals_var, color=colors, edgecolor="white")
        if col == 0:
            ax.set_ylabel(row_titles[3], fontsize=12, color="#0B2A44")
        _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = out / "shop_comparison_all_modes.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_job_cost_comparison(agg_primary: dict, agg_secondary_rand: dict, agg_secondary_qual: dict,
                              output_dir: str = ".", text_overrides: dict[str, str] | None = None):
    """
    Two-panel figure comparing job costs across all three simulation modes.

    Top panel:    Grouped bar chart — mean total cost per job type, one bar per mode.
    Bottom panel: Grouped bar chart — cost variance per job type, one bar per mode.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    job_types = sorted(
        set(agg_primary.get("job_type_indices", []))
        | set(agg_secondary_rand.get("job_type_indices", []))
        | set(agg_secondary_qual.get("job_type_indices", []))
    )
    if not job_types:
        job_types = [1]
    x = np.arange(len(job_types))
    w = 0.25   # bar width

    modes = [
        (agg_primary,        "Primary",              MODE_COLORS["primary"]),
        (agg_secondary_rand, "Secondary: Random",    MODE_COLORS["secondary_random"]),
        (agg_secondary_qual, "Secondary: Qual-Top",  MODE_COLORS["secondary_quality"]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    _base_style(fig)
    fig.suptitle("Job Cost Comparison Across Simulation Modes", fontsize=18, fontweight="bold", color="#0B2A44")

    # --- Top: mean cost ---
    ax = axes[0]
    for i, (agg, label, color) in enumerate(modes):
        vals = [agg["agg_cost_mean"].get(jt, 0) for jt in job_types]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_title("Mean Total Cost per Job Type", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Mean Cost ($)", fontsize=13, color="#0B2A44")
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=True)

    # Add value labels on top of each bar
    for i, (agg, label, color) in enumerate(modes):
        vals = [agg["agg_cost_mean"].get(jt, 0) for jt in job_types]
        offset = (i - 1) * w
        for xi, v in zip(x, vals):
            if v > 0:
                ax.text(xi + offset, v * 1.005, f"${v/1000:.0f}k",
                        ha="center", va="bottom", fontsize=6.5, color="#333333")

    # --- Bottom: cost variance ---
    ax = axes[1]
    for i, (agg, label, color) in enumerate(modes):
        vals = [agg["agg_cost_var"].get(jt, 0) for jt in job_types]
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, label=label, color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"T{jt}" for jt in job_types])
    ax.set_title("Cost Variance per Job Type", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Job Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Variance ($ squared)", fontsize=13, color="#0B2A44")
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=True)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / "job_cost_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_mode_operational_comparison(agg_primary: dict, agg_secondary_rand: dict, agg_secondary_qual: dict,
                                     output_dir: str = ".", text_overrides: dict[str, str] | None = None):
    """
    Compare operational behavior across modes.

    Left: average shop assignments per type, per mode.
    Right: average days late per mode.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shop_types = ["Elite", "Strong", "Average", "Risky"]
    modes = [
        (agg_primary, "Primary", MODE_COLORS["primary"]),
        (agg_secondary_rand, "Secondary: Random", MODE_COLORS["secondary_random"]),
        (agg_secondary_qual, "Secondary: Qual-Top", MODE_COLORS["secondary_quality"]),
    ]

    x = np.arange(len(shop_types))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _base_style(fig)
    fig.suptitle("Mode Comparison: Shop Usage and Days Late", fontsize=18, fontweight="bold", color="#0B2A44")

    ax = axes[0]
    for idx, (agg, label, color) in enumerate(modes):
        vals = [agg.get("agg_shop_usage_counts", {}).get(st, 0.0) for st in shop_types]
        offset = (idx - 1) * width
        ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(shop_types)
    ax.set_title("Average Shop Assignments per Run\n(total assignments for one single-allocation job sums to 1)", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Shop Type", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Assignments", fontsize=13, color="#0B2A44")
    ax.legend(fontsize=10, frameon=False)
    _style_axis(ax, grid_y=True)

    ax = axes[1]
    mode_labels = [label for _, label, _ in modes]
    mode_colors = [color for _, _, color in modes]
    values = [agg.get("agg_avg_days_late", 0.0) for agg, _, _ in modes]
    bars = ax.bar(mode_labels, values, color=mode_colors, edgecolor="white")
    ax.set_title("Average Days Late by Mode\n(per job type per run, then averaged across runs)", fontsize=14, color="#0B2A44")
    ax.set_xlabel("Mode", fontsize=13, color="#0B2A44")
    ax.set_ylabel("Days Late", fontsize=13, color="#0B2A44")
    _style_axis(ax, grid_y=True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01 + 0.01,
                f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    _apply_text_overrides(fig, text_overrides)
    plt.tight_layout()
    fname = out / "mode_operational_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
