"""
Matplotlib visualizations for DoseTrack simulations.

Produces clean, publication-quality figures. Traffic-light color scheme
for zones. Designed to be glanceable — the visual should answer the
question faster than reading numbers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .simulation import SimResult, MultiDayResult
from .dosing import Dose
from .outputs import RedoseAnalysis, RecoveryEstimate
from .pd_models import ZONE_COLORS


# ── Style defaults (dark teal theme — matches dis-solved.com) ────────────

_BG_COLOR = "#0a1120"
_FACE_COLOR = "#0a1120"
_TEXT_COLOR = "#e2e8f0"
_GRID_COLOR = "#162032"
_SPINE_COLOR = "#1e3a3a"

_ZONE_BANDS = [
    (0, 15, ZONE_COLORS["baseline"], "Baseline"),
    (15, 40, ZONE_COLORS["subtherapeutic"], "Sub-therapeutic"),
    (40, 65, ZONE_COLORS["therapeutic"], "Therapeutic"),
    (65, 85, ZONE_COLORS["peak"], "Peak"),
    (85, 100, ZONE_COLORS["supratherapeutic"], "Supra-therapeutic"),
]

_CA_COLOR = "#5eead4"   # accent cyan (dis-solved)
_PA_COLOR = "#fb923c"   # warm orange
_DA_COLOR = "#14b8a6"   # secondary teal (dis-solved)
_PLASMA_COLOR = "#38bdf8"  # sky blue


def _apply_style(ax: Axes, xlabel: str = "Time (h)", ylabel: str = "") -> None:
    """Consistent axis styling — dark theme."""
    ax.set_facecolor(_FACE_COLOR)
    ax.figure.set_facecolor(_BG_COLOR)
    ax.set_xlabel(xlabel, fontsize=11, color="#94a3b8", fontweight=500)
    ax.set_ylabel(ylabel, fontsize=11, color="#94a3b8", fontweight=500)
    ax.tick_params(labelsize=10, colors="#64748b", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.grid(True, alpha=0.08, linewidth=0.5, color="#1e293b")


def _add_zone_bands(ax: Axes, ymax: float = 100.0) -> None:
    """Add semi-transparent zone background bands (subtle on dark bg)."""
    for lo, hi, color, _ in _ZONE_BANDS:
        ax.axhspan(lo, min(hi, ymax), color=color, alpha=0.05)


def _add_dose_markers(ax: Axes, doses: list[Dose], ymax: float = 100.0) -> None:
    """Vertical lines at each dose time."""
    for d in doses:
        ax.axvline(d.time_h, color="#475569", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(d.time_h, ymax * 0.95, f"{d.amount_mg:.0f}mg",
                fontsize=7, ha="center", va="top", color="#94a3b8")


# ── Single-day plots ─────────────────────────────────────────────────────

def plot_effect_curve(result: SimResult, ax: Axes | None = None) -> Figure:
    """
    Central + peripheral activation over time with zone bands.

    The primary visualization: shows what the user will actually feel
    and how it changes over the day.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    _add_zone_bands(ax)
    _add_dose_markers(ax, result.doses)

    ax.fill_between(result.t, result.central_activation, alpha=0.15, color=_CA_COLOR, label="Cognitive")
    ax.plot(result.t, result.central_activation, color=_CA_COLOR, linewidth=1.8)
    ax.plot(result.t, result.peripheral_activation, color=_PA_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.6, label="Peripheral")

    ax.set_ylim(-2, 108)
    ax.set_xlim(result.t[0], result.t[-1])
    _apply_style(ax, ylabel="Effect (%)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.2, facecolor=_FACE_COLOR,
              edgecolor="none", labelcolor=_TEXT_COLOR)
    ax.set_title("Effect Curve", fontsize=14, fontweight="bold", color=_TEXT_COLOR, pad=14)

    return fig or ax.get_figure()


def plot_pk_curve(result: SimResult, ax: Axes | None = None) -> Figure:
    """Plasma concentration (ng/mL) over time."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    _add_dose_markers(ax, result.doses, ymax=result.peak_conc * 1.1)

    ax.fill_between(result.t, result.plasma_conc, alpha=0.12, color=_PLASMA_COLOR)
    ax.plot(result.t, result.plasma_conc, color=_PLASMA_COLOR, linewidth=1.8)

    ax.set_xlim(result.t[0], result.t[-1])
    _apply_style(ax, ylabel="d-AMP Plasma Conc (ng/mL)")
    ax.set_title("Pharmacokinetic Curve", fontsize=14, fontweight="bold", color=_TEXT_COLOR, pad=14)

    # Annotate peak
    ax.annotate(
        f"Cmax = {result.peak_conc:.1f} ng/mL\nTmax = {result.tmax_h:.1f}h",
        xy=(result.tmax_h, result.peak_conc),
        xytext=(result.tmax_h + 2, result.peak_conc * 0.85),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="#94a3b8"),
        color="#94a3b8",
    )

    return fig or ax.get_figure()


def plot_da_stores_single(result: SimResult, ax: Axes | None = None) -> Figure:
    """DA store depletion over a single simulation."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    ax.fill_between(result.t, result.da_stores * 100, alpha=0.12, color=_DA_COLOR)
    ax.plot(result.t, result.da_stores * 100, color=_DA_COLOR, linewidth=1.8)

    ax.set_ylim(-2, 108)
    ax.set_xlim(result.t[0], result.t[-1])
    ax.axhline(80, color="#fbbf24", linewidth=0.8, linestyle=":", alpha=0.3, label="80% threshold")
    _apply_style(ax, ylabel="DA Stores (%)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.2, facecolor=_FACE_COLOR,
              edgecolor="none", labelcolor=_TEXT_COLOR)
    ax.set_title("Dopamine Vesicle Stores", fontsize=14, fontweight="bold", color=_TEXT_COLOR, pad=14)

    return fig or ax.get_figure()


# ── Multi-day plots ──────────────────────────────────────────────────────

def plot_multi_day_effect(multi: MultiDayResult) -> Figure:
    """Continuous effect curve across multiple days."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})

    ax_eff, ax_da = axes

    # Effect curve
    _add_zone_bands(ax_eff)
    ax_eff.plot(multi.cumulative_t / 24, multi.cumulative_effect,
                color=_CA_COLOR, linewidth=1.5)
    ax_eff.fill_between(multi.cumulative_t / 24, multi.cumulative_effect,
                        alpha=0.15, color=_CA_COLOR)
    ax_eff.set_ylim(-2, 108)
    ax_eff.set_ylabel("Effect (%)", fontsize=11, color="#94a3b8", fontweight=500)
    ax_eff.set_title("Multi-Day Effect Profile", fontsize=14, fontweight="bold",
                     color=_TEXT_COLOR, pad=16)

    # Add day separators
    for d in range(len(multi.days)):
        ax_eff.axvline(d, color="#1e293b", linewidth=0.5, linestyle=":")
        ax_eff.text(d + 0.5, 103, f"Day {d+1}", ha="center", fontsize=8,
                    color="#475569", fontweight=500)
        ax_da.axvline(d, color="#1e293b", linewidth=0.5, linestyle=":")

    # DA stores
    ax_da.plot(multi.cumulative_t / 24, multi.cumulative_da * 100,
               color=_DA_COLOR, linewidth=1.5)
    ax_da.fill_between(multi.cumulative_t / 24, multi.cumulative_da * 100,
                       alpha=0.1, color=_DA_COLOR)
    ax_da.axhline(80, color="#fbbf24", linewidth=0.8, linestyle=":", alpha=0.3)
    ax_da.set_ylim(-2, 108)
    ax_da.set_ylabel("DA Stores (%)", fontsize=11, color="#94a3b8", fontweight=500)
    ax_da.set_xlabel("Days", fontsize=11, color="#94a3b8", fontweight=500)

    for ax in axes:
        ax.set_facecolor(_FACE_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(_SPINE_COLOR)
        ax.spines["left"].set_color(_SPINE_COLOR)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(True, alpha=0.06, linewidth=0.5, color="#1e293b")
        ax.tick_params(labelsize=10, colors="#64748b", length=0)

    fig.set_facecolor(_BG_COLOR)
    fig.subplots_adjust(hspace=0.12, left=0.08, right=0.96, top=0.92, bottom=0.10)
    return fig


def plot_da_recovery(multi: MultiDayResult) -> Figure:
    """DA store levels at the end of each day (after overnight recovery)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    days = list(range(1, len(multi.da_stores_trajectory) + 1))
    stores = [s * 100 for s in multi.da_stores_trajectory]

    colors = []
    for s in stores:
        if s >= 80:
            colors.append("#14b8a6")
        elif s >= 60:
            colors.append("#fbbf24")
        else:
            colors.append("#f87171")

    ax.bar(days, stores, color=colors, alpha=0.75, edgecolor=_FACE_COLOR, linewidth=0.5,
           width=0.65, zorder=3)
    ax.axhline(80, color="#fbbf24", linewidth=1, linestyle="--", alpha=0.3, label="80% threshold")

    ax.set_ylim(0, 108)
    ax.set_title("Cumulative Dopamine Depletion", fontsize=14, fontweight="bold",
                 color=_TEXT_COLOR, pad=14)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=_FACE_COLOR,
              edgecolor="none", labelcolor=_TEXT_COLOR)
    _apply_style(ax, xlabel="Day", ylabel="DA Stores (%)")

    return fig


# ── Redose comparison ────────────────────────────────────────────────────

def plot_redose_comparison(
    baseline: SimResult,
    augmented: SimResult,
    proposed_time_h: float,
) -> Figure:
    """Overlay showing the marginal gain (or lack thereof) from redosing."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _add_zone_bands(ax)

    ax.plot(baseline.t, baseline.effect_pct, color=_CA_COLOR, linewidth=1.8,
            label="Without redose", alpha=0.7)
    ax.plot(augmented.t, augmented.effect_pct, color="#6ee7b7", linewidth=1.8,
            label="With redose", linestyle="--")

    ax.axvline(proposed_time_h, color="#f87171", linewidth=1, linestyle=":",
               label=f"Redose at {proposed_time_h:.1f}h")

    # Shade the marginal gain
    t_common = np.union1d(baseline.t, augmented.t)
    base_interp = np.interp(t_common, baseline.t, baseline.effect_pct)
    aug_interp = np.interp(t_common, augmented.t, augmented.effect_pct)
    mask = t_common >= proposed_time_h
    ax.fill_between(t_common[mask], base_interp[mask], aug_interp[mask],
                    alpha=0.12, color="#14b8a6", label="Marginal gain")

    ax.set_ylim(-2, 108)
    ax.set_xlim(baseline.t[0], baseline.t[-1])
    _apply_style(ax, ylabel="Effect (%)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.2, facecolor=_FACE_COLOR,
              edgecolor="none", labelcolor=_TEXT_COLOR)
    ax.set_title("Redose Utility Analysis", fontsize=14, fontweight="bold", color=_TEXT_COLOR, pad=14)

    return fig


# ── Recovery projection ──────────────────────────────────────────────────

def plot_recovery_timeline(
    current_stores: float,
    hours: float = 120.0,
    da_k_synth: float = 0.045,
    sleep_quality: float = 1.0,
) -> Figure:
    """Project DA store recovery over time."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    t = np.linspace(0, hours, 500)
    k_eff = da_k_synth * max(0.3, sleep_quality)
    S = 1.0 - (1.0 - current_stores) * np.exp(-k_eff * t)

    ax.plot(t / 24, S * 100, color=_DA_COLOR, linewidth=2)
    ax.fill_between(t / 24, S * 100, alpha=0.08, color=_DA_COLOR)

    for target, label in [(80, "80%"), (90, "90%"), (95, "95%")]:
        ax.axhline(target, color="#334155", linewidth=0.5, linestyle=":")
        ax.text(hours / 24 * 0.98, target + 1, label, fontsize=9,
                ha="right", color="#64748b", fontweight=500)

    ax.set_ylim(current_stores * 100 - 5, 103)
    ax.set_title(
        f"Recovery Timeline (from {current_stores:.0%})",
        fontsize=14, fontweight="bold", color=_TEXT_COLOR, pad=14,
    )
    _apply_style(ax, xlabel="Days", ylabel="DA Stores (%)")

    return fig


# ── Dashboard (multi-panel overview) ─────────────────────────────────────

def plot_dashboard(result: SimResult) -> Figure:
    """
    Four-panel overview: PK curve, effect curve, DA stores, zone timeline.
    Designed to give a complete picture at a glance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    (ax_pk, ax_eff), (ax_da, ax_zone) = axes

    # PK
    plot_pk_curve(result, ax=ax_pk)

    # Effect
    plot_effect_curve(result, ax=ax_eff)

    # DA
    plot_da_stores_single(result, ax=ax_da)

    # Zone timeline (horizontal bar)
    zone_map = {"baseline": 0, "subtherapeutic": 1, "therapeutic": 2, "peak": 3, "supratherapeutic": 4}
    zone_nums = [zone_map.get(z, 0) for z in result.zones]
    zone_colors_arr = [ZONE_COLORS.get(z, "#475569") for z in result.zones]

    ax_zone.scatter(result.t, zone_nums, c=zone_colors_arr, s=8, alpha=0.6)
    ax_zone.set_yticks(range(5))
    ax_zone.set_yticklabels(["Baseline", "Sub-ther.", "Therapeutic", "Peak", "Supra-ther."],
                            fontsize=8, color=_TEXT_COLOR)
    ax_zone.set_xlim(result.t[0], result.t[-1])
    _apply_style(ax_zone, ylabel="Zone")
    ax_zone.set_title("Therapeutic Zone", fontsize=12, fontweight="bold", color=_TEXT_COLOR)

    fig.set_facecolor(_BG_COLOR)
    fig.suptitle("DoseTrack Simulation Dashboard", fontsize=16, fontweight="bold",
                 y=1.02, color=_TEXT_COLOR)
    fig.tight_layout(pad=1.5)
    return fig
