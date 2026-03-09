"""
Simulation runner — the orchestrator.

Wires the RK4 solver, PK models, PD models, dosing, and dopamine depletion
into a single ODE system and integrates it. Handles dose event injection
(bolus additions mid-integration) and multi-day chaining.

The combined ODE system for the full prodrug model has 6 state variables:
  [A_gut_ldx, A_prodrug_blood, A_central_damp, A_peripheral_damp, DA_stores, T_acute]

Each model variant uses a subset of these states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .solver import StateVec, integrate, rk4_step
from .dosing import Dose, DoseSchedule, SleepLog
from .pk_models import (
    ProdrugParams, TwoCmtMMParams, OneCmtMMParams, OneCmtLinearParams,
    make_prodrug, make_2cmt_mm, make_1cmt_mm, make_1cmt_linear,
    ode_prodrug, ode_2cmt_mm, ode_1cmt_mm, ode_1cmt_linear,
    MW_RATIO, F_ORAL,
)
from .pd_models import (
    EmaxParams, DopamineParams, ToleranceParams, SleepParams,
    sigmoid_emax, central_activation, peripheral_activation,
    da_store_derivatives, da_recovery_analytic,
    tolerance_derivatives, chronic_tolerance, adjusted_ec50,
    sleep_debt_index, sleep_penalty, classify_zone,
)


# ── Result container ─────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Output of a simulation run. All arrays are time-aligned."""
    t: np.ndarray                          # hours from simulation start
    plasma_conc: np.ndarray                # ng/mL (d-AMP)
    effect_pct: np.ndarray                 # 0–100, central activation
    central_activation: np.ndarray         # 0–100
    peripheral_activation: np.ndarray      # 0–100
    da_stores: np.ndarray                  # 0–1
    tolerance_acute: np.ndarray            # 0–0.35
    tolerance_chronic: float               # constant for this window
    zones: list[str]                       # per-timepoint zone label
    doses: list[Dose]                      # doses included in this simulation
    weight_kg: float
    sleep_penalty_factor: float            # applied sleep debt penalty
    raw_states: np.ndarray                 # full state matrix for debugging

    @property
    def peak_conc(self) -> float:
        return float(np.max(self.plasma_conc))

    @property
    def peak_effect(self) -> float:
        return float(np.max(self.effect_pct))

    @property
    def tmax_h(self) -> float:
        """Time of peak plasma concentration."""
        return float(self.t[np.argmax(self.plasma_conc)])

    @property
    def tmax_effect_h(self) -> float:
        """Time of peak effect."""
        return float(self.t[np.argmax(self.effect_pct)])


@dataclass
class MultiDayResult:
    """Output of a multi-day simulation."""
    days: list[SimResult]
    da_stores_trajectory: list[float]   # DA store level at end of each day
    cumulative_t: np.ndarray            # continuous time axis (hours)
    cumulative_effect: np.ndarray       # continuous effect curve
    cumulative_plasma: np.ndarray       # continuous plasma curve
    cumulative_da: np.ndarray           # continuous DA stores


# ── Combined ODE builder ─────────────────────────────────────────────────

def _make_combined_ode_prodrug(
    pk: ProdrugParams,
    emax: EmaxParams,
    da_p: DopamineParams,
    tol_p: ToleranceParams,
    chronic_tol: float,
    sleep_pen: float,
) -> Callable[[float, StateVec], StateVec]:
    """
    Build the combined f(t, y) for the full prodrug 2-cmt model.

    State: [A_gut, A_prodrug, A_central, A_peripheral, S, T_acute]

    Coupling: PK drives PD (plasma conc → effect), PD drives DA depletion
    and tolerance. DA and tolerance feed back into PD via effect modulation.
    This bidirectional coupling is why everything must be in one ODE system.
    """
    pk_ode = ode_prodrug(pk)

    def f(t: float, y: StateVec) -> StateVec:
        # PK derivatives (first 4 states)
        pk_dy = pk_ode(t, y[:4])

        # Current plasma concentration of d-AMP (ng/mL)
        C_ngml = (y[2] / pk.V1) * 1000.0

        S = y[4]          # DA stores
        T_acute = y[5]    # acute tolerance

        # Adjusted EC50 (tolerance shifts the curve right)
        ec50_adj = adjusted_ec50(emax.EC50, T_acute, chronic_tol, tol_p)

        # Raw receptor occupancy (before DA modulation)
        E_raw = sigmoid_emax(C_ngml, ec50_adj, emax.gamma)

        # Effective subjective effect (DA-modulated)
        E_eff = E_raw * (S ** da_p.alpha) * sleep_pen

        # DA store dynamics
        dS = da_store_derivatives(S, E_eff, da_p)

        # Acute tolerance dynamics
        dT = tolerance_derivatives(T_acute, E_eff, tol_p)

        return np.array([pk_dy[0], pk_dy[1], pk_dy[2], pk_dy[3], dS, dT])

    return f


def _make_combined_ode_2cmt(
    pk: TwoCmtMMParams,
    emax: EmaxParams,
    da_p: DopamineParams,
    tol_p: ToleranceParams,
    chronic_tol: float,
    sleep_pen: float,
) -> Callable[[float, StateVec], StateVec]:
    """Combined ODE for 2-compartment (direct drug, no prodrug conversion)."""
    pk_ode = ode_2cmt_mm(pk)

    def f(t: float, y: StateVec) -> StateVec:
        pk_dy = pk_ode(t, y[:3])
        C_ngml = (y[1] / pk.V1) * 1000.0  # A_central / V1
        S, T_acute = y[3], y[4]
        ec50_adj = adjusted_ec50(emax.EC50, T_acute, chronic_tol, tol_p)
        E_raw = sigmoid_emax(C_ngml, ec50_adj, emax.gamma)
        E_eff = E_raw * (S ** da_p.alpha) * sleep_pen
        dS = da_store_derivatives(S, E_eff, da_p)
        dT = tolerance_derivatives(T_acute, E_eff, tol_p)
        return np.array([pk_dy[0], pk_dy[1], pk_dy[2], dS, dT])

    return f


def _make_combined_ode_1cmt(
    pk: OneCmtMMParams | OneCmtLinearParams,
    emax: EmaxParams,
    da_p: DopamineParams,
    tol_p: ToleranceParams,
    chronic_tol: float,
    sleep_pen: float,
) -> Callable[[float, StateVec], StateVec]:
    """Combined ODE for 1-compartment models."""
    if isinstance(pk, OneCmtLinearParams):
        pk_ode = ode_1cmt_linear(pk)
        Vd = pk.Vd
    else:
        pk_ode = ode_1cmt_mm(pk)
        Vd = pk.Vd

    def f(t: float, y: StateVec) -> StateVec:
        pk_dy = pk_ode(t, y[:2])
        C_ngml = (y[1] / Vd) * 1000.0
        S, T_acute = y[2], y[3]
        ec50_adj = adjusted_ec50(emax.EC50, T_acute, chronic_tol, tol_p)
        E_raw = sigmoid_emax(C_ngml, ec50_adj, emax.gamma)
        E_eff = E_raw * (S ** da_p.alpha) * sleep_pen
        dS = da_store_derivatives(S, E_eff, da_p)
        dT = tolerance_derivatives(T_acute, E_eff, tol_p)
        return np.array([pk_dy[0], pk_dy[1], dS, dT])

    return f


# ── State vector layout helpers ──────────────────────────────────────────

_LAYOUTS = {
    "prodrug":  {"pk_states": 4, "S_idx": 4, "T_idx": 5, "C_idx": 2, "V_attr": "V1"},
    "2cmt":     {"pk_states": 3, "S_idx": 3, "T_idx": 4, "C_idx": 1, "V_attr": "V1"},
    "1cmt_mm":  {"pk_states": 2, "S_idx": 2, "T_idx": 3, "C_idx": 1, "V_attr": "Vd"},
    "1cmt":     {"pk_states": 2, "S_idx": 2, "T_idx": 3, "C_idx": 1, "V_attr": "Vd"},
}


# ── Main simulation function ─────────────────────────────────────────────

def simulate(
    doses: list[Dose],
    weight_kg: float = 70.0,
    model: str = "prodrug",
    t_span: tuple[float, float] = (0.0, 24.0),
    dt: float = 0.01,
    dt_output: float = 0.05,
    sleep_logs: list[SleepLog] | None = None,
    dose_history_7d: list[float] | None = None,
    initial_da_stores: float = 1.0,
    initial_tolerance: float = 0.0,
    emax_params: EmaxParams | None = None,
    da_params: DopamineParams | None = None,
    tol_params: ToleranceParams | None = None,
    sleep_params: SleepParams | None = None,
) -> SimResult:
    """
    Run a full PK/PD simulation.

    Parameters
    ----------
    doses : list of Dose events (time_h relative to t_span start)
    weight_kg : patient body weight
    model : "prodrug" | "2cmt" | "1cmt_mm" | "1cmt" (linear, for validation)
    t_span : (start_h, end_h) simulation window
    dt : RK4 internal step size (hours)
    dt_output : output time resolution (hours)
    sleep_logs : recent sleep data for sleep-debt calculation
    dose_history_7d : daily mg totals for past 7 days (chronic tolerance)
    initial_da_stores : starting DA store level (0–1), for multi-day chaining
    initial_tolerance : starting acute tolerance level
    emax_params, da_params, tol_params, sleep_params : override defaults
    """
    emax = emax_params or EmaxParams()
    da_p = da_params or DopamineParams()
    tol_p = tol_params or ToleranceParams()
    slp_p = sleep_params or SleepParams()

    # Precompute chronic tolerance and sleep penalty (constant for this window)
    chronic_tol = chronic_tolerance(dose_history_7d or [], tol_p)
    sdi = sleep_debt_index(sleep_logs or [], slp_p)
    sleep_pen = sleep_penalty(sdi, slp_p)

    # For non-prodrug models, convert LDX doses to d-AMP equivalent.
    # The prodrug model handles conversion internally via MM kinetics;
    # direct models receive the active metabolite mass directly.
    if model != "prodrug":
        sim_doses = [
            Dose(time_h=d.time_h, amount_mg=d.amount_mg * MW_RATIO,
                 with_food=d.with_food, label=d.label)
            for d in doses
        ]
    else:
        sim_doses = list(doses)

    schedule = DoseSchedule(sorted(sim_doses, key=lambda d: d.time_h))
    layout = _LAYOUTS[model]

    if model == "prodrug":
        pk = make_prodrug(weight_kg)
        combined_ode = _make_combined_ode_prodrug(pk, emax, da_p, tol_p, chronic_tol, sleep_pen)
        y0 = np.zeros(6)
        V_central = pk.V1
    elif model == "2cmt":
        pk = make_2cmt_mm(weight_kg)
        combined_ode = _make_combined_ode_2cmt(pk, emax, da_p, tol_p, chronic_tol, sleep_pen)
        y0 = np.zeros(5)
        V_central = pk.V1
    elif model == "1cmt_mm":
        pk = make_1cmt_mm(weight_kg)
        combined_ode = _make_combined_ode_1cmt(pk, emax, da_p, tol_p, chronic_tol, sleep_pen)
        y0 = np.zeros(4)
        V_central = pk.Vd
    else:  # "1cmt" linear
        pk = make_1cmt_linear(weight_kg)
        combined_ode = _make_combined_ode_1cmt(pk, emax, da_p, tol_p, chronic_tol, sleep_pen)
        y0 = np.zeros(4)
        V_central = pk.Vd

    # Set initial PD states
    y0[layout["S_idx"]] = initial_da_stores
    y0[layout["T_idx"]] = initial_tolerance

    # Integrate with dose event injection
    t_out, y_out = _integrate_with_doses(
        combined_ode, y0, t_span, schedule, layout, dt, dt_output,
    )

    # Extract output arrays
    S_idx = layout["S_idx"]
    T_idx = layout["T_idx"]
    C_idx = layout["C_idx"]

    plasma_conc = (y_out[:, C_idx] / V_central) * 1000.0  # ng/mL
    np.maximum(plasma_conc, 0.0, out=plasma_conc)
    da_stores = y_out[:, S_idx]
    tol_acute = y_out[:, T_idx]

    # Compute PD outputs at each timepoint
    n = len(t_out)
    effect_pct = np.empty(n)
    ca = np.empty(n)
    pa = np.empty(n)
    zones = []

    for i in range(n):
        ec50_adj = adjusted_ec50(emax.EC50, tol_acute[i], chronic_tol, tol_p)
        ca[i] = central_activation(
            plasma_conc[i], ec50_adj, emax, da_stores[i], da_p, sleep_pen,
        )
        pa[i] = peripheral_activation(plasma_conc[i], ec50_adj, emax)
        effect_pct[i] = ca[i]
        zones.append(classify_zone(effect_pct[i]))

    return SimResult(
        t=t_out,
        plasma_conc=plasma_conc,
        effect_pct=effect_pct,
        central_activation=ca,
        peripheral_activation=pa,
        da_stores=da_stores,
        tolerance_acute=tol_acute,
        tolerance_chronic=chronic_tol,
        zones=zones,
        doses=doses,
        weight_kg=weight_kg,
        sleep_penalty_factor=sleep_pen,
        raw_states=y_out,
    )


def _integrate_with_doses(
    f: Callable[[float, StateVec], StateVec],
    y0: StateVec,
    t_span: tuple[float, float],
    schedule: DoseSchedule,
    layout: dict,
    dt: float,
    dt_output: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RK4 integration with bolus dose injection at event times.

    When a dose falls within an integration step, we:
    1. Integrate up to the dose time
    2. Add the dose mass to A_gut (state index 0)
    3. Continue integrating from there

    This preserves continuity of all other states while correctly modeling
    the discontinuous addition of drug to the GI tract.
    """
    t_start, t_end = t_span
    dose_times = sorted(set(d.time_h for d in schedule.doses if t_start <= d.time_h <= t_end))

    # Build segments: [t_start, dose1, dose2, ..., t_end]
    boundaries = [t_start] + [t for t in dose_times if t > t_start] + [t_end]
    boundaries = sorted(set(boundaries))

    all_t = []
    all_y = []
    y_current = y0.copy()

    for seg_idx in range(len(boundaries) - 1):
        seg_start = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]

        # Inject any doses at the segment start
        for dose in schedule.in_window(seg_start - 1e-9, seg_start + 1e-9):
            y_current[0] += dose.amount_mg  # bolus into A_gut

            # Apply food effect by adjusting ka if needed
            # (handled at parameter construction level, not here)

        if seg_end - seg_start < 1e-12:
            continue

        t_seg, y_seg = integrate(f, y_current, (seg_start, seg_end), dt=dt, dt_output=dt_output)

        # Avoid duplicating the boundary point
        if all_t and len(t_seg) > 0 and abs(t_seg[0] - all_t[-1][-1]) < 1e-9:
            t_seg = t_seg[1:]
            y_seg = y_seg[1:]

        if len(t_seg) > 0:
            all_t.append(t_seg)
            all_y.append(y_seg)
            y_current = y_seg[-1].copy()

    t_out = np.concatenate(all_t)
    y_out = np.concatenate(all_y, axis=0)
    return t_out, y_out


# ── Multi-day simulation ─────────────────────────────────────────────────

def simulate_multi_day(
    daily_doses: list[list[Dose]],
    weight_kg: float = 70.0,
    model: str = "prodrug",
    sleep_logs: list[SleepLog] | None = None,
    dose_history_7d: list[float] | None = None,
    overnight_hours: float = 8.0,
    **kwargs,
) -> MultiDayResult:
    """
    Chain multiple daily simulations, carrying DA stores and tolerance across days.

    Parameters
    ----------
    daily_doses : list of dose lists, one per day. Dose times are 0-24h within each day.
    overnight_hours : hours of recovery between simulation days (E=0, pure DA synthesis)
    """
    da_p = kwargs.get("da_params", DopamineParams())
    tol_p = kwargs.get("tol_params", ToleranceParams())

    da_stores = 1.0
    acute_tol = 0.0
    day_results = []
    da_trajectory = []
    dose_hist = list(dose_history_7d or [])

    for day_idx, day_doses in enumerate(daily_doses):
        # Build sleep logs for this day's context
        day_sleep = [s for s in (sleep_logs or []) if s.day <= day_idx]

        result = simulate(
            doses=day_doses,
            weight_kg=weight_kg,
            model=model,
            t_span=(0.0, 24.0),
            initial_da_stores=da_stores,
            initial_tolerance=acute_tol,
            dose_history_7d=dose_hist,
            sleep_logs=day_sleep,
            **{k: v for k, v in kwargs.items() if k not in ("dose_history_7d", "sleep_logs")},
        )

        day_results.append(result)

        # End-of-day states
        da_stores = float(result.da_stores[-1])
        acute_tol = float(result.tolerance_acute[-1])

        # Overnight recovery (no drug effect, pure DA synthesis)
        da_stores = da_recovery_analytic(da_stores, overnight_hours, da_p.k_synth)
        # Acute tolerance decays overnight
        acute_tol *= math.exp(-tol_p.k_off * overnight_hours)

        da_trajectory.append(da_stores)

        # Update dose history for chronic tolerance
        day_total = sum(d.amount_mg for d in day_doses) * MW_RATIO * F_ORAL
        dose_hist.append(day_total)
        if len(dose_hist) > 7:
            dose_hist = dose_hist[-7:]

    # Build continuous arrays
    all_t = []
    all_effect = []
    all_plasma = []
    all_da = []
    offset = 0.0

    for r in day_results:
        all_t.append(r.t + offset)
        all_effect.append(r.effect_pct)
        all_plasma.append(r.plasma_conc)
        all_da.append(r.da_stores)
        offset += 24.0

    return MultiDayResult(
        days=day_results,
        da_stores_trajectory=da_trajectory,
        cumulative_t=np.concatenate(all_t),
        cumulative_effect=np.concatenate(all_effect),
        cumulative_plasma=np.concatenate(all_plasma),
        cumulative_da=np.concatenate(all_da),
    )
