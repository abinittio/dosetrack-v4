"""
Pharmacodynamic models: effect mapping, tolerance, dopamine depletion, sleep.

The PD layer converts plasma concentration into subjective effect. Three
mechanisms modulate this mapping:

1. Sigmoid Emax (Hill equation) — captures the fundamental nonlinearity of
   receptor binding. At high occupancy, additional drug barely moves the
   needle on felt effect even though plasma levels keep rising.

2. Dopamine vesicle depletion — amphetamines release DA from presynaptic
   vesicles. Repeated dosing depletes the releasable pool faster than
   tyrosine hydroxylase can resynthesize it. This is why day 5 of
   consecutive use feels different from day 1 at the same plasma level.

3. Tolerance (acute + chronic) — receptor desensitization shifts the
   dose-response curve rightward (EC50 increases), requiring higher
   concentrations for the same effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dosing import SleepLog


# ── PD parameters ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmaxParams:
    """Sigmoid Emax (Hill equation) parameters."""
    EC50: float = 30.0       # ng/mL, plasma concentration for 50% effect
    gamma: float = 1.5       # Hill coefficient (steepness)
    Emax: float = 100.0      # maximum possible effect (%)

    # Central vs peripheral activation split
    central_fraction: float = 0.65
    periph_fraction: float = 0.35
    periph_ec50_ratio: float = 0.7   # peripheral is more sensitive
    periph_gamma: float = 1.2


@dataclass(frozen=True)
class DopamineParams:
    """
    Vesicular dopamine store dynamics.

    k_synth: DA synthesis rate. At S=0, recovery half-life is ln(2)/k_synth ≈ 35h.
    Because synthesis slows as stores fill (TH product inhibition via the
    (1-S) term), effective half-life from S=0.5 to baseline is ~70h (3 days)
    — matching the clinical "2-3 day drug holiday" recommendation for
    meaningful recovery.

    k_release: depletion rate per unit of stimulant effect. Calibrated so that
    5 consecutive days of therapeutic use depletes stores to ~50-60% of
    baseline, producing a noticeable subjective effect reduction.

    alpha: exponent for how store level scales effect. 0.5 (square root) means
    partial depletion has a gentler impact than total depletion — the remaining
    DA is preferentially released from readily docked vesicles.
    """
    k_synth: float = 0.020     # h⁻¹ (slower recovery, t½ ≈ 35h at S=0)
    k_release: float = 0.06    # h⁻¹ per unit effect (stronger depletion)
    alpha: float = 0.5         # S^alpha scaling on subjective effect
    sleep_recovery_bonus: float = 1.3  # synthesis multiplier during good sleep


@dataclass(frozen=True)
class ToleranceParams:
    """
    Acute tolerance modeled as an ODE state variable.
    Chronic tolerance computed from a 7-day rolling window (precomputed).

    Acute: receptor desensitization builds during active dosing (proportional
    to effect intensity) and decays with a 4h half-life.

    Chronic: reflects sustained upregulation of homeostatic mechanisms over
    a 7-day window. Computed externally, not as an ODE, because it changes
    negligibly within a single-day simulation.
    """
    k_on: float = 0.05                  # tolerance induction rate (h⁻¹)
    k_off: float = math.log(2) / 4.0    # 4h half-life decay
    acute_max: float = 0.35             # maximum acute EC50 shift (35%)
    chronic_coeff: float = 0.004        # chronic shift per mg daily average
    chronic_max: float = 0.45           # maximum chronic EC50 shift (45%)
    max_total_shift: float = 0.60       # combined cap (60%)


@dataclass(frozen=True)
class SleepParams:
    """Sleep debt impact on cognitive activation."""
    ideal_h: float = 8.0
    debt_threshold_h: float = 6.0   # no penalty below this cumulative deficit
    max_penalty: float = 0.30       # up to 30% reduction in central activation
    window_days: int = 7


# ── Core PD functions ────────────────────────────────────────────────────

def sigmoid_emax(conc_ngml: float, ec50: float, gamma: float) -> float:
    """
    Fractional receptor occupancy (0–1) via Hill equation.

    This is the sigmoid Emax model: at low concentrations the relationship
    is approximately linear, but it saturates as occupancy approaches 1.0.
    At 70-80% occupancy, doubling the concentration barely changes the
    subjective effect — this is why dose escalation has diminishing returns.
    """
    if conc_ngml <= 0:
        return 0.0
    cg = conc_ngml ** gamma
    eg = ec50 ** gamma
    return cg / (eg + cg)


def central_activation(
    conc_ngml: float,
    ec50_adjusted: float,
    params: EmaxParams,
    da_stores: float,
    da_params: DopamineParams,
    sleep_penalty: float = 1.0,
) -> float:
    """
    Cognitive activation (0–100%).

    Modulated by three factors on top of raw receptor occupancy:
    - DA store level (depletion reduces available neurotransmitter)
    - Sleep debt (cognitive impairment from sleep loss)
    - Tolerance (EC50 shift, already baked into ec50_adjusted)

    Note: central_fraction is NOT applied as a scaling factor — the full
    sigmoid range (0–100%) represents cognitive effect. The fraction is
    metadata describing which pathway is responsible, not a limiter.
    """
    occupancy = sigmoid_emax(conc_ngml, ec50_adjusted, params.gamma)
    da_factor = da_stores ** da_params.alpha
    return params.Emax * occupancy * da_factor * sleep_penalty


def peripheral_activation(
    conc_ngml: float,
    ec50_adjusted: float,
    params: EmaxParams,
) -> float:
    """
    Sympathetic / cardiovascular activation (0–100%).

    Not modulated by DA stores or sleep — the peripheral nervous system
    responds to circulating amphetamine regardless of central DA state.
    Uses a lower EC50 (peripheral receptors are more sensitive) and
    shallower Hill coefficient.
    """
    periph_ec50 = ec50_adjusted * params.periph_ec50_ratio
    occupancy = sigmoid_emax(conc_ngml, periph_ec50, params.periph_gamma)
    return params.Emax * occupancy


# ── Dopamine depletion ODE terms ─────────────────────────────────────────

def da_store_derivatives(
    S: float,
    effect_fraction: float,
    params: DopamineParams,
) -> float:
    """
    dS/dt for vesicular DA stores.

    Synthesis: first-order approach to S=1, slowing as stores fill.
    This models TH product inhibition — dopamine itself inhibits its
    own synthesis via feedback on tyrosine hydroxylase.

    Depletion: proportional to both current effect and remaining stores.
    You can't release what you don't have.
    """
    synthesis = params.k_synth * (1.0 - S)
    release = params.k_release * effect_fraction * S
    return synthesis - release


def da_recovery_analytic(S0: float, hours: float, k_synth: float) -> float:
    """
    Analytical recovery when effect = 0 (e.g. overnight, drug holiday).
    S(t) = 1 - (1 - S0) * exp(-k_synth * t)
    """
    return 1.0 - (1.0 - S0) * math.exp(-k_synth * hours)


# ── Tolerance computation ────────────────────────────────────────────────

def tolerance_derivatives(
    T_acute: float,
    effect_fraction: float,
    params: ToleranceParams,
) -> float:
    """
    dT_acute/dt — acute tolerance builds with effect, decays at 4h half-life.

    Uses a ceiling term (T_max - T_acute) so tolerance cannot exceed the
    biological maximum: receptor desensitization has a finite capacity.
    """
    induction = params.k_on * effect_fraction * (params.acute_max - T_acute)
    decay = params.k_off * T_acute
    return induction - decay


def chronic_tolerance(
    daily_mg_history: list[float],
    params: ToleranceParams,
) -> float:
    """
    Chronic tolerance from 7-day rolling average daily dose.

    This is computed once before simulation, not as an ODE, because it
    operates on a much slower timescale. One new day's dosing changes the
    7-day average by at most 1/7th — negligible within a single simulation.
    """
    if not daily_mg_history:
        return 0.0
    recent = daily_mg_history[-7:]
    avg_daily = sum(recent) / len(recent)
    return min(params.chronic_max, params.chronic_coeff * avg_daily)


def adjusted_ec50(
    base_ec50: float,
    acute_tol: float,
    chronic_tol: float,
    params: ToleranceParams,
) -> float:
    """EC50 shifted rightward by combined tolerance. Capped at 60%."""
    combined = min(params.max_total_shift, acute_tol + chronic_tol)
    return base_ec50 * (1.0 + combined)


# ── Sleep debt ───────────────────────────────────────────────────────────

def sleep_debt_index(sleep_logs: list[SleepLog], params: SleepParams) -> float:
    """Cumulative sleep deficit (hours) over the lookback window."""
    recent = [s for s in sleep_logs if s.day >= -params.window_days]
    if not recent:
        return 0.0
    return sum(max(0.0, params.ideal_h - s.hours_slept) for s in recent)


def sleep_penalty(sdi: float, params: SleepParams) -> float:
    """
    Multiplicative penalty on central activation from sleep debt.

    No penalty up to 6h cumulative deficit (roughly one bad night).
    Linear ramp to 30% penalty at severe deficit (28h over 7 days).
    """
    if sdi <= params.debt_threshold_h:
        return 1.0
    fraction = min(1.0, (sdi - params.debt_threshold_h) / 22.0)
    return 1.0 - params.max_penalty * fraction


# ── Zone classification ──────────────────────────────────────────────────

ZONE_THRESHOLDS = [
    (85.0, "supratherapeutic"),
    (65.0, "peak"),
    (40.0, "therapeutic"),
    (15.0, "subtherapeutic"),
    (0.0,  "baseline"),
]

ZONE_COLORS = {
    "supratherapeutic": "#EF4444",
    "peak": "#059669",
    "therapeutic": "#10B981",
    "subtherapeutic": "#F59E0B",
    "baseline": "#94A3B8",
}


def classify_zone(effect_pct: float) -> str:
    """Map effect percentage to therapeutic zone label."""
    for threshold, label in ZONE_THRESHOLDS:
        if effect_pct >= threshold:
            return label
    return "baseline"
