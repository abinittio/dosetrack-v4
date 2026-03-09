"""
User-facing output analysis.

Takes SimResult objects and computes the answers a user actually needs:
  - Time to onset
  - Functional duration remaining
  - Redose utility (diminishing returns)
  - Recovery timeline
  - Crash risk
  - Risk flags
  - Human-readable summaries

Designed for someone with ADHD: clear answers, not walls of numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .simulation import SimResult, simulate
from .dosing import Dose, DoseSchedule, SleepLog
from .pd_models import (
    EmaxParams, DopamineParams, ToleranceParams, SleepParams,
    sigmoid_emax, classify_zone, da_recovery_analytic,
)


# ── Time to onset ────────────────────────────────────────────────────────

def time_to_onset(result: SimResult, threshold_pct: float = 15.0) -> float | None:
    """
    Hours from first dose to first crossing of the effect threshold.

    Returns None if threshold is never reached (e.g. very low dose).
    Accounts for dose-dependent Tmax shifts — higher prodrug doses take
    longer because enzymatic conversion is rate-limited.
    """
    if not result.doses:
        return None
    first_dose_t = min(d.time_h for d in result.doses)
    for i in range(len(result.t)):
        if result.t[i] >= first_dose_t and result.effect_pct[i] >= threshold_pct:
            return float(result.t[i] - first_dose_t)
    return None


# ── Functional duration ──────────────────────────────────────────────────

def functional_duration(
    result: SimResult,
    threshold_pct: float = 40.0,
    from_time: float | None = None,
) -> float:
    """
    Total hours of effect above threshold (therapeutic zone or above).

    If from_time is given, only counts duration after that time
    (useful for "how many hours of functional effect are LEFT").
    """
    dt = np.diff(result.t, prepend=result.t[0])
    mask = result.effect_pct >= threshold_pct
    if from_time is not None:
        mask &= result.t >= from_time
    return float(np.sum(dt[mask]))


def duration_remaining(result: SimResult, current_time: float, threshold_pct: float = 40.0) -> float:
    """Hours of above-threshold effect remaining from current_time."""
    return functional_duration(result, threshold_pct, from_time=current_time)


# ── Crash risk ───────────────────────────────────────────────────────────

@dataclass
class CrashRisk:
    """Multi-factor crash prediction."""
    level: str           # "low", "moderate", "high"
    score: float         # 0–1 composite
    factors: dict        # individual factor contributions

    @property
    def description(self) -> str:
        if self.level == "high":
            return "Significant crash likely — steep effect decline combined with other risk factors"
        elif self.level == "moderate":
            return "Moderate crash risk — effect is declining, consider winding down"
        return "Low crash risk"


def compute_crash_risk(
    result: SimResult,
    time_h: float,
    sleep_debt_h: float = 0.0,
    doses: list[Dose] | None = None,
) -> CrashRisk:
    """
    4-factor weighted crash risk at a specific time.

    Factors (matching V2 weights):
      40% decline rate — how fast effect is dropping
      20% sleep debt — accumulated deficit
      20% tolerance — higher tolerance = harder crash
      20% late dosing — afternoon doses that wear off at night
    """
    # Find closest time index
    idx = int(np.argmin(np.abs(result.t - time_h)))

    # Factor 1: decline rate
    decline = 0.0
    if idx > 0:
        dt = result.t[idx] - result.t[idx - 1]
        if dt > 0:
            rate = (result.effect_pct[idx - 1] - result.effect_pct[idx]) / dt  # %/h
            decline = min(1.0, max(0.0, (rate - 2.0) / 8.0))  # kicks in at >2%/h

    # Factor 2: sleep debt
    sleep_factor = min(1.0, max(0.0, (sleep_debt_h - 4.0) / 16.0))

    # Factor 3: tolerance
    tol_combined = result.tolerance_acute[idx] + result.tolerance_chronic
    tol_factor = min(1.0, tol_combined / 0.6)

    # Factor 4: late dosing (dose after 14h = 2pm, checked at >20h = 8pm)
    late_factor = 0.0
    if doses and time_h > 20.0:
        has_late = any(d.time_h > 14.0 for d in doses)
        if has_late:
            late_factor = 0.6

    score = 0.4 * decline + 0.2 * sleep_factor + 0.2 * tol_factor + 0.2 * late_factor
    level = "high" if score >= 0.6 else "moderate" if score >= 0.3 else "low"

    return CrashRisk(
        level=level,
        score=score,
        factors={
            "decline_rate": decline,
            "sleep_debt": sleep_factor,
            "tolerance": tol_factor,
            "late_dosing": late_factor,
        },
    )


# ── Risk flags ───────────────────────────────────────────────────────────

@dataclass
class RiskFlags:
    """Binary safety flags."""
    overstimulation: bool
    sleep_disruption: bool
    escalation: bool
    dose_stacking: bool

    @property
    def active(self) -> list[str]:
        flags = []
        if self.overstimulation:
            flags.append("Overstimulation — total daily dose exceeds safety threshold or effect >95%")
        if self.sleep_disruption:
            flags.append("Sleep disruption risk — dose taken after 2pm may affect tonight's sleep")
        if self.escalation:
            flags.append("Escalation pattern — this week's total is 20%+ higher than last week")
        if self.dose_stacking:
            flags.append("Dose stacking — multiple doses within 3 hours")
        return flags


def compute_risk_flags(
    result: SimResult,
    dose_history_7d: list[float] | None = None,
) -> RiskFlags:
    """Evaluate safety flags from simulation results."""
    doses = result.doses

    # Overstimulation: >70mg d-AMP equivalent or effect >95%
    from .pk_models import MW_RATIO, F_ORAL
    daily_damp = sum(d.amount_mg for d in doses) * MW_RATIO * F_ORAL
    overstim = daily_damp > 70.0 or float(np.max(result.effect_pct)) > 95.0

    # Sleep disruption: any dose after 2pm (14h)
    sleep_risk = any(d.time_h > 14.0 for d in doses)

    # Escalation: this week > last week × 1.2
    escalation = False
    if dose_history_7d and len(dose_history_7d) >= 7:
        this_week = sum(dose_history_7d[-7:])
        last_week = sum(dose_history_7d[-14:-7]) if len(dose_history_7d) >= 14 else 0
        if last_week > 0:
            escalation = this_week > last_week * 1.2

    # Dose stacking: any two doses <3h apart
    stacking = False
    times = sorted(d.time_h for d in doses)
    for i in range(1, len(times)):
        if times[i] - times[i - 1] < 3.0:
            stacking = True
            break

    return RiskFlags(
        overstimulation=overstim,
        sleep_disruption=sleep_risk,
        escalation=escalation,
        dose_stacking=stacking,
    )


# ── Redose utility ───────────────────────────────────────────────────────

@dataclass
class RedoseAnalysis:
    """
    Should the user take another dose? Quantifies diminishing returns.

    The sigmoid Emax curve means that at high receptor occupancy, additional
    drug barely increases subjective effect but DOES increase cardiovascular
    load (peripheral activation keeps climbing because its EC50 is lower).
    """
    current_effect_pct: float
    marginal_effect_increase: float      # additional CA %
    marginal_pa_increase: float          # additional peripheral %
    marginal_duration_hours: float       # extra hours above threshold
    diminishing_returns_pct: float       # % of fresh-dose effect you'd actually get
    risk_level: str                      # "beneficial", "marginal", "counterproductive"
    explanation: str


def analyze_redose(
    current_result: SimResult,
    proposed_dose_mg: float,
    proposed_time_h: float,
    weight_kg: float = 70.0,
    model: str = "prodrug",
    **kwargs,
) -> RedoseAnalysis:
    """
    Compare current trajectory with and without an additional dose.

    Runs two simulations:
    1. Current doses only → baseline trajectory
    2. Current doses + proposed dose → augmented trajectory

    Then computes marginal gains and rates the utility.
    """
    # Augmented simulation
    augmented_doses = list(current_result.doses) + [
        Dose(time_h=proposed_time_h, amount_mg=proposed_dose_mg)
    ]
    augmented = simulate(
        doses=augmented_doses,
        weight_kg=weight_kg,
        model=model,
        t_span=(current_result.t[0], current_result.t[-1]),
        initial_da_stores=float(current_result.da_stores[0]),
        initial_tolerance=float(current_result.tolerance_acute[0]),
        **kwargs,
    )

    # Reference: what would this dose do on a completely fresh system?
    reference = simulate(
        doses=[Dose(time_h=0.0, amount_mg=proposed_dose_mg)],
        weight_kg=weight_kg,
        model=model,
        t_span=(0.0, 24.0),
        **kwargs,
    )

    # Current effect at proposed time
    idx_now = int(np.argmin(np.abs(current_result.t - proposed_time_h)))
    current_effect = float(current_result.effect_pct[idx_now])
    current_pa = float(current_result.peripheral_activation[idx_now])

    # Peak effect comparison (after proposed time)
    mask = augmented.t >= proposed_time_h
    if np.any(mask):
        aug_peak_effect = float(np.max(augmented.effect_pct[mask]))
        aug_peak_pa = float(np.max(augmented.peripheral_activation[mask]))
    else:
        aug_peak_effect = current_effect
        aug_peak_pa = current_pa

    base_mask = current_result.t >= proposed_time_h
    if np.any(base_mask):
        base_peak_effect = float(np.max(current_result.effect_pct[base_mask]))
    else:
        base_peak_effect = current_effect

    marginal_effect = aug_peak_effect - base_peak_effect
    marginal_pa = aug_peak_pa - current_pa

    # Duration comparison
    base_dur = functional_duration(current_result, from_time=proposed_time_h)
    aug_dur = functional_duration(augmented, from_time=proposed_time_h)
    marginal_duration = aug_dur - base_dur

    # Diminishing returns: marginal gain / fresh dose peak
    ref_peak = float(np.max(reference.effect_pct))
    diminishing = (marginal_effect / ref_peak * 100.0) if ref_peak > 0 else 0.0

    # Risk classification
    if marginal_effect > 15.0 and diminishing > 40.0:
        risk = "beneficial"
        explanation = (
            f"This dose adds ~{marginal_effect:.0f}% to peak effect "
            f"and extends functional duration by ~{marginal_duration:.1f}h. "
            f"You're getting {diminishing:.0f}% of what a fresh dose would give."
        )
    elif marginal_effect > 5.0:
        risk = "marginal"
        explanation = (
            f"This top-up adds only ~{marginal_effect:.0f}% to subjective effect "
            f"but increases cardiovascular load by ~{marginal_pa:.0f}%. "
            f"Duration extends by ~{marginal_duration:.1f}h. "
            f"Diminishing returns: you're only getting {diminishing:.0f}% efficiency."
        )
    else:
        risk = "counterproductive"
        explanation = (
            f"At current receptor occupancy, this dose adds just ~{marginal_effect:.0f}% "
            f"to felt effect while adding significant peripheral activation (+{marginal_pa:.0f}%). "
            f"Your dopamine stores are at {current_result.da_stores[idx_now]:.0%} — "
            f"consider a rest period instead."
        )

    return RedoseAnalysis(
        current_effect_pct=current_effect,
        marginal_effect_increase=marginal_effect,
        marginal_pa_increase=marginal_pa,
        marginal_duration_hours=marginal_duration,
        diminishing_returns_pct=diminishing,
        risk_level=risk,
        explanation=explanation,
    )


# ── Recovery timeline ────────────────────────────────────────────────────

@dataclass
class RecoveryEstimate:
    """How long until dopamine stores recover to target levels."""
    current_da_stores: float
    hours_to_80pct: float
    hours_to_90pct: float
    hours_to_95pct: float
    sleep_modifier: float   # 1.0 = normal, <1.0 = poor sleep slows recovery
    explanation: str


def estimate_recovery(
    current_da_stores: float,
    sleep_quality: float = 1.0,
    da_params: DopamineParams | None = None,
) -> RecoveryEstimate:
    """
    Project DA store recovery timeline.

    Uses the analytical solution: S(t) = 1 - (1-S0) * exp(-k_synth * k_sleep * t)

    Poor sleep slows recovery by reducing the effective synthesis rate.
    Sleep quality (0–1) modifies k_synth multiplicatively.
    """
    da_p = da_params or DopamineParams()
    k_eff = da_p.k_synth * max(0.3, sleep_quality)  # floor at 30% synthesis rate

    def hours_to_target(target: float) -> float:
        if current_da_stores >= target:
            return 0.0
        # S(t) = 1 - (1-S0)*exp(-k*t) = target
        # exp(-k*t) = (1-target)/(1-S0)
        ratio = (1.0 - target) / (1.0 - current_da_stores)
        if ratio <= 0:
            return 0.0
        return -math.log(ratio) / k_eff

    h80 = hours_to_target(0.80)
    h90 = hours_to_target(0.90)
    h95 = hours_to_target(0.95)

    # Human-readable explanation
    if current_da_stores >= 0.90:
        explanation = "Your dopamine stores are near baseline — you're well recovered."
    elif current_da_stores >= 0.70:
        explanation = (
            f"Stores at {current_da_stores:.0%}. "
            f"Full recovery (~90%) in roughly {h90:.0f}h ({h90/24:.1f} days). "
            f"A dose today will feel close to normal."
        )
    elif current_da_stores >= 0.50:
        explanation = (
            f"Stores at {current_da_stores:.0%} — noticeably depleted. "
            f"A dose today will feel weaker than usual. "
            f"Consider {h80:.0f}h ({h80/24:.1f} days) rest to reach 80%. "
            f"Full recovery to 90% takes ~{h90:.0f}h ({h90/24:.1f} days)."
        )
    else:
        explanation = (
            f"Stores at {current_da_stores:.0%} — significantly depleted. "
            f"Dosing today will have substantially reduced effect. "
            f"Recommend at least {h80:.0f}h ({h80/24:.1f} days) off. "
            f"Sleep well — poor sleep slows DA recovery."
        )

    return RecoveryEstimate(
        current_da_stores=current_da_stores,
        hours_to_80pct=h80,
        hours_to_90pct=h90,
        hours_to_95pct=h95,
        sleep_modifier=sleep_quality,
        explanation=explanation,
    )


# ── Human-readable summary ───────────────────────────────────────────────

@dataclass
class UserSummary:
    """
    Glanceable answers — designed for someone with ADHD.
    No walls of numbers. Clear, direct statements.
    """
    onset: str
    duration: str
    current_zone: str
    da_status: str
    recovery: str
    risk_summary: str


def summarize(
    result: SimResult,
    current_time_h: float | None = None,
    sleep_logs: list[SleepLog] | None = None,
) -> UserSummary:
    """Generate a human-readable summary of the simulation state."""

    # Time to onset
    onset_h = time_to_onset(result)
    if onset_h is None:
        onset = "No active dose detected"
    elif onset_h < 0.5:
        onset = "You should be feeling it already"
    else:
        onset = f"You'll feel it in ~{onset_h * 60:.0f} min"

    # Functional duration
    t_now = current_time_h if current_time_h is not None else 0.0
    dur = duration_remaining(result, t_now)
    if dur < 0.5:
        duration = "Effect has mostly worn off"
    else:
        duration = f"~{dur:.1f}h of functional effect left"

    # Current zone
    if current_time_h is not None:
        idx = int(np.argmin(np.abs(result.t - current_time_h)))
        zone = result.zones[idx]
        effect = result.effect_pct[idx]
        current_zone = f"{zone.capitalize()} ({effect:.0f}%)"
    else:
        current_zone = f"Peak: {result.peak_effect:.0f}% at {result.tmax_effect_h:.1f}h"

    # DA status
    da = float(result.da_stores[-1]) if current_time_h is None else float(
        result.da_stores[int(np.argmin(np.abs(result.t - current_time_h)))]
    )
    if da >= 0.85:
        da_status = f"Dopamine stores at {da:.0%} — healthy"
    elif da >= 0.65:
        da_status = f"Dopamine stores at {da:.0%} — mildly depleted"
    elif da >= 0.50:
        da_status = f"Dopamine stores at {da:.0%} — consider a rest day"
    else:
        da_status = f"Dopamine stores at {da:.0%} — significantly depleted, rest recommended"

    # Recovery
    recovery_est = estimate_recovery(da)
    recovery = recovery_est.explanation

    # Risk summary
    flags = compute_risk_flags(result)
    active = flags.active
    if active:
        risk_summary = " | ".join(active)
    else:
        risk_summary = "No active risk flags"

    return UserSummary(
        onset=onset,
        duration=duration,
        current_zone=current_zone,
        da_status=da_status,
        recovery=recovery,
        risk_summary=risk_summary,
    )
