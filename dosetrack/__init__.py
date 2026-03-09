"""
DoseTrack V3 — PK/PD Simulation Engine

A pharmacokinetic/pharmacodynamic simulation engine for lisdexamfetamine
(Vyvanse) and d-amphetamine. Models prodrug conversion, multi-compartment
distribution, Michaelis-Menten elimination, dopamine vesicle depletion,
receptor tolerance, and sleep-debt modulation.

Quick start:
    from dosetrack import simulate, Dose, summarize

    result = simulate(
        doses=[Dose(time_h=8.0, amount_mg=50.0)],
        weight_kg=70.0,
    )
    print(summarize(result, current_time_h=12.0))
"""

from .dosing import Dose, SleepLog, DoseSchedule
from .simulation import simulate, simulate_multi_day, SimResult, MultiDayResult
from .outputs import (
    time_to_onset,
    functional_duration,
    duration_remaining,
    compute_crash_risk,
    compute_risk_flags,
    analyze_redose,
    estimate_recovery,
    summarize,
    CrashRisk,
    RiskFlags,
    RedoseAnalysis,
    RecoveryEstimate,
    UserSummary,
)
from .plotting import (
    plot_effect_curve,
    plot_pk_curve,
    plot_da_stores_single,
    plot_multi_day_effect,
    plot_da_recovery,
    plot_redose_comparison,
    plot_recovery_timeline,
    plot_dashboard,
)
from .pk_models import (
    bateman_conc,
    make_prodrug,
    make_1cmt_linear,
    make_1cmt_mm,
    make_2cmt_mm,
)
from .pd_models import EmaxParams, DopamineParams, ToleranceParams, SleepParams

__version__ = "3.0.0"
