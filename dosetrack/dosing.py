"""
Dose schedule handling.

Manages a list of dose events and provides them to the simulation runner.
Doses are modeled as instantaneous bolus additions to the gut compartment
at the specified time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dose:
    """A single oral dose event."""
    time_h: float          # hours from simulation start (t=0)
    amount_mg: float       # mg of drug (LDX if prodrug model, d-AMP if direct)
    with_food: bool = False
    label: str = ""


@dataclass
class SleepLog:
    """One night of sleep data for sleep-debt calculation."""
    day: int               # day index (0 = simulation start day)
    hours_slept: float
    quality: int = 3       # 1-5 scale


@dataclass
class DoseSchedule:
    """
    Ordered collection of doses with query methods.

    Keeps doses sorted by time so the simulation runner can iterate through
    them efficiently during integration.
    """
    doses: list[Dose] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.doses.sort(key=lambda d: d.time_h)

    def add(self, dose: Dose) -> None:
        self.doses.append(dose)
        self.doses.sort(key=lambda d: d.time_h)

    def in_window(self, t_start: float, t_end: float) -> list[Dose]:
        """Return doses with time_h in [t_start, t_end)."""
        return [d for d in self.doses if t_start <= d.time_h < t_end]

    def total_mg(self, t_start: float = 0.0, t_end: float = float("inf")) -> float:
        """Sum of dose amounts in time window."""
        return sum(d.amount_mg for d in self.in_window(t_start, t_end))

    def daily_totals(self, n_days: int) -> list[float]:
        """Total mg per day for days 0..n_days-1 (24h blocks)."""
        return [self.total_mg(d * 24, (d + 1) * 24) for d in range(n_days)]
