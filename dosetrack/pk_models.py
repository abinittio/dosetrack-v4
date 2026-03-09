"""
Pharmacokinetic compartmental models.

Defines ODE right-hand-side functions for drug concentration dynamics.
Supports four model variants:
  - 1-compartment with linear (first-order) elimination
  - 1-compartment with Michaelis-Menten elimination
  - 2-compartment with Michaelis-Menten elimination
  - Prodrug (LDX→d-AMP) with 2-compartment MM elimination

Why Michaelis-Menten over first-order: CYP enzymes and renal transporters are
saturable proteins. At therapeutic doses the system approximates first-order,
but dose stacking or supratherapeutic doses cause enzyme saturation, slowing
clearance and prolonging exposure. MM captures this honestly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .solver import StateVec

# ── Published PK constants for lisdexamfetamine / d-amphetamine ──────────

KA_FASTING = 0.85       # h⁻¹, absorption rate (Tmax ≈ 3.8h fasting)
KA_FED = 0.50           # h⁻¹, absorption rate (Tmax ≈ 4.7h with food)
F_ORAL = 0.96           # oral bioavailability (Pennick 2010)
MW_RATIO = 0.5135       # d-AMP mass / LDX mass
KE = math.log(2) / 11   # h⁻¹, first-order elimination rate (t½ = 11h)
VD_PER_KG = 3.5         # L/kg, volume of distribution

# MM elimination calibration: at low C, Vmax/(Km·Vd) ≈ ke
# Km = 0.3 mg/L (300 ng/mL) → nonlinearity emerges at supratherapeutic range
KM_ELIM = 0.3           # mg/L

# Prodrug enzymatic conversion (RBC peptidases, saturable)
# Km_conv must be well above typical blood concentrations at normal doses
# so that conversion appears first-order. 50mg LDX in ~5L blood ≈ 10 mg/L,
# so Km_conv = 15 mg/L keeps us in the approximately linear regime for
# standard doses but shows saturation at abuse-range (>100mg).
KM_CONV = 15.0           # mg/L, substrate concentration at half-max conversion
V_BLOOD_PER_KG = 0.07    # L/kg, approximate blood volume for prodrug compartment

# 2-compartment distribution
V2_RATIO = 1.5          # V_peripheral / V_central
Q_RATIO = 0.5           # intercompartmental clearance / V_central (h⁻¹)


# ── Parameter dataclasses ────────────────────────────────────────────────

@dataclass(frozen=True)
class OneCmtLinearParams:
    """One-compartment, first-order elimination. Equivalent to Bateman model."""
    ka: float    # absorption rate (h⁻¹)
    ke: float    # elimination rate (h⁻¹)
    Vd: float    # volume of distribution (L)
    F: float     # bioavailability


@dataclass(frozen=True)
class OneCmtMMParams:
    """One-compartment, Michaelis-Menten elimination."""
    ka: float    # absorption rate (h⁻¹)
    Vd: float    # volume of distribution (L)
    Vmax: float  # maximum elimination rate (mg/h)
    Km: float    # Michaelis constant (mg/L)
    F: float     # bioavailability


@dataclass(frozen=True)
class TwoCmtMMParams:
    """Two-compartment, Michaelis-Menten elimination from central."""
    ka: float
    V1: float    # central volume (L)
    V2: float    # peripheral volume (L)
    Q: float     # intercompartmental clearance (L/h)
    Vmax: float
    Km: float
    F: float


@dataclass(frozen=True)
class ProdrugParams:
    """
    LDX prodrug → d-AMP active metabolite.

    The prodrug is absorbed into blood, then cleaved by RBC peptidases
    (saturable, Michaelis-Menten). The active metabolite (d-AMP) follows
    2-compartment kinetics with MM elimination.
    """
    ka: float          # GI absorption rate of LDX (h⁻¹)
    F: float           # bioavailability
    mw_ratio: float    # d-AMP mass / LDX mass
    V_blood: float     # blood volume for prodrug compartment (L)
    Vmax_conv: float   # max conversion rate (mg/h)
    Km_conv: float     # conversion MM constant (mg/L)
    V1: float          # central volume for d-AMP (L)
    V2: float          # peripheral volume for d-AMP (L)
    Q: float           # intercompartmental clearance (L/h)
    Vmax: float        # elimination Vmax for d-AMP (mg/h)
    Km: float          # elimination Km for d-AMP (mg/L)


# ── ODE constructors ─────────────────────────────────────────────────────

def ode_1cmt_linear(p: OneCmtLinearParams) -> Callable[[float, StateVec], StateVec]:
    """
    State: [A_gut, A_central] (mg).
    For validation against the analytical Bateman solution.
    """
    def f(t: float, y: StateVec) -> StateVec:
        A_gut, A_cen = y
        dA_gut = -p.ka * A_gut
        dA_cen = p.ka * A_gut * p.F - p.ke * A_cen
        return np.array([dA_gut, dA_cen])
    return f


def ode_1cmt_mm(p: OneCmtMMParams) -> Callable[[float, StateVec], StateVec]:
    """
    State: [A_gut, A_central] (mg).
    MM elimination: rate = Vmax·C/(Km+C) where C = A_central/Vd.
    """
    def f(t: float, y: StateVec) -> StateVec:
        A_gut, A_cen = y
        C = A_cen / p.Vd  # mg/L
        dA_gut = -p.ka * A_gut
        dA_cen = p.ka * A_gut * p.F - p.Vmax * C / (p.Km + C)
        return np.array([dA_gut, dA_cen])
    return f


def ode_2cmt_mm(p: TwoCmtMMParams) -> Callable[[float, StateVec], StateVec]:
    """
    State: [A_gut, A_central, A_peripheral] (mg).
    Central compartment has MM elimination + intercompartmental exchange.
    """
    def f(t: float, y: StateVec) -> StateVec:
        A_gut, A1, A2 = y
        C1 = A1 / p.V1
        C2 = A2 / p.V2
        dA_gut = -p.ka * A_gut
        dA1 = p.ka * A_gut * p.F - p.Vmax * C1 / (p.Km + C1) - p.Q * (C1 - C2)
        dA2 = p.Q * (C1 - C2)
        return np.array([dA_gut, dA1, dA2])
    return f


def ode_prodrug(p: ProdrugParams) -> Callable[[float, StateVec], StateVec]:
    """
    State: [A_gut_ldx, A_prodrug_blood, A_central_damp, A_peripheral_damp] (mg).

    LDX is absorbed from gut into blood, then enzymatically cleaved to d-AMP
    (saturable). d-AMP distributes into a 2-compartment system and is eliminated
    via MM kinetics.
    """
    def f(t: float, y: StateVec) -> StateVec:
        A_gut, A_pro, A1, A2 = y
        C_ldx = A_pro / p.V_blood          # prodrug blood concentration
        C1 = A1 / p.V1                     # d-AMP central concentration
        C2 = A2 / p.V2                     # d-AMP peripheral concentration

        # Absorption: LDX from gut to blood
        dA_gut = -p.ka * A_gut

        # Conversion: LDX → d-AMP (saturable, MM)
        conv_rate = p.Vmax_conv * C_ldx / (p.Km_conv + C_ldx + 1e-12)
        dA_pro = p.ka * A_gut * p.F - conv_rate

        # d-AMP appears in central compartment (mass-corrected by MW ratio)
        drug_input = conv_rate * p.mw_ratio
        elim_rate = p.Vmax * C1 / (p.Km + C1 + 1e-12)
        dA1 = drug_input - elim_rate - p.Q * (C1 - C2)
        dA2 = p.Q * (C1 - C2)

        return np.array([dA_gut, dA_pro, dA1, dA2])
    return f


# ── Parameter factories ──────────────────────────────────────────────────

def make_1cmt_linear(weight_kg: float, with_food: bool = False) -> OneCmtLinearParams:
    """Create 1-compartment linear params for d-amphetamine."""
    return OneCmtLinearParams(
        ka=KA_FED if with_food else KA_FASTING,
        ke=KE,
        Vd=VD_PER_KG * weight_kg,
        F=F_ORAL,
    )


def make_1cmt_mm(weight_kg: float, with_food: bool = False) -> OneCmtMMParams:
    """Create 1-compartment MM params. Vmax derived so low-C behavior matches ke."""
    Vd = VD_PER_KG * weight_kg
    Vmax = KE * Vd * KM_ELIM  # mg/h — at low C, Vmax·C/(Km+C) ≈ ke·Vd·C
    return OneCmtMMParams(
        ka=KA_FED if with_food else KA_FASTING,
        Vd=Vd,
        Vmax=Vmax,
        Km=KM_ELIM,
        F=F_ORAL,
    )


def make_2cmt_mm(weight_kg: float, with_food: bool = False) -> TwoCmtMMParams:
    """Create 2-compartment MM params."""
    V1 = VD_PER_KG * weight_kg
    V2 = V1 * V2_RATIO
    Q = V1 * Q_RATIO
    Vmax = KE * V1 * KM_ELIM
    return TwoCmtMMParams(
        ka=KA_FED if with_food else KA_FASTING,
        V1=V1, V2=V2, Q=Q,
        Vmax=Vmax, Km=KM_ELIM,
        F=F_ORAL,
    )


def make_prodrug(weight_kg: float, with_food: bool = False) -> ProdrugParams:
    """
    Create full LDX→d-AMP prodrug model params.

    Prodrug conversion Vmax is calibrated so that at low LDX concentrations,
    conversion appears first-order with effective rate ~1.5 h⁻¹ (rapid
    cleavage). At high substrate, the RBC peptidase pathway saturates,
    delaying Tmax — matching the observed dose-dependent Tmax shift
    (Ermer et al. 2010).
    """
    V1 = VD_PER_KG * weight_kg
    V2 = V1 * V2_RATIO
    Q = V1 * Q_RATIO
    V_blood = V_BLOOD_PER_KG * weight_kg
    Vmax_elim = KE * V1 * KM_ELIM

    # Conversion: at low C, effective first-order rate ≈ Vmax_conv / (Km_conv · V_blood)
    # Target ~2.0 h⁻¹ apparent first-order rate for rapid low-dose conversion.
    # At 50mg (C_ldx ≈ 10 mg/L, ~0.67×Km), conversion is moderately saturated
    # → slightly delayed Tmax. At 100mg+ severe saturation → long delayed Tmax.
    k_conv_apparent = 2.0  # h⁻¹
    Vmax_conv = k_conv_apparent * KM_CONV * V_blood

    return ProdrugParams(
        ka=KA_FED if with_food else KA_FASTING,
        F=F_ORAL,
        mw_ratio=MW_RATIO,
        V_blood=V_blood,
        Vmax_conv=Vmax_conv,
        Km_conv=KM_CONV,
        V1=V1, V2=V2, Q=Q,
        Vmax=Vmax_elim,
        Km=KM_ELIM,
    )


# ── Analytical reference (for validation only) ──────────────────────────

def bateman_conc(
    t: float | np.ndarray,
    dose_mg: float,
    weight_kg: float,
    with_food: bool = False,
) -> float | np.ndarray:
    """
    Analytical Bateman equation for 1-compartment oral absorption.
    Returns plasma concentration in ng/mL.
    Used only to validate the RK4 solver against the V2 TypeScript engine.
    """
    ka = KA_FED if with_food else KA_FASTING
    Vd = VD_PER_KG * weight_kg
    dose_damp = dose_mg * MW_RATIO * F_ORAL  # effective d-AMP mass (mg)
    # C(t) in mg/L, then convert to ng/mL (×1000)
    conc = (dose_damp / Vd) * (ka / (ka - KE)) * (np.exp(-KE * t) - np.exp(-ka * t))
    return np.maximum(conc * 1000.0, 0.0)  # ng/mL
