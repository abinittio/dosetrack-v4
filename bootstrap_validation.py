"""
Bootstrap MAPE confidence intervals and 1-cmt linear model comparison.
Produces statistics needed for the JPKPD manuscript.

Two analyses:
  1. Parametric Monte Carlo: sample Cmax from N(obs, sd) for each of the 6
     datasets, compute MAPE 10,000 times → 95% CI on MAPE.
  2. 1-cmt linear (Bateman analytical) vs prodrug 2-cmt MM — same 6 datasets.
     Reports MAPE, AIC proxy (sum of squared log-errors, proportional to AIC
     for a model with equal observation weights).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dosetrack import simulate, Dose
from dosetrack.pk_models import bateman_conc

np.random.seed(42)
N_BOOT = 10_000

# ── Observations (mean, sd, or ci95-derived sd) ──────────────────────────────

# (label, dose_mg, weight_kg, cmax_obs, cmax_sd, auc_obs, auc_sd_or_se)
# For Dolder: sd estimated from 95%CI as (upper-lower)/3.92
DATASETS = [
    ("Boellner 30mg child",  30,  34.0,  53.2,   9.62,  844.6,  116.7),
    ("Boellner 50mg child",  50,  34.0,  93.3,  18.20, 1510.0,  241.6),
    ("Boellner 70mg child",  70,  34.0, 134.0,  26.10, 2157.0,  383.3),
    ("Ermer 2016  50mg adult",50, 70.0,  41.4,   9.00,  748.7,  165.0),
    ("Krishnan   70mg adult", 70, 70.0,  69.3,  14.30, 1110.0,  314.2),
    ("Dolder 2017 100mg adult",100,70.0,118.0,   5.10, 1817.0,   96.9),
    # Dolder Cmax SD = (128-108)/3.92 = 5.1; AUC SD = (2017-1637)/3.92 = 97.0
]

# ── Run prodrug MM simulations (once — deterministic) ─────────────────────────

KE_TRUE = np.log(2) / 11  # published d-AMP elimination rate (t1/2 = 11h)


def run_prodrug(dose_mg, weight_kg):
    r = simulate(
        doses=[Dose(time_h=0.0, amount_mg=float(dose_mg))],
        weight_kg=weight_kg,
        model="prodrug",
        t_span=(0.0, 72.0),
    )
    t = np.array(r.t)
    c = np.array(r.plasma_conc)
    cmax = float(np.max(c))

    # AUC0-inf: trapezoidal to 24h + tail extrapolation using published ke
    # Using t1/2=11h (Pennick 2010) rather than the simulated terminal slope,
    # because the 2-cmt model has a spurious redistribution tail beyond ~24h.
    mask24 = t <= 24.0
    auc = float(np.trapz(c[mask24], t[mask24])) + float(c[mask24][-1]) / KE_TRUE
    return cmax, auc


def run_1cmt_linear(dose_mg, weight_kg):
    """Bateman analytical solution — fasting ka."""
    t = np.linspace(0, 72, 7200)
    c = bateman_conc(t, dose_mg, weight_kg, with_food=False)
    cmax = float(np.max(c))
    mask24 = t <= 24.0
    auc = float(np.trapz(c[mask24], t[mask24])) + float(c[mask24][-1]) / KE_TRUE
    return cmax, auc


print("Running simulations…")
prodrug_results = []
linear_results  = []
for label, dose, wt, *_ in DATASETS:
    prodrug_results.append(run_prodrug(dose, wt))
    linear_results.append(run_1cmt_linear(dose, wt))
    print(f"  {label}: done")

# ── Point-estimate MAPE (Cmax and AUC) ───────────────────────────────────────

def mape(sims, obs):
    return 100.0 * np.mean([abs(s - o) / o for s, o in zip(sims, obs)])

cmax_obs = [d[3] for d in DATASETS]
auc_obs  = [d[5] for d in DATASETS]

prodrug_cmax_sim = [r[0] for r in prodrug_results]
prodrug_auc_sim  = [r[1] for r in prodrug_results]
linear_cmax_sim  = [r[0] for r in linear_results]
linear_auc_sim   = [r[1] for r in linear_results]

print("\n" + "=" * 70)
print("POINT-ESTIMATE MAPE")
print("=" * 70)
print(f"  Prodrug 2-cmt MM  — Cmax MAPE: {mape(prodrug_cmax_sim, cmax_obs):.1f}%  "
      f"AUC MAPE: {mape(prodrug_auc_sim, auc_obs):.1f}%")
print(f"  1-cmt linear      — Cmax MAPE: {mape(linear_cmax_sim, cmax_obs):.1f}%  "
      f"AUC MAPE: {mape(linear_auc_sim, auc_obs):.1f}%")

# ── Parametric bootstrap (Monte Carlo on observed noise) ─────────────────────
# For each bootstrap iteration, sample obs Cmax from N(obs_mean, obs_sd)
# and recompute MAPE against the fixed simulation output.

cmax_sd = [d[4] for d in DATASETS]
auc_sd  = [d[6] for d in DATASETS]

boot_mape_prodrug_cmax = []
boot_mape_prodrug_auc  = []
boot_mape_linear_cmax  = []
boot_mape_linear_auc   = []

cmax_obs_arr  = np.array(cmax_obs)
cmax_sd_arr   = np.array(cmax_sd)
auc_obs_arr   = np.array(auc_obs)
auc_sd_arr    = np.array(auc_sd)
p_cmax = np.array(prodrug_cmax_sim)
p_auc  = np.array(prodrug_auc_sim)
l_cmax = np.array(linear_cmax_sim)
l_auc  = np.array(linear_auc_sim)

for _ in range(N_BOOT):
    sampled_cmax = np.random.normal(cmax_obs_arr, cmax_sd_arr)
    sampled_auc  = np.random.normal(auc_obs_arr,  auc_sd_arr)
    # keep positive
    sampled_cmax = np.maximum(sampled_cmax, 1.0)
    sampled_auc  = np.maximum(sampled_auc,  1.0)

    boot_mape_prodrug_cmax.append(100 * np.mean(np.abs(p_cmax - sampled_cmax) / sampled_cmax))
    boot_mape_prodrug_auc.append( 100 * np.mean(np.abs(p_auc  - sampled_auc)  / sampled_auc))
    boot_mape_linear_cmax.append( 100 * np.mean(np.abs(l_cmax - sampled_cmax) / sampled_cmax))
    boot_mape_linear_auc.append(  100 * np.mean(np.abs(l_auc  - sampled_auc)  / sampled_auc))

def ci95(arr):
    a = np.array(arr)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)

print("\n" + "=" * 70)
print(f"PARAMETRIC BOOTSTRAP MAPE  (N={N_BOOT:,} iterations, 95% CI)")
print("=" * 70)

pc_lo, pc_hi = ci95(boot_mape_prodrug_cmax)
pa_lo, pa_hi = ci95(boot_mape_prodrug_auc)
lc_lo, lc_hi = ci95(boot_mape_linear_cmax)
la_lo, la_hi = ci95(boot_mape_linear_auc)

print(f"  Prodrug 2-cmt MM:")
print(f"    Cmax MAPE = {mape(prodrug_cmax_sim, cmax_obs):.1f}%  (95% CI: {pc_lo:.1f}–{pc_hi:.1f}%)")
print(f"    AUC  MAPE = {mape(prodrug_auc_sim,  auc_obs):.1f}%  (95% CI: {pa_lo:.1f}–{pa_hi:.1f}%)")
print(f"  1-cmt linear (Bateman):")
print(f"    Cmax MAPE = {mape(linear_cmax_sim, cmax_obs):.1f}%  (95% CI: {lc_lo:.1f}–{lc_hi:.1f}%)")
print(f"    AUC  MAPE = {mape(linear_auc_sim,  auc_obs):.1f}%  (95% CI: {la_lo:.1f}–{la_hi:.1f}%)")

# ── AIC proxy: sum of squared log-prediction-errors ──────────────────────────
# AIC ∝ n·ln(RSS/n) + 2k  (Gaussian likelihood approximation)
# Prodrug MM: k=0 free params (all fixed); 1-cmt linear: k=0 free params
# So model with lower RSS is preferred — we just report RSS and delta-AIC

def rss_log(sims, obs):
    return np.sum([(np.log(s) - np.log(o))**2 for s, o in zip(sims, obs)])

n = len(DATASETS)
rss_prod = rss_log(prodrug_cmax_sim, cmax_obs)
rss_lin  = rss_log(linear_cmax_sim,  cmax_obs)

# Both models have 0 free params fitted to these data (all from literature)
aic_prod = n * np.log(rss_prod / n)
aic_lin  = n * np.log(rss_lin  / n)

print("\n" + "=" * 70)
print("AIC COMPARISON (Gaussian log-likelihood, Cmax, k=0 free params each)")
print("=" * 70)
print(f"  Prodrug 2-cmt MM  RSS_log = {rss_prod:.4f}  AIC_proxy = {aic_prod:.2f}")
print(f"  1-cmt linear      RSS_log = {rss_lin:.4f}   AIC_proxy = {aic_lin:.2f}")
print(f"  dAIC (linear - MM) = {aic_lin - aic_prod:.2f}  (positive -> MM preferred)")

# ── Per-dataset breakdown table ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("PER-DATASET BREAKDOWN — Cmax (ng/mL)")
print("=" * 70)
print(f"{'Dataset':<28} {'Obs':>7} {'±SD':>6} {'MM_sim':>8} {'MM%err':>8} {'Lin_sim':>8} {'Lin%err':>8}")
print("-" * 78)
for i, (label, dose, wt, obs, sd, auc_o, auc_s) in enumerate(DATASETS):
    mm  = prodrug_cmax_sim[i]
    lin = linear_cmax_sim[i]
    print(f"{label:<28} {obs:>7.1f} {sd:>6.1f} {mm:>8.1f} {100*(mm-obs)/obs:>7.1f}% "
          f"{lin:>8.1f} {100*(lin-obs)/obs:>7.1f}%")

print("\n" + "=" * 70)
print("PER-DATASET BREAKDOWN — AUC (ng·h/mL)")
print("=" * 70)
print(f"{'Dataset':<28} {'Obs':>8} {'±SD':>7} {'MM_sim':>9} {'MM%err':>8} {'Lin_sim':>9} {'Lin%err':>8}")
print("-" * 82)
for i, (label, dose, wt, cmax_o, cmax_s, auc_o, auc_s) in enumerate(DATASETS):
    mm  = prodrug_auc_sim[i]
    lin = linear_auc_sim[i]
    print(f"{label:<28} {auc_o:>8.1f} {auc_s:>7.1f} {mm:>9.1f} {100*(mm-auc_o)/auc_o:>7.1f}% "
          f"{lin:>9.1f} {100*(lin-auc_o)/auc_o:>7.1f}%")

print("\nDone.")
