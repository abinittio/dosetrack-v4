"""
DoseTrack V4 — PK Validation Against Published Literature

Datasets:
  1. Boellner et al. 2010  — children (6-12y, ~34 kg), 30/50/70 mg, Clin Ther 32:252
  2. Ermer 2016 review     — adults, 50 mg (4 studies consolidated), PMC free
     Krishnan & Zhang 2008 — adults, 70 mg fasted, PMID 18991468
  3. Dolder et al. 2017    — Swiss adults, 100 mg LDX, Front Pharmacol (open access)
  4. Ermer et al. 2010     — supratherapeutic 50-250 mg (dose-proportionality claim)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dosetrack import simulate, Dose

# ── Published reference values ────────────────────────────────────────────────

# Dataset 1: Boellner et al. 2010 — children (~34 kg)
BOELLNER = {
    30:  {'Cmax': 53.2,  'Cmax_sd': 9.62,  'Tmax': 3.41, 'Tmax_sd': 1.09, 'AUC': 844.6,  'AUC_sd': 116.7, 't_half': 8.90,  't_half_sd': 1.33},
    50:  {'Cmax': 93.3,  'Cmax_sd': 18.2,  'Tmax': 3.58, 'Tmax_sd': 1.18, 'AUC': 1510.0, 'AUC_sd': 241.6, 't_half': 8.61,  't_half_sd': 1.04},
    70:  {'Cmax': 134.0, 'Cmax_sd': 26.1,  'Tmax': 3.46, 'Tmax_sd': 1.34, 'AUC': 2157.0, 'AUC_sd': 383.3, 't_half': 8.64,  't_half_sd': 1.32},
}
BOELLNER_WEIGHT = 34.0  # mean body weight, children 6-12y

# Dataset 2: Adults — Ermer 2016 review (50 mg mean of 4 studies) + Krishnan 2008 (70 mg)
ADULT_REF = {
    50:  {'Cmax': 41.4,  'Cmax_sd': 9.0,   'Tmax': 3.9,  'Tmax_sd': 0.8,  'AUC': 748.7,  'AUC_sd': 165.0, 't_half': 10.95, 't_half_sd': 2.2},
    70:  {'Cmax': 69.3,  'Cmax_sd': 14.3,  'Tmax': 3.78, 'Tmax_sd': 1.01, 'AUC': 1110.0, 'AUC_sd': 314.2, 't_half': 9.69,  't_half_sd': 1.96},
}
ADULT_WEIGHT = 70.0

# Dataset 3: Dolder et al. 2017 — Swiss adults, 100 mg LDX (open access)
DOLDER = {
    100: {'Cmax': 118.0, 'Cmax_ci95': (108, 128), 'Tmax': 4.6, 'Tmax_ci95': (4.1, 5.2),
          'AUC': 1817.0, 'AUC_ci95': (1637, 2017), 't_half': 7.9, 't_half_ci95': (7.1, 8.9)},
}
DOLDER_WEIGHT = 70.0

# Supratherapeutic: Ermer 2010 — dose-proportional up to 250 mg
# Confirmed dose-proportional Cmax and AUC; Tmax 4-6h range reported
# We extrapolate Cmax from adult 70mg reference assuming linearity
ERMER_SUPRA_DOSES = [50, 100, 150, 200, 250]
ERMER_CMAX_PER_MG = 69.3 / 70.0  # ng/mL per mg, from Krishnan 70mg
ERMER_SUPRA_LINEAR = {d: ERMER_CMAX_PER_MG * d for d in ERMER_SUPRA_DOSES}

# ── Simulation functions ──────────────────────────────────────────────────────

def run(mg, weight_kg=70.0, sim_h=72.0):
    r = simulate(
        doses=[Dose(time_h=0.0, amount_mg=float(mg))],
        weight_kg=weight_kg,
        model="prodrug",
        t_span=(0.0, sim_h),
    )
    return r

def extract_pk(r, sim_h=72.0):
    t   = np.array(r.t)
    c   = np.array(r.plasma_conc)

    idx_max  = int(np.argmax(c))
    cmax     = float(c[idx_max])
    tmax     = float(t[idx_max])

    # Terminal t½ via log-linear regression on last 40% of curve
    t_start  = t[int(len(t) * 0.6)]
    mask     = (t >= t_start) & (c > cmax * 0.01)
    if mask.sum() > 5:
        slope, _ = np.polyfit(t[mask], np.log(c[mask] + 1e-12), 1)
        t_half = float(-np.log(2) / slope) if slope < 0 else float('nan')
    else:
        t_half = float('nan')

    # AUC0-inf via trapezoidal + extrapolation
    auc_trap = float(np.trapz(c, t))
    ke       = np.log(2) / t_half if not np.isnan(t_half) else 0
    c_last   = float(c[-1])
    auc_extrap = c_last / ke if ke > 0 else 0
    auc      = auc_trap + auc_extrap

    return {'Cmax': round(cmax, 1), 'Tmax': round(tmax, 2),
            't_half': round(t_half, 2), 'AUC': round(auc, 1)}

def pct_err(sim, obs):
    return 100.0 * (sim - obs) / obs

# ── Run all simulations ───────────────────────────────────────────────────────

print("=" * 70)
print("DATASET 1: Boellner et al. 2010 — Children (34 kg), 30/50/70 mg")
print("=" * 70)
results_boellner = {}
for mg, ref in BOELLNER.items():
    r   = run(mg, weight_kg=BOELLNER_WEIGHT)
    pk  = extract_pk(r)
    results_boellner[mg] = pk
    print(f"\n{mg} mg LDX (child, {BOELLNER_WEIGHT} kg):")
    print(f"  Cmax   sim={pk['Cmax']:6.1f}  obs={ref['Cmax']:.1f}±{ref['Cmax_sd']:.1f}  err={pct_err(pk['Cmax'],ref['Cmax']):+.1f}%")
    print(f"  Tmax   sim={pk['Tmax']:6.2f}  obs={ref['Tmax']:.2f}±{ref['Tmax_sd']:.2f}  err={pct_err(pk['Tmax'],ref['Tmax']):+.1f}%")
    print(f"  t½     sim={pk['t_half']:6.2f}  obs={ref['t_half']:.2f}±{ref['t_half_sd']:.2f}  err={pct_err(pk['t_half'],ref['t_half']):+.1f}%")
    print(f"  AUC    sim={pk['AUC']:6.1f}  obs={ref['AUC']:.1f}±{ref['AUC_sd']:.1f}  err={pct_err(pk['AUC'],ref['AUC']):+.1f}%")

print("\n" + "=" * 70)
print("DATASET 2: Ermer 2016 review / Krishnan 2008 — Adults (70 kg), 50/70 mg")
print("=" * 70)
results_adult = {}
for mg, ref in ADULT_REF.items():
    r   = run(mg, weight_kg=ADULT_WEIGHT)
    pk  = extract_pk(r)
    results_adult[mg] = pk
    print(f"\n{mg} mg LDX (adult, {ADULT_WEIGHT} kg):")
    print(f"  Cmax   sim={pk['Cmax']:6.1f}  obs={ref['Cmax']:.1f}±{ref['Cmax_sd']:.1f}  err={pct_err(pk['Cmax'],ref['Cmax']):+.1f}%")
    print(f"  Tmax   sim={pk['Tmax']:6.2f}  obs={ref['Tmax']:.2f}±{ref['Tmax_sd']:.2f}  err={pct_err(pk['Tmax'],ref['Tmax']):+.1f}%")
    print(f"  t½     sim={pk['t_half']:6.2f}  obs={ref['t_half']:.2f}±{ref['t_half_sd']:.2f}  err={pct_err(pk['t_half'],ref['t_half']):+.1f}%")
    print(f"  AUC    sim={pk['AUC']:6.1f}  obs={ref['AUC']:.1f}±{ref['AUC_sd']:.1f}  err={pct_err(pk['AUC'],ref['AUC']):+.1f}%")

print("\n" + "=" * 70)
print("DATASET 3: Dolder et al. 2017 — Swiss Adults (70 kg), 100 mg LDX")
print("=" * 70)
results_dolder = {}
for mg, ref in DOLDER.items():
    r   = run(mg, weight_kg=DOLDER_WEIGHT)
    pk  = extract_pk(r)
    results_dolder[mg] = pk
    print(f"\n{mg} mg LDX (adult, {DOLDER_WEIGHT} kg):")
    print(f"  Cmax   sim={pk['Cmax']:6.1f}  obs={ref['Cmax']:.1f} (95%CI {ref['Cmax_ci95']})  err={pct_err(pk['Cmax'],ref['Cmax']):+.1f}%")
    print(f"  Tmax   sim={pk['Tmax']:6.2f}  obs={ref['Tmax']:.2f} (95%CI {ref['Tmax_ci95']})  err={pct_err(pk['Tmax'],ref['Tmax']):+.1f}%")
    print(f"  t½     sim={pk['t_half']:6.2f}  obs={ref['t_half']:.2f} (95%CI {ref['t_half_ci95']})  err={pct_err(pk['t_half'],ref['t_half']):+.1f}%")
    print(f"  AUC    sim={pk['AUC']:6.1f}  obs={ref['AUC']:.1f} (95%CI {ref['AUC_ci95']})  err={pct_err(pk['AUC'],ref['AUC']):+.1f}%")

print("\n" + "=" * 70)
print("DATASET 4: Supratherapeutic doses — MM non-linearity vs. linear model")
print("=" * 70)
supra_doses = [50, 100, 150, 200, 250]
supra_sim   = {}
for mg in supra_doses:
    r  = run(mg, weight_kg=ADULT_WEIGHT)
    pk = extract_pk(r)
    supra_sim[mg] = pk
    linear_cmax = ERMER_CMAX_PER_MG * mg
    print(f"  {mg:3d} mg: sim Cmax={pk['Cmax']:7.1f}  linear pred={linear_cmax:7.1f}  "
          f"divergence={pct_err(pk['Cmax'], linear_cmax):+.1f}%  sim Tmax={pk['Tmax']:.2f}h")

# ── Figures ───────────────────────────────────────────────────────────────────

DARK_BG   = "#08111a"
TEAL      = "#5eead4"
ORANGE    = "#f97316"
AMBER     = "#f59e0b"
BLUE      = "#60a5fa"
GREY_LINE = "#334155"
GREY_TEXT = "#64748b"

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel A: Concentration-time curves — children (Boellner) ─────────────────
ax_a = fig.add_subplot(gs[0, :2])
ax_a.set_facecolor(DARK_BG)
colours = [TEAL, ORANGE, AMBER]
for i, (mg, ref) in enumerate(BOELLNER.items()):
    r = run(mg, weight_kg=BOELLNER_WEIGHT)
    t = np.array(r.t)
    c = np.array(r.plasma_conc)
    ax_a.plot(t, c, color=colours[i], linewidth=2.0, label=f"Sim {mg} mg", zorder=4)
    ax_a.errorbar(ref['Tmax'], ref['Cmax'], yerr=ref['Cmax_sd'],
                  fmt='o', color=colours[i], markersize=7, capsize=4,
                  label=f"Boellner {mg} mg", zorder=5, alpha=0.9)

ax_a.set_xlim(0, 36); ax_a.set_ylim(0, 200)
ax_a.set_xlabel("Time (h)", color=GREY_TEXT, fontsize=9)
ax_a.set_ylabel("d-AMP plasma conc. (ng/mL)", color=GREY_TEXT, fontsize=9)
ax_a.set_title("A — Boellner et al. 2010: Children (34 kg), 30/50/70 mg",
               color="#e2e8f0", fontsize=9.5, pad=8, loc="left")
ax_a.legend(fontsize=7.5, ncol=2, framealpha=0, labelcolor="#94a3b8")
for sp in ax_a.spines.values(): sp.set_visible(False)
ax_a.tick_params(colors=GREY_TEXT, labelsize=8); ax_a.grid(color="#111f2e", linewidth=0.8)

# ── Panel B: Cmax comparison bar chart — all datasets ────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
ax_b.set_facecolor(DARK_BG)

labels    = ['30c', '50c', '70c', '50a', '70a', '100a']
obs_cmax  = [BOELLNER[30]['Cmax'], BOELLNER[50]['Cmax'], BOELLNER[70]['Cmax'],
             ADULT_REF[50]['Cmax'], ADULT_REF[70]['Cmax'], DOLDER[100]['Cmax']]
obs_sd    = [BOELLNER[30]['Cmax_sd'], BOELLNER[50]['Cmax_sd'], BOELLNER[70]['Cmax_sd'],
             ADULT_REF[50]['Cmax_sd'], ADULT_REF[70]['Cmax_sd'],
             (DOLDER[100]['Cmax_ci95'][1]-DOLDER[100]['Cmax_ci95'][0])/3.92]
sim_cmax  = [results_boellner[30]['Cmax'], results_boellner[50]['Cmax'], results_boellner[70]['Cmax'],
             results_adult[50]['Cmax'], results_adult[70]['Cmax'], results_dolder[100]['Cmax']]

x = np.arange(len(labels))
w = 0.35
ax_b.bar(x - w/2, obs_cmax, w, yerr=obs_sd, color=TEAL, alpha=0.7, label="Published", capsize=3, error_kw={'ecolor': '#94a3b8', 'linewidth': 1})
ax_b.bar(x + w/2, sim_cmax, w, color=ORANGE, alpha=0.8, label="DoseTrack sim")
ax_b.set_xticks(x); ax_b.set_xticklabels(labels, fontsize=7.5, color=GREY_TEXT)
ax_b.set_ylabel("Cmax (ng/mL)", color=GREY_TEXT, fontsize=8)
ax_b.set_title("B — Cmax: Observed vs. Simulated", color="#e2e8f0", fontsize=9.5, pad=8, loc="left")
ax_b.legend(fontsize=7.5, framealpha=0, labelcolor="#94a3b8")
for sp in ax_b.spines.values(): sp.set_visible(False)
ax_b.tick_params(colors=GREY_TEXT, labelsize=8); ax_b.grid(axis='y', color="#111f2e", linewidth=0.8)
ax_b.set_ylim(0, 200)
# Annotate c=children, a=adult
ax_b.text(0.5, -0.18, "c=children (34 kg)   a=adults (70 kg)", transform=ax_b.transAxes,
          fontsize=6.5, color=GREY_TEXT, ha='center')

# ── Panel C: Adult concentration-time — 50 mg and 70 mg ─────────────────────
ax_c = fig.add_subplot(gs[1, :2])
ax_c.set_facecolor(DARK_BG)

for mg, ref, col in [(50, ADULT_REF[50], TEAL), (70, ADULT_REF[70], ORANGE)]:
    r = run(mg, weight_kg=ADULT_WEIGHT)
    t = np.array(r.t)
    c = np.array(r.plasma_conc)
    ax_c.plot(t, c, color=col, linewidth=2.0, label=f"Sim {mg} mg adult", zorder=4)
    ax_c.errorbar(ref['Tmax'], ref['Cmax'], yerr=ref['Cmax_sd'], xerr=ref['Tmax_sd'],
                  fmt='D', color=col, markersize=7, capsize=4, zorder=5, alpha=0.9,
                  label=f"Published {mg} mg")

# Dolder 100mg
r100 = run(100, weight_kg=ADULT_WEIGHT)
t100 = np.array(r100.t); c100 = np.array(r100.plasma_conc)
ax_c.plot(t100, c100, color=BLUE, linewidth=2.0, label="Sim 100 mg adult", zorder=4)
dol = DOLDER[100]
dol_ci = dol['Cmax_ci95']
ax_c.errorbar(dol['Tmax'], dol['Cmax'],
              yerr=[[dol['Cmax']-dol_ci[0]], [dol_ci[1]-dol['Cmax']]],
              fmt='s', color=BLUE, markersize=7, capsize=4, zorder=5, alpha=0.9,
              label="Dolder 2017, 100 mg")

ax_c.set_xlim(0, 48); ax_c.set_ylim(0, 160)
ax_c.set_xlabel("Time (h)", color=GREY_TEXT, fontsize=9)
ax_c.set_ylabel("d-AMP plasma conc. (ng/mL)", color=GREY_TEXT, fontsize=9)
ax_c.set_title("C — Adults (70 kg): Ermer 2016 review / Krishnan 2008 / Dolder 2017",
               color="#e2e8f0", fontsize=9.5, pad=8, loc="left")
ax_c.legend(fontsize=7.5, ncol=2, framealpha=0, labelcolor="#94a3b8")
for sp in ax_c.spines.values(): sp.set_visible(False)
ax_c.tick_params(colors=GREY_TEXT, labelsize=8); ax_c.grid(color="#111f2e", linewidth=0.8)

# ── Panel D: MM non-linearity vs. linear model — supratherapeutic ─────────────
ax_d = fig.add_subplot(gs[1, 2])
ax_d.set_facecolor(DARK_BG)

doses_x   = np.array(supra_doses)
sim_cmax_s = [supra_sim[d]['Cmax'] for d in supra_doses]
lin_cmax_s = [ERMER_CMAX_PER_MG * d for d in supra_doses]

ax_d.plot(doses_x, lin_cmax_s, '--', color=GREY_LINE, linewidth=1.8,
          label="Linear model (Ermer 2010)", zorder=3)
ax_d.plot(doses_x, sim_cmax_s, 'o-', color=AMBER, linewidth=2.2, markersize=7,
          label="DoseTrack (MM model)", zorder=4)
# Mark divergence region
ax_d.axvspan(150, 260, alpha=0.07, color="#ef4444")
ax_d.text(195, max(sim_cmax_s)*0.55, "MM\nsat.", color="#ef4444",
          fontsize=7.5, ha='center', alpha=0.8)

ax_d.set_xlabel("LDX dose (mg)", color=GREY_TEXT, fontsize=9)
ax_d.set_ylabel("Predicted Cmax (ng/mL)", color=GREY_TEXT, fontsize=9)
ax_d.set_title("D — MM Non-Linearity at Supratherapeutic Doses",
               color="#e2e8f0", fontsize=9.5, pad=8, loc="left")
ax_d.legend(fontsize=7.5, framealpha=0, labelcolor="#94a3b8")
for sp in ax_d.spines.values(): sp.set_visible(False)
ax_d.tick_params(colors=GREY_TEXT, labelsize=8); ax_d.grid(color="#111f2e", linewidth=0.8)

fig.suptitle("DoseTrack V4 — Pharmacokinetic Validation", color="#f0fdfa",
             fontsize=12, fontweight='bold', y=0.98)

out_path = os.path.join(os.path.dirname(__file__), "validation_figure.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close(fig)
print(f"\nFigure saved: {out_path}")
