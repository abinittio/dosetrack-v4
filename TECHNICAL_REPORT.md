# DoseTrack V4 — Technical Report

**dis-solved.com** · ML is the solvent.

---

## 1. Overview

DoseTrack V4 is a pharmacokinetic/pharmacodynamic (PK/PD) simulation tool for lisdexamfetamine (Vyvanse/LDX). Users log doses with a date, time, and mg amount. The app computes and renders the cumulative drug effect curve in real time using a full compartmental simulation engine.

**Design principle:** maximum technical depth, minimum interface. The only inputs are date, time, and dose in mg.

---

## 2. PK/PD Engine

### 2.1 Pharmacokinetic Model

The default model is a **prodrug 2-compartment system with Michaelis-Menten elimination**.

LDX is a prodrug. It is absorbed from the GI tract into blood, then enzymatically cleaved by RBC peptidases to produce d-amphetamine (d-AMP), the active metabolite. This conversion is saturable — modelled as Michaelis-Menten kinetics. d-AMP then distributes into a 2-compartment system (central + peripheral) and is eliminated via MM kinetics.

**State vector (6 variables):**

| Index | Variable | Units | Description |
|-------|----------|-------|-------------|
| 0 | A_gut | mg | LDX in GI tract |
| 1 | A_prodrug | mg | LDX in blood |
| 2 | A_central | mg | d-AMP in central compartment |
| 3 | A_peripheral | mg | d-AMP in peripheral compartment |
| 4 | S | 0–1 | Dopamine vesicle store level |
| 5 | T_acute | 0–0.35 | Acute tolerance |

**ODE system:**

```
dA_gut/dt     = -ka · A_gut
dA_prodrug/dt = ka · A_gut · F  -  Vmax_conv · C_ldx / (Km_conv + C_ldx)
dA_central/dt = conv_rate · MW_ratio  -  Vmax · C1/(Km + C1)  -  Q·(C1 - C2)
dA_periph/dt  = Q · (C1 - C2)
dS/dt         = k_synth·(1 - S)  -  k_release · E_eff · S
dT_acute/dt   = k_on · E_eff · (T_max - T_acute)  -  k_off · T_acute
```

**Published constants (Pennick 2010, Ermer et al. 2010):**

| Parameter | Value | Source |
|-----------|-------|--------|
| ka (fasting) | 0.85 h⁻¹ | Ermer et al. |
| ka (fed) | 0.50 h⁻¹ | Ermer et al. |
| F (oral bioavailability) | 0.96 | Pennick 2010 |
| MW ratio (d-AMP/LDX) | 0.5135 | Molecular weights |
| t½ d-AMP | 11 h | Published mean |
| Vd | 3.5 L/kg | Published |
| Km (elimination) | 0.3 mg/L | Calibrated |
| Km (conversion) | 15.0 mg/L | Calibrated |

### 2.2 Pharmacodynamic Model

**Effect** is computed via the **sigmoid Emax (Hill) equation**:

```
E_raw = C^γ / (EC50^γ + C^γ)
```

Where:
- `C` = plasma d-AMP concentration (ng/mL)
- `EC50` = 30 ng/mL (concentration for 50% effect)
- `γ` = 1.5 (Hill coefficient, controls steepness)

**Effective subjective effect** is then modulated by dopamine store level and sleep:

```
E_eff = Emax · E_raw · S^α · sleep_penalty
```

- `S^α` (α = 0.5): square-root scaling. Depleted stores reduce available DA for release. The square root means partial depletion has a gentler impact than total depletion.
- `sleep_penalty`: multiplicative reduction (up to 30%) based on 7-day sleep debt.

**Tolerance — EC50 rightward shift:**

```
EC50_adj = EC50 · (1 + min(0.60, T_acute + T_chronic))
```

- Acute tolerance builds during dosing (4h half-life decay)
- Chronic tolerance computed from 7-day rolling average daily dose

**Dopamine store dynamics:**

```
dS/dt = k_synth·(1 - S) - k_release·E_eff·S
```

- `k_synth = 0.020 h⁻¹` → recovery t½ ≈ 35h at S=0 (models TH product inhibition)
- `k_release = 0.06 h⁻¹` → 5 consecutive days depletes to ~50-60% (matches clinical reports)

**Therapeutic zones:**

| Zone | Effect % | Interpretation |
|------|----------|----------------|
| Supratherapeutic | ≥ 85% | Overstimulation risk |
| Peak | 65–85% | Optimal focus window |
| Therapeutic | 40–65% | Functional range |
| Subtherapeutic | 15–40% | Wearing off |
| Baseline | < 15% | No active effect |

### 2.3 Numerical Solver

The ODE system is integrated using **fixed-step 4th-order Runge-Kutta (RK4)**:

```
k1 = f(t, y)
k2 = f(t + dt/2, y + dt/2·k1)
k3 = f(t + dt/2, y + dt/2·k2)
k4 = f(t + dt,   y + dt·k3)
y_next = y + (dt/6)·(k1 + 2k2 + 2k3 + k4)
```

**Step size:** `dt = 0.01h` (36 seconds). Output decimated to `dt_output = 0.05h` (3 minutes).

**Dose injection:** doses are treated as bolus events. The integration is split at each dose time — the drug mass is added to `A_gut` at the event boundary, then integration resumes. This preserves continuity of all other states.

RK4 was chosen over scipy ODE solvers for transparency and zero dependency on adaptive stepping. MM kinetics at therapeutic doses is not stiff enough to require adaptive step control.

---

## 3. Application Architecture

```
dosetrack-v4/
├── app.py                    # Streamlit UI (single file)
├── dosetrack/                # PK/PD engine (self-contained module)
│   ├── __init__.py
│   ├── solver.py             # RK4 integrator
│   ├── pk_models.py          # ODE constructors + parameter factories
│   ├── pd_models.py          # Sigmoid Emax, DA depletion, tolerance, sleep
│   ├── simulation.py         # Orchestrator — wires PK + PD into one ODE system
│   ├── dosing.py             # Dose and SleepLog dataclasses
│   ├── outputs.py            # Analysis functions (onset, duration, risk flags)
│   └── plotting.py           # Matplotlib figure builders
├── tests/
│   └── test_engine.py        # Pytest test suite
├── .github/workflows/ci.yml  # CI/CD pipeline
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev/test dependencies
└── ruff.toml                 # Linter configuration
```

### 3.1 Data Persistence

Doses are stored in a local **SQLite** database (`doses.db`), keyed by username.

```sql
CREATE TABLE doses (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    NOT NULL,
    dt       TEXT    NOT NULL,   -- ISO 8601 datetime string
    mg       REAL    NOT NULL
);
```

Each page load reads from the DB directly — no session state caching. This means:
- Data persists across browser refreshes and app restarts
- Multiple named users can coexist in the same DB
- No account passwords — name is the only identifier (personal/local use)

---

## 4. CI/CD Pipeline

### 4.1 Pipeline Overview

```
push to main
    │
    ▼
[lint] ──── ruff check app.py tests/
    │
    ▼
[test] ──── pytest on Python 3.10 / 3.11 / 3.12 (parallel matrix)
         └─ pytest-cov coverage report (fail if < 60%)
         └─ upload .coverage artifact (3.11 only)
    │
    ▼ (only on push to main, after tests pass)
[deploy] ── Streamlit Cloud auto-deploys from main branch
```

### 4.2 Lint (ruff)

**ruff** is a Rust-based Python linter, ~100x faster than flake8. It checks:

- `E` — pycodestyle errors (spacing, indentation, syntax)
- `F` — pyflakes (undefined names, unused imports)
- `W` — warnings

The `dosetrack/` engine module is excluded from linting — it's a copied library module, not application code.

### 4.3 Matrix Testing

Tests run in parallel across **Python 3.10, 3.11, and 3.12**. This catches:
- Syntax or API differences between Python versions
- Dependency behaviour changes across interpreter versions
- Type annotation compatibility

### 4.4 Coverage

**pytest-cov** tracks which lines of the engine are exercised by the test suite. Current coverage: **50% total** (engine includes V3 plotting/output functions not used in V4's minimal UI).

The pipeline fails if coverage drops below **60%** on core modules. The threshold is set conservatively because the dosetrack/ engine includes V3 analysis functions (redose, multi-day, plotting) not exercised by the V4 test suite.

The `.coverage` artifact is uploaded after the 3.11 run for inspection.

### 4.5 Continuous Deployment

Streamlit Cloud is connected directly to the GitHub repository. On every push to `main` that passes CI, Streamlit Cloud automatically pulls and redeploys the app. No additional Actions step is required.

**Note on persistence:** Streamlit Cloud's filesystem is ephemeral — `doses.db` does not persist between deploys or restarts. For a production deployment, SQLite should be replaced with a hosted database (e.g. Supabase Postgres via `psycopg2`, with credentials stored in Streamlit secrets).

---

## 5. Dependencies

**Runtime (`requirements.txt`):**

| Package | Role |
|---------|------|
| streamlit ≥ 1.30 | Web UI framework |
| numpy ≥ 1.24 | Numerical arrays, ODE state vectors |
| matplotlib ≥ 3.7 | PK/PD curve rendering |

**Dev (`requirements-dev.txt`):**

| Package | Role |
|---------|------|
| pytest ≥ 7.4 | Test runner |
| pytest-cov ≥ 4.1 | Coverage measurement |
| ruff ≥ 0.4 | Linter |

The engine has **no external scientific dependencies** — scipy, rdkit, and torch are not required. The RK4 solver is implemented from scratch in ~30 lines of numpy.

---

## 6. References

- Pennick M. (2010). Absorption of lisdexamfetamine and its conversion to d-amphetamine. *Neuropsychiatric Disease and Treatment*, 6, 317–327.
- Ermer J. et al. (2010). Lisdexamfetamine dimesylate: linear dose-proportionality, low intersubject and intrasubject variability, and safety in an open-label single-dose pharmacokinetic study in healthy adult volunteers. *Journal of Clinical Pharmacology*, 50(9), 1001–1010.

---

*DoseTrack V4 — dis-solved.com — For personal and educational use only. Not medical advice.*
