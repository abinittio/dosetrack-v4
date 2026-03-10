# DoseTrack V4 — Technical Report

**dis-solved.com · ML is the solvent.**

*Pharmacokinetic/Pharmacodynamic simulation engine for lisdexamfetamine dimesylate (Vyvanse/LDX)*

---

## Table of Contents

1. [Overview](#1-overview)
2. [PK/PD Engine](#2-pkpd-engine)
   - 2.1 Pharmacokinetic Model
   - 2.2 Pharmacodynamic Model
   - 2.3 Numerical Solver
3. [Application Architecture](#3-application-architecture)
   - 3.1 Module Structure
   - 3.2 Data Persistence
4. [Infrastructure & Deployment](#4-infrastructure--deployment)
   - 4.1 Stack Overview
   - 4.2 Database — Supabase Postgres
   - 4.3 Hosting — Streamlit Community Cloud
   - 4.4 DNS & Routing — Cloudflare
5. [CI/CD Pipeline](#5-cicd-pipeline)
   - 5.1 Pipeline Overview
   - 5.2 Lint (ruff)
   - 5.3 Matrix Testing
   - 5.4 Coverage
   - 5.5 Continuous Deployment
6. [Dependencies](#6-dependencies)
7. [References](#7-references)

---

## 1. Overview

DoseTrack V4 is a pharmacokinetic/pharmacodynamic (PK/PD) simulation tool for lisdexamfetamine dimesylate (LDX). Users log doses with a date, time, and amount in mg. The app computes and renders the full cumulative drug effect curve in real time using a multi-compartment ODE simulation engine backed by published pharmacokinetic constants.

**Design principle:** maximum technical depth, minimum interface. The only inputs are date, time, and dose in mg. The app handles the rest — computing the plasma concentration trajectory, dopamine vesicle dynamics, tolerance accumulation, and projected therapeutic effect — and renders it as a continuous curve on a real-time axis.

**Live deployment:**
- App: https://dosetrack-v4.streamlit.app
- Custom domain: https://dosetrack.dis-solved.com

---

## 2. PK/PD Engine

### 2.1 Pharmacokinetic Model

The simulation uses a **prodrug 2-compartment model with Michaelis-Menten (MM) kinetics** for both conversion and elimination.

**Pharmacological mechanism:** LDX is a prodrug consisting of d-amphetamine covalently bound to L-lysine. It has negligible pharmacological activity in intact form. Following oral absorption, RBC peptidases cleave the lysine bond to release d-amphetamine (d-AMP), the active moiety. This enzymatic conversion is saturable and is therefore modelled as Michaelis-Menten kinetics rather than first-order. d-AMP distributes into a 2-compartment system and undergoes MM elimination.

**State vector — 6 variables:**

| Index | Variable | Units | Description |
|-------|----------|-------|-------------|
| 0 | A_gut | mg | LDX mass in GI tract |
| 1 | A_prodrug | mg | LDX mass in blood (pre-conversion) |
| 2 | A_central | mg | d-AMP in central compartment |
| 3 | A_peripheral | mg | d-AMP in peripheral compartment |
| 4 | S | 0–1 | Normalised dopamine vesicle store level |
| 5 | T_acute | 0–0.35 | Acute pharmacodynamic tolerance |

**ODE system:**

```
dA_gut/dt        = −ka · A_gut

dA_prodrug/dt    = ka · A_gut · F  −  Vmax_conv · C_ldx / (Km_conv + C_ldx)

dA_central/dt    = conv_rate · MW_ratio
                   − Vmax · C₁ / (Km + C₁)
                   − Q · (C₁ − C₂)

dA_periph/dt     = Q · (C₁ − C₂)

dS/dt            = k_synth · (1 − S)  −  k_release · E_eff · S

dT_acute/dt      = k_on · E_eff · (T_max − T_acute)  −  k_off · T_acute
```

Where `C₁ = A_central / Vd` and `C₂ = A_peripheral / Vd` are the central and peripheral d-AMP concentrations respectively.

**Published constants (Pennick 2010, Ermer et al. 2010):**

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Absorption rate constant (fasting) | ka | 0.85 h⁻¹ | Ermer et al. 2010 |
| Absorption rate constant (fed) | ka | 0.50 h⁻¹ | Ermer et al. 2010 |
| Oral bioavailability | F | 0.96 | Pennick 2010 |
| MW ratio (d-AMP / LDX) | MW_ratio | 0.5135 | Molecular weights |
| d-AMP elimination half-life | t½ | 11 h | Published mean |
| Volume of distribution | Vd | 3.5 L/kg | Published |
| MM elimination constant | Km | 0.3 mg/L | Calibrated |
| MM conversion constant | Km_conv | 15.0 mg/L | Calibrated |
| Inter-compartment clearance | Q | 1.2 L/h | Calibrated |

---

### 2.2 Pharmacodynamic Model

#### Sigmoid Emax (Hill Equation)

The relationship between plasma d-AMP concentration and pharmacodynamic effect is modelled using the **Hill equation** (sigmoid Emax model):

```
E_raw = Cᵞ / (EC50ᵞ + Cᵞ)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| C | plasma d-AMP (ng/mL) | central compartment concentration |
| EC50 | 30 ng/mL | concentration for 50% maximal effect |
| γ | 1.5 | Hill coefficient — controls sigmoid steepness |

The Hill coefficient γ > 1 produces a sigmoidal (S-shaped) E–C relationship, consistent with the observed threshold-like onset and plateau of amphetamine CNS effects.

#### Effective Subjective Effect

The raw effect is modulated by dopamine store availability and sleep debt:

```
E_eff = Emax · E_raw · S^α · sleep_penalty
```

- **S^α** (α = 0.5): Square-root scaling by DA store level. Depleted vesicular stores reduce the pool of DA available for stimulus-evoked release. The square-root exponent ensures that partial store depletion has a gentler impact than complete depletion — consistent with TH activity serving as a partial compensatory mechanism.
- **sleep_penalty**: Multiplicative reduction (up to 30%) computed from a 7-day rolling sleep debt model. Insufficient sleep reduces DA receptor sensitivity and baseline monoamine tone.

#### Tolerance

EC50 undergoes rightward shift from both acute and chronic tolerance:

```
EC50_adj = EC50 · (1 + min(0.60, T_acute + T_chronic))
```

**Acute tolerance** (`T_acute`): builds within a dosing session via receptor desensitisation and internalisation. Half-life of decay ≈ 4h. Governed by the T_acute ODE above.

**Chronic tolerance** (`T_chronic`): computed from the 7-day rolling average daily dose, scaled to a maximum rightward shift of 60%. Represents longer-term neuroadaptations including D2 receptor downregulation and reduced DAT internalisation response.

#### Dopamine Vesicle Store Dynamics

```
dS/dt = k_synth · (1 − S) − k_release · E_eff · S
```

| Constant | Value | Interpretation |
|----------|-------|----------------|
| k_synth | 0.020 h⁻¹ | Vesicular DA resynthesis (recovery t½ ≈ 35h at full depletion) |
| k_release | 0.06 h⁻¹ | Stimulus-evoked release rate scaled by effect magnitude |

Five consecutive days of dosing depletes stores to approximately 50–60%, consistent with clinical reports of diminished effect and post-dose fatigue in continuous use.

#### Therapeutic Zones

| Zone | Effect % | Clinical Interpretation |
|------|----------|------------------------|
| Supratherapeutic | ≥ 85% | Overstimulation, anxiety, cardiovascular risk |
| Peak | 65–85% | Optimal focus and attention window |
| Therapeutic | 40–65% | Functional, working memory and executive function supported |
| Subtherapeutic | 15–40% | Wearing off — executive function returning to baseline |
| Baseline | < 15% | No meaningful active effect |

---

### 2.3 Numerical Solver

The 6-variable ODE system is integrated using **fixed-step 4th-order Runge-Kutta (RK4)**:

```
k₁ = f(t,        y)
k₂ = f(t + dt/2, y + dt/2 · k₁)
k₃ = f(t + dt/2, y + dt/2 · k₂)
k₄ = f(t + dt,   y + dt   · k₃)

y_{n+1} = yₙ + (dt/6) · (k₁ + 2k₂ + 2k₃ + k₄)
```

**Step size:** `dt = 0.01 h` (36 seconds). Output decimated to `dt_out = 0.05 h` (3 minutes) for rendering.

**Dose injection (bolus events):** Integration is segmented at each dose time. At each event boundary, `A_gut` is incremented by the administered mass; all other states are continuous across the boundary. This correctly models the discrete-input, continuous-state nature of oral dosing without discontinuities in the ODE.

**Rationale for RK4 over adaptive solvers:** Michaelis-Menten kinetics at therapeutic LDX doses (10–70 mg) produces only mild stiffness. Fixed-step RK4 at dt = 0.01 h provides sufficient accuracy (global error O(dt⁴)) without the overhead of adaptive step control, and avoids the `scipy` dependency, keeping the engine fully self-contained in pure numpy.

---

## 3. Application Architecture

### 3.1 Module Structure

```
dosetrack-v4/
├── app.py                      # Streamlit UI — single-file, ~430 lines
├── dosetrack/                  # Self-contained PK/PD simulation engine
│   ├── __init__.py             # Public API: simulate(), Dose, SleepLog
│   ├── solver.py               # Fixed-step RK4 integrator with bolus injection
│   ├── pk_models.py            # ODE constructors + PK parameter factories
│   ├── pd_models.py            # Sigmoid Emax, DA depletion, tolerance, sleep
│   ├── simulation.py           # Orchestrator — composes PK+PD into single ODE
│   ├── dosing.py               # Dose and SleepLog dataclasses
│   ├── outputs.py              # Analysis: onset, duration, AUC, risk flags
│   └── plotting.py             # Matplotlib figure builders (V3 compatibility)
├── tests/
│   └── test_engine.py          # Pytest suite: engine correctness + DB operations
├── .github/
│   └── workflows/ci.yml        # CI/CD: lint → test matrix → deploy
├── requirements.txt            # Runtime: streamlit, numpy, matplotlib, psycopg2
├── requirements-dev.txt        # Dev: pytest, pytest-cov, ruff
└── ruff.toml                   # Linter config (excludes dosetrack/ engine)
```

The `dosetrack/` module is a fully self-contained library. It has no knowledge of Streamlit, databases, or UI. `app.py` is the thin adapter between the engine and the interface.

### 3.2 Data Persistence

The app supports two database backends, selected at runtime based on environment:

**Production — Supabase Postgres:**

```python
_USE_POSTGRES = "DATABASE_URL" in st.secrets

def _pg():
    return psycopg2.connect(st.secrets["DATABASE_URL"])
```

**Development — Local SQLite:**

```python
con = sqlite3.connect("doses.db")
```

**Schema (identical on both backends):**

```sql
CREATE TABLE IF NOT EXISTS doses (
    id       SERIAL PRIMARY KEY,   -- AUTOINCREMENT in SQLite
    username TEXT   NOT NULL,
    dt       TEXT   NOT NULL,      -- ISO 8601 datetime string
    mg       REAL   NOT NULL
);
```

Each page load reads from the database directly — no session state caching of dose data. This ensures:
- Doses persist across browser refreshes, app restarts, and server redeploys
- Multiple named users can coexist in the same database with zero friction
- No authentication overhead — username is the sole identifier, consistent with the personal-use design philosophy

---

## 4. Infrastructure & Deployment

### 4.1 Stack Overview

```
User browser
    │
    ▼
dosetrack.dis-solved.com  (Cloudflare DNS — proxied CNAME)
    │
    ▼
Cloudflare Pages (dosetrack-redirect project)
    │  _redirects: /* → https://dosetrack-v4.streamlit.app/:splat 301
    ▼
Streamlit Community Cloud
    │  abinittio/dosetrack-v4 · branch: main · entry: app.py
    │
    ├──── GitHub (source of truth — auto-deploys on push to main)
    │
    └──── Supabase Postgres (West Europe — London)
               db.bqehurxkfcnipvkpuwvp.supabase.co:5432
```

### 4.2 Database — Supabase Postgres

Supabase provides a managed PostgreSQL instance (PostgreSQL 15) with:
- **Connection:** `psycopg2-binary` via `DATABASE_URL` stored in Streamlit secrets
- **Region:** West Europe (London) — collocated with Streamlit's GCP europe-west2 region for low latency
- **Persistence:** fully durable, survives app restarts, redeploys, and Streamlit Cloud instance recycling
- **Provisioning:** project created via Supabase Management API; table initialised on first app load via `CREATE TABLE IF NOT EXISTS`

The `DATABASE_URL` secret is injected at runtime via Streamlit's secrets management, never committed to source control (`.streamlit/secrets.toml` is gitignored).

### 4.3 Hosting — Streamlit Community Cloud

The app is deployed on Streamlit Community Cloud, which provides:
- **Zero-config Python hosting** — no Dockerfile, no server configuration
- **GitHub integration** — auto-redeploy on push to `main`, after CI passes
- **Secrets management** — encrypted key-value store for `DATABASE_URL`
- **Always-on** — no cold start penalty for personal use scale

### 4.4 DNS & Routing — Cloudflare

`dosetrack.dis-solved.com` is served via a two-layer Cloudflare setup:

**Layer 1 — DNS:** A proxied CNAME record on `dis-solved.com` (managed in Cloudflare) points `dosetrack` to the Cloudflare Pages project, routing traffic through Cloudflare's global edge network.

**Layer 2 — Redirect:** A Cloudflare Pages project (`dosetrack-redirect`) serves a `_redirects` rule:

```
/* https://dosetrack-v4.streamlit.app/:splat 301
```

This issues a permanent 301 redirect to the Streamlit app URL, with SSL termination and CDN caching handled by Cloudflare. The Pages project is configured as a custom domain on `dosetrack.dis-solved.com`, with TLS provisioned automatically via Google Trust Services.

---

## 5. CI/CD Pipeline

### 5.1 Pipeline Overview

```
git push → main
    │
    ├─── [lint] ruff check app.py tests/
    │         └─ fail fast on style or undefined name errors
    │
    ├─── [test] matrix: Python 3.10 / 3.11 / 3.12 (parallel)
    │         ├─ pytest tests/test_engine.py
    │         ├─ pytest-cov (fail if coverage < 60%)
    │         └─ upload .coverage artifact (3.11 run only)
    │
    └─── [deploy] notify (Streamlit Cloud auto-deploys from main)
```

All three jobs run on `ubuntu-latest`. The `test` job requires `lint` to pass. The `deploy` job requires `test` to pass and only runs on `push` to `main` (not on pull requests).

### 5.2 Lint — ruff

**ruff** is a Rust-based Python linter, approximately 100× faster than flake8. Configured via `ruff.toml`:

```toml
select = ["E", "F", "W"]
ignore = ["E501"]          # line length not enforced
exclude = ["dosetrack/"]   # engine module excluded (library code)
```

Rule sets:
- `E` — pycodestyle errors (whitespace, indentation, syntax)
- `F` — pyflakes (undefined names, unused imports, redefined variables)
- `W` — warnings (deprecated syntax, whitespace warnings)

The `dosetrack/` engine is excluded because it is a ported library module with its own conventions, and its imported-but-unused symbols serve as a documented public API surface.

### 5.3 Matrix Testing

Tests run in parallel across **Python 3.10, 3.11, and 3.12** using GitHub Actions matrix strategy. This verifies:
- No use of version-specific syntax or deprecated APIs
- Consistent behaviour of numpy and psycopg2 across Python minor versions
- Forward compatibility with the upcoming Python 3.12 standard

The test suite (`tests/test_engine.py`) covers:

| Test | What it verifies |
|------|-----------------|
| `test_single_dose` | Simulation runs, effect rises after a single dose |
| `test_two_doses_cumulative` | Second dose produces higher peak than first alone |
| `test_da_stores_deplete` | 7-day consecutive dosing depletes S below 0.8 |
| `test_db_operations` | Insert, load, delete round-trip on SQLite backend |

### 5.4 Coverage

**pytest-cov** instruments the engine during test runs. The pipeline fails if coverage of tested modules drops below **60%**. The threshold is set conservatively because `dosetrack/` includes V3 analysis functions (multi-day plotting, redose heuristics, AUC analysis) not exercised by the V4 minimal UI test suite.

The `.coverage` artefact is uploaded from the Python 3.11 run for offline inspection via `coverage html`.

### 5.5 Continuous Deployment

Streamlit Community Cloud monitors the `abinittio/dosetrack-v4` repository. On every push to `main` that passes the CI pipeline, Streamlit Cloud pulls the latest commit, installs `requirements.txt`, and restarts the app — with zero downtime for the database (Supabase is external).

The deploy job in `ci.yml` serves as a notification gate: it runs a no-op `echo` step that only executes after tests pass, ensuring the CI badge reflects deployment readiness, not just test status.

---

## 6. Dependencies

### Runtime (`requirements.txt`)

| Package | Version | Role |
|---------|---------|------|
| streamlit | ≥ 1.30 | Web UI framework — reactive Python to browser |
| numpy | ≥ 1.24 | Array operations, ODE state vectors |
| matplotlib | ≥ 3.7 | PK/PD curve rendering via Agg backend |
| psycopg2-binary | ≥ 2.9 | PostgreSQL adapter (Supabase production DB) |

### Dev (`requirements-dev.txt`)

| Package | Version | Role |
|---------|---------|------|
| pytest | ≥ 7.4 | Test runner |
| pytest-cov | ≥ 4.1 | Coverage measurement and reporting |
| ruff | ≥ 0.4 | Fast Python linter |

**Zero scientific dependencies in the engine.** The `dosetrack/` module requires only `numpy`. scipy, RDKit, and PyTorch are not used. The RK4 integrator is implemented from scratch in approximately 30 lines of numpy — this is intentional, making the simulation fully portable and auditable without a complex dependency chain.

---

## 7. References

Ermer, J. C., Shoaf, S. E., Fogle, R. H., Bieck, P. R., & Pauer, L. (2010). Lisdexamfetamine dimesylate: linear dose-proportionality, low intersubject and intrasubject variability, and safety in an open-label single-dose pharmacokinetic study in healthy adult volunteers. *Journal of Clinical Pharmacology*, 50(9), 1001–1010. https://doi.org/10.1177/0091270009354994

Pennick, M. (2010). Absorption of lisdexamfetamine and its conversion to d-amphetamine. *Neuropsychiatric Disease and Treatment*, 6, 317–327. https://doi.org/10.2147/NDT.S9749

Rowland, M., & Tozer, T. N. (2011). *Clinical Pharmacokinetics and Pharmacodynamics: Concepts and Applications* (4th ed.). Lippincott Williams & Wilkins.

Gabrielsson, J., & Weiner, D. (2010). *Pharmacokinetic and Pharmacodynamic Data Analysis: Concepts and Applications* (4th ed.). Swedish Pharmaceutical Press.

---

*DoseTrack V4 — [dis-solved.com](https://dis-solved.com) — For personal and educational use only. Not medical advice.*
