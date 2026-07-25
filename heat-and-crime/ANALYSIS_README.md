# Heat and Crime (Chicago, 2015–2025)

**Live article:** [along777.github.io/along/heat-and-crime](https://along777.github.io/along/heat-and-crime/)

Technical walkthrough of a city-day fixed-effects study: daily max temperature and reported crime. Public data only. No API keys.

| Quantity | Estimate |
|---|---|
| Crimes in window | 2,757,885 across 4,018 days |
| Same-day violent effect | **+5.6%** [+5.1, +6.0] per +10°F daily max |
| Property | +3.7% [+3.2, +4.2] |
| Hot-spell cumulative (L\*=1) | **+6.1%** [+5.6, +6.6] |
| Battery only | +6.0% [+5.5, +6.5] |
| LOYO band (violent) | ~+5.4% to +5.7% |

Identification: compare days **inside the same month×year**, with day-of-week, federal holidays, rain, and snow controlled. Report percent effects as `100 * expm1(β)` on log daily counts. Errors: Newey-West HAC (default lag 9).

---

## Why this project

Heat and crime is an old Freakonomics-style question: weather as a clean daily shock while most crime drivers move slowly. The fun is not rediscovering “summer has more crime.” The fun is re-running that idea with:

- a full **2015–2025** Chicago panel from free public APIs  
- **server-side** SoQL aggregation so the project runs on a laptop  
- modern **month×year FE + HAC**, count-model checks, distributed lags, and an explicit **falsification suite**

Build split (July 2026): **Claude Fable 5** (max) started the blank repo in two heavy prompts. **Grok 4.5** finished audit, lab, stress tests, modules, and the public article. Different tools, different tasks. Not better or worse.

---

## Pipeline walkthrough

End-to-end path from raw APIs to published claim.

### Stage 1. Ingest

| Input | How |
|---|---|
| Crime | Socrata `ijzp-q8t2`, SoQL `date_trunc_ymd(date), primary_type, count(*)` grouped, paginated |
| Weather | Open-Meteo ERA5 archive at Loop (41.88, −87.63), America/Chicago, °F |

Crime is **not** downloaded as 2.8M incident rows. The portal returns ~125k day×type aggregates. That is the load-bearing engineering choice.

```python
params = {
    "$select": "date_trunc_ymd(date) AS day, primary_type, count(*) AS n",
    "$where": "date >= '2010-01-01' AND date < '2026-01-01'",
    "$group": "day, primary_type",
    "$order": "day, primary_type",
    "$limit": 50000,
    "$offset": offset,
}
```

Fetch window: 2010–2025 (extra years for robustness). Analysis window: **2015-01-01 → 2025-12-31**.

### Stage 2. Validate and lock

`data/validation_counts.json` stores independent `count(*)` checks (spot months, annual totals, THEFT probes) plus cache integrity fields. `data/MANIFEST.json` stores SHA-256 hashes. Re-download without rewriting the manifest fails on purpose.

### Stage 3. Build the panel (`panel.py`)

Single source of truth. Both the notebook path and the lab import this.

```python
from panel import build_panel, CTRL_M2, HAC_DEFAULT

dfa, dffull = build_panel()
# dfa: 4018 days, 2015–2025
# dffull: 2010–2025 for extended robustness
```

What `build_panel()` does:

1. Map primary types → violent / property / other (CSA dual spellings merged into violent)  
2. Also emit **component** outcomes: battery, assault, robbery, theft, homicide  
3. Full calendar reindex (no missing days)  
4. Join weather 1:1 on local date  
5. Features: `tmax10`, `tmean10`, `app_tmax10`, heat index `hi10`, rain/snow flags, federal holidays, COVID/unrest flags  
6. Lags `tmax10_L1..L7` and **leads** `tmax10_F1..F3` for placebo tests  
7. Keys: `ym` (month×year FE), `week_id` (cluster option)

```python
VIOLENT = [
    "BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE",
    "CRIM SEXUAL ASSAULT", "CRIMINAL SEXUAL ASSAULT",
]
PROPERTY = ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE"]
CTRL_M2 = "C(ym) + C(dow) + holiday + rain + snow_day"
HAC_DEFAULT = 9
```

### Stage 4. EDA diagnostics (`run_eda_diagnostics.py`)

Not the headline. Checks that residual heat still tracks residual crime after demeaning by `ym`, and that future heat is weaker than same-day heat.

```bash
python run_eda_diagnostics.py
# figures/eda_leads_residual_corr.png
# figures/eda_type_residual_gradients.png
# data/eda_diagnostics.json
```

### Stage 5. Baseline model ladder (notebook M1–M6)

Workhorse (M2):

```text
log C_t = β (T_t / 10) + α_month×year + δ_dow + γ′W_t + ε_t
```

```python
from models import fit_ols, pct_ci, fmt_pct

res = fit_ols("violent", "tmax10", dfa, hac=9)
print(fmt_pct(pct_ci(res, "tmax10")))
# +5.6% [+5.1, +6.0]
```

| Spec | Role |
|---|---|
| M1 | DOW + month + year (not interacted) |
| **M2** | **Month×year FE (headline)** |
| M3 | Poisson QMLE + Negative Binomial |
| M4 | 10°F bins + natural cubic spline (df selected) |
| M5 | Stacked violent vs property, day-clustered interaction |
| M6 | Hottest-decile dummy; summer-only ≥90°F vs 70s |

### Stage 6. Model lab G0–G7 (`run_model_lab.py`)

Selection and robustness that used to be fiat knobs.

```bash
python run_model_lab.py
# data/model_lab_results.json
# figures/g1_*.png … g5_*.png
```

| Block | Decision |
|---|---|
| G0 | Lock baseline ~+5.6% violent (tolerance check) |
| G1 | HAC lag grid → preferred **9** |
| G2 | Treatment race: tmax / app / tmean / heat index, **per +10°F and per 1 within-ym SD** |
| G3 | Distributed lag L=0…7 by AIC → violent **L\*=1**, cum **+6.1%** |
| G4 | Spline df grid + kink τ search → violent **τ\*=80°F** |
| G5 | Leave-one-year-out + rolling 3-year windows |
| G6 | Poisson + week-clustered OLS parity |
| G7 | Residual ridge CV (not a causal headline) |

### Stage 7. Falsification F1–F7 (same lab)

| ID | Test | Pass criterion (what we saw) |
|---|---|---|
| F1 | Future heat leads | Lead coeffs << same-day |
| F2 | Battery / assault / robbery / theft | Battery +6.0%, large |
| F3 | Heat index race | Per-SD effects similar; tmax stays public |
| F4 | HAC-9 / HAC-14 / week / HC1 | All print +5.6% |
| F5 | Shuffle tmax within `ym` (B=300) | p ≈ 0 |
| F6 | Drop unrest, NYD, top 1%, 2020 | \|Δ\| < 0.1 pp |
| F7 | Benjamini-Hochberg on multi outcomes | Key series still clear |

### Stage 8. Public article

`index.html` + `figures/` (hand-written). Not a Jupyter export. Deployed as `along/heat-and-crime/`.

---

## Design choices (why these, not others)

| Choice | Why |
|---|---|
| SoQL aggregation | Laptop-reproducible; forces city-day design honesty |
| Month×year FE | Kills season and year shocks; uses within-month weather noise (~9°F mean within-month SD) |
| Daily **max** temp | Daytime/evening peak; literature standard; public ruler |
| Log OLS + HAC | Percent effects; serial correlation on daily counts |
| Poisson/NB check | Result must not depend on logging counts |
| No ML for headline β | This is reduced-form identification, not a forecasting contest |
| Leads + permutation | Cheap reverse-causality and null-cloud checks |

**tmax vs tmean:** tmean can win AIC and look “bigger” per +10°F because means move less. On a **within-month SD** scale, tmax and tmean sit close. Public treatment stays daily max.

---

## Reproduce

```bash
cd heat
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_model_lab.py
python run_eda_diagnostics.py

# optional full notebook
jupyter nbconvert --to notebook --execute --inplace heat_and_crime.ipynb

# local article
python -m http.server 8000
# http://127.0.0.1:8000/
```

With `data/` present, everything is offline. Delete caches only for a full re-download.

---

## Outputs map (what is truth for what)

| Claim | Source of truth |
|---|---|
| Same-day M2 effects | Notebook M2 / lab G0 |
| Cumulative spell | Lab G3 `summary.dlag_cumulative_violent` |
| Kink / spline | Lab G4 |
| Falsification pass/fail | Lab `summary.falsification` + F\* blocks |
| Public narrative | `index.html` |
| Cache integrity | `data/MANIFEST.json` |

```python
import json
lab = json.loads(open("data/model_lab_results.json", encoding="utf-8").read())
print(lab["summary"]["baseline_m2_violent_same_day"])
print(lab["summary"]["dlag_cumulative_violent"])
print(lab["summary"]["falsification"])
print(lab["F2_battery"]["battery"])
```

---

## Layout

```
heat/
  index.html                 # public article
  figures/                   # article + lab charts
  README.md                  # this walkthrough
  requirements.txt
  panel.py                   # panel construction
  models.py                  # FE / DLAG / FDR helpers
  run_model_lab.py           # G0–G7 + F1–F7
  run_eda_diagnostics.py
  heat_and_crime.ipynb       # full executed science notebook
  data/
    chicago_crime_daily_by_type.csv
    chicago_weather_daily.csv
    validation_counts.json
    MANIFEST.json
    model_lab_results.json
    eda_diagnostics.json
```

Optional internal notes: `PEER_REVIEW.md`, `SCIENCE_A_PLUS.md`, `REVIEW_*.md`.

---

## Data sources

| | |
|---|---|
| Crime | [Chicago Data Portal `ijzp-q8t2`](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) |
| Weather | [Open-Meteo Historical / ERA5](https://open-meteo.com/en/docs/historical-weather-api) |

Public data. Cite sources if you republish numbers.
