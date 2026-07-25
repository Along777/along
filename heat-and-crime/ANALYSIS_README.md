# Heat and Crime (Chicago city-day panel)

**Article (narrative):** https://along777.github.io/along/heat-and-crime/

This README is the **code path**. Clone the repo, open a Python shell in `heat/`, and follow the steps in order. You should be able to reproduce the headline **+5.6% violent per +10°F** without reading the notebook first.

| | |
|---|---|
| Window | 2015-01-01 … 2025-12-31 (4,018 days) |
| Crimes | 2,757,885 reported incidents (aggregated) |
| Same-day violent | **+5.6%** [+5.1, +6.0] per +10°F daily max |
| Property | +3.7% [+3.2, +4.2] |
| Hot-spell cumulative (L\*=1) | **+6.1%** [+5.6, +6.6] |
| Battery only | +6.0% [+5.5, +6.5] |

---

## 0. Mental model (30 seconds)

**Object:** one row per calendar day for Chicago.

**Treatment:** daily maximum temperature in units of 10°F (`tmax10 = tmax / 10`).

**Outcome:** log of daily counts (`log(violent)`, `log(property)`, …).

**Identification:** month×year fixed effects absorb “it was July 2019.” Day-of-week, holiday, rain, and snow soak up calendar and wet-weather confounds. What is left is day-to-day temperature noise *inside* a month. That residual heat is treated as as-good-as-random weather.

**Report:** percent change ≈ `100 * (exp(β) - 1)` from the log model. Implemented as `100 * expm1(β)`.

```text
log C_t = β · (T_t / 10) + α_{month×year} + δ_{dow} + γ' W_t + ε_t
```

---

## 1. Environment

```bash
cd heat
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

With `data/` present, **no network** is required. Caches are public-API downloads locked by `data/MANIFEST.json`.

---

## 2. Follow-along: from disk to +5.6%

Run these in a REPL or a scratch script from `heat/` as cwd.

### Step 1 — Build the panel

```python
from panel import build_panel, CTRL_M2, HAC_DEFAULT, WIN_START, WIN_END

dfa, dffull = build_panel()
print(dfa.shape)                    # (4018, …)
print(WIN_START, WIN_END)
print(dfa[["total", "violent", "property", "battery", "tmax"]].mean())
print([c for c in dfa.columns if c.startswith("tmax10")])
# expect tmax10, tmax10_L1..L7, tmax10_F1..F3, tmax10_roll3, ...
```

**What `panel.build_panel()` does (read `panel.py`):**

1. Load `data/chicago_crime_daily_by_type.csv` (SoQL day×type counts, not raw incidents).
2. Map types → violent / property / other; also split **battery, assault, robbery, theft**.
3. Reindex to a complete daily calendar (assert no missing days).
4. Join ERA5 weather (`chicago_weather_daily.csv`) on local date.
5. Build FE keys: `ym`, `dow`, `holiday`, `rain`, `snow_day`, `week_id`.
6. Build lags (dynamics) and **leads** (placebo).

Crime was aggregated at the API so you never materialize ~2.8M incident rows:

```python
# conceptual Socrata params (notebook fetch path)
{
    "$select": "date_trunc_ymd(date) AS day, primary_type, count(*) AS n",
    "$group": "day, primary_type",
    "$where": "date >= '2010-01-01' AND date < '2026-01-01'",
}
```

### Step 2 — Controls string (every term earns its seat)

```python
print(CTRL_M2)
# C(ym) + C(dow) + holiday + rain + snow_day
```

| Term | Job |
|---|---|
| `C(ym)` | Kill season + year shocks (the “summer” problem) |
| `C(dow)` | Weekend vs weekday crime patterns |
| `holiday` | Federal holidays (US, observed) |
| `rain` / `snow_day` | Precipitation confounds with temp and outdoor activity |

Default HAC lag:

```python
print(HAC_DEFAULT)  # 9  (selected in Model Lab G1; 14 kept as sensitivity)
```

### Step 3 — Headline fit (this is the public number)

```python
from models import fit_ols, pct_ci, fmt_pct

res = fit_ols("violent", "tmax10", dfa, hac=9)
print(res.params["tmax10"])
print(fmt_pct(pct_ci(res, "tmax10")))
# expect something like: +5.6% [+5.1, +6.0]
```

`models.fit_ols` is thin on purpose:

```python
# models.py (conceptual)
smf.ols(f"np.log({yvar}) ~ {xterms} + {CTRL_M2}", data=data).fit(
    cov_type="HAC", cov_kwds={"maxlags": hac}
)
```

`pct_ci` turns log-points into percent:

```python
# 100 * expm1(β), with HAC CI endpoints transformed the same way
```

Repeat for property / total / battery:

```python
for y in ["total", "violent", "property", "battery"]:
    r = fit_ols(y, "tmax10", dfa[dfa[y] > 0] if y == "battery" else dfa, hac=9)
    print(y, fmt_pct(pct_ci(r, "tmax10")))
```

### Step 4 — Same design, count models (sanity)

In the notebook (M3) you will see Poisson QMLE / NegBin under the same CTRL_M2. Point estimates sit on top of log-OLS (~+5.5% violent). If log-OLS and Poisson disagree wildly, stop and debug the panel.

### Step 5 — Run the lab (selection + attacks)

```bash
python run_model_lab.py
python run_eda_diagnostics.py
```

Read the machine-readable results:

```python
import json
from pathlib import Path

lab = json.loads(Path("data/model_lab_results.json").read_text(encoding="utf-8"))
s = lab["summary"]
print(s["baseline_m2_violent_same_day"])
print(s["preferred_hac_maxlags"])          # 9
print(s["dlag_L_star_violent"])            # 1
print(s["dlag_cumulative_violent"])        # ~+6.1%
print(s["kink_tau_violent"])               # 80
print(s["falsification"])                  # all True / p~0
```

**What to open in the JSON if you are auditing:**

| Key | Meaning |
|---|---|
| `G0_baseline` | Lock to ~+5.6% violent |
| `G1_hac` | Bandwidth grid + preferred lag |
| `G2_treatment` | tmax / app / tmean / heat index; **per +10°F and per 1 within-ym SD** |
| `G3_dlag` | Lag order by AIC; cumulative effect |
| `G4_shape` | Spline df* + kink τ* |
| `G5_stability` | Leave-one-year-out + rolling windows |
| `F1_leads` | Future heat should not carry the same-day effect |
| `F5_permutation` | Shuffle `tmax` inside each `ym` (B=300); observed residual slope extreme |
| `F2_battery` | Component outcomes |

**Figures produced:**

- `figures/g1_hac_bandwidth.png` … `g5_stability.png`
- `figures/f5_permutation.png`
- `figures/eda_leads_residual_corr.png`, `eda_type_residual_gradients.png`
- `figures/article_*.png` (used by the HTML article)

### Step 6 — Two falsification ideas you should re-derive by hand

**Leads (F1):** include future temperature. Same-day should dominate.

```python
use = dfa.dropna(subset=["tmax10_F1", "tmax10_F2"])
r = fit_ols("violent", "tmax10 + tmax10_F1 + tmax10_F2", use, hac=9)
print("same", fmt_pct(pct_ci(r, "tmax10")))
print("lead1", fmt_pct(pct_ci(r, "tmax10_F1")))
print("lead2", fmt_pct(pct_ci(r, "tmax10_F2")))
# expect same ~+5%, leads small
```

**Distributed lag (G3 idea):** AIC picks L*=1 for violence; cumulative ~+6.1%.

```python
from models import fit_dlag
res, terms, cum = fit_dlag("violent", L=1, data=dfa, hac=9)
print(terms)
print(fmt_pct(pct_ci(res, "tmax10")))
print(fmt_pct(cum))
```

### Step 7 — Full narrative notebook (optional)

```bash
jupyter nbconvert --to notebook --execute --inplace heat_and_crime.ipynb
```

Notebook owns the M1–M6 writeup, plots, and the integrated lab/falsification sections. Lab scripts own the selectable knobs and attack suite.

---

## Module map

| File | Role | Start here |
|---|---|---|
| `panel.py` | Daily panel construction | `build_panel()` |
| `models.py` | FE OLS/HAC, cluster, DLAG, BH-FDR | `fit_ols`, `pct_ci`, `fit_dlag` |
| `run_model_lab.py` | G0–G7 selection + F1–F7 attacks | `main()` |
| `run_eda_diagnostics.py` | Residual leads + type gradients | `main()` |
| `heat_and_crime.ipynb` | Full science narrative | top → bottom |
| `index.html` | Public article | browser |
| `data/model_lab_results.json` | Machine-readable lab output | `summary`, `F*`, `G*` |

---

## Spec cheatsheet

```python
# panel.py
WIN_START, WIN_END = "2015-01-01", "2025-12-31"
CTRL_M2 = "C(ym) + C(dow) + holiday + rain + snow_day"
HAC_DEFAULT = 9

VIOLENT = [
    "BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE",
    "CRIM SEXUAL ASSAULT", "CRIMINAL SEXUAL ASSAULT",  # dual spellings
]
PROPERTY = ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE"]
```

**tmax vs tmean:** tmean can win AIC and look larger *per +10°F* because daily means move less than maxes. Lab G2 reports **per 1 within-ym SD** so rulers are comparable. Public treatment stays **daily max**.

---

## Layout

```
heat/
  panel.py / models.py
  run_model_lab.py
  run_eda_diagnostics.py
  heat_and_crime.ipynb
  index.html + figures/     # public write-up
  data/                     # caches + MANIFEST + lab JSON
  requirements.txt
```

## Sources

- Crime: [Chicago Data Portal `ijzp-q8t2`](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- Weather: [Open-Meteo Historical / ERA5](https://open-meteo.com/en/docs/historical-weather-api)
