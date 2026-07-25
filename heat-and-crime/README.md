# Heat and Crime — Python walkthrough (Chicago city-day panel)

**Story / charts:** https://along777.github.io/along/heat-and-crime/  
**This folder:** code + cached data. Follow the session below to reproduce the headline yourself.

| | |
|---|---|
| Window | 2015-01-01 … 2025-12-31 · **4,018** days |
| Crimes | **2,757,885** (SoQL day×type aggregates, not raw incidents) |
| Same-day violent | **+5.6%** [+5.1, +6.0] per +10°F daily max |
| Property | +3.7% [+3.2, +4.2] |
| Hot-spell cumulative (L\*=1) | **+6.1%** [+5.6, +6.6] |
| Battery only | +6.0% [+5.5, +6.5] |

**Identification in one line:** inside each month×year cell, residual daily max temperature is treated as weather noise; report `100 * expm1(β)` from log daily counts with HAC(9).

---

## Mental model

```text
log C_t = β (T_t / 10) + α_month×year + δ_dow + γ' W_t + ε_t
```

| Piece | Meaning |
|---|---|
| `C_t` | Daily count (violent, property, battery, …) |
| `T_t / 10` | Daily max °F in “per +10°F” units (`tmax10`) |
| `α_month×year` | Kills “it was July 2019” (season + year) |
| `δ_dow` | Monday…Sunday patterns |
| `W_t` | holiday, rain (≥1 mm), snow day |
| `β` | Log-point effect; publish `100 * expm1(β)` |

If you only regress crime on temperature without month×year FE, you measure summer, not heat.

---

## Repo map (open these)

| Path | What it is |
|---|---|
| `examples/headline_fit.py` | **Start here.** Minimal path to +5.6%. |
| `examples/leads_placebo.py` | Future-heat check (F1). |
| `panel.py` | Single panel builder (`build_panel`). |
| `models.py` | `fit_ols`, `pct_ci`, `fit_dlag`, `bh_fdr`. |
| `run_model_lab.py` | G0–G7 selection + F1–F7 attacks → JSON + figures. |
| `run_eda_diagnostics.py` | Residual EDA plots. |
| `data/` | Cached crime/weather + MANIFEST + lab results. |
| `index.html` | Public article (not a notebook dump). |

---

## Session: follow along in order

### 0) Environment

```bash
cd heat-and-crime   # or heat/
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Offline if `data/` is present (it is, in this package).

### 1) One command → headline numbers

```bash
python examples/headline_fit.py
```

**Expected shape of output:**

```text
panel: 4018 days | mean violent/day=212.5
controls: C(ym) + C(dow) + holiday + rain + snow_day
HAC maxlags: 9

total       +4.0% [+3.5, +4.4]
violent     +5.6% [+5.1, +6.0]
property    +3.7% [+3.2, +4.2]
battery     +6.0% [+5.5, +6.5]
```

That script is the entire public estimator path: `build_panel` → `fit_ols` → `pct_ci`.

### 2) What `build_panel()` actually builds

```python
from panel import build_panel

dfa, dffull = build_panel()
print(dfa.shape)
print(dfa[["violent", "property", "battery", "tmax", "tmax10"]].head())
print([c for c in dfa.columns if "tmax10" in c])
```

You should see lags `tmax10_L1…L7` and **leads** `tmax10_F1…F3` (placebos).

**Inside `panel.py` (read the file):**

```python
# 1) map types
crime["cat"] = np.select(
    [crime["primary_type"].isin(VIOLENT), crime["primary_type"].isin(PROPERTY)],
    ["violent", "property"],
    default="other",
)

# 2) day x category matrix
daily = crime.pivot_table(index="day", columns="cat", values="n", aggfunc="sum", fill_value=0)
daily["total"] = daily.sum(axis=1)

# 3) complete calendar (no missing days at city scale)
cal = pd.date_range("2010-01-01", "2025-12-31", freq="D")
daily = daily.reindex(cal).fillna(0).astype(int)

# 4) join weather 1:1 on local date; assert no tmax gaps
# 5) features: tmax10, rain, snow, ym, holiday, unrest, lags, leads, heat index
```

Crime never arrives as 2.8M incident rows. SoQL did the aggregate:

```python
params = {
    "$select": "date_trunc_ymd(date) AS day, primary_type, count(*) AS n",
    "$where": "date >= '2010-01-01' AND date < '2026-01-01'",
    "$group": "day, primary_type",
    "$limit": 50000,
    "$offset": offset,
}
```

### 3) How estimation is wired (`models.py`)

```python
from models import fit_ols, pct_ci, fmt_pct
from panel import CTRL_M2, HAC_DEFAULT

# fit_ols is intentionally thin:
# smf.ols("np.log(y) ~ x + C(ym)+C(dow)+holiday+rain+snow_day").fit(
#     cov_type="HAC", cov_kwds={"maxlags": 9}
# )

res = fit_ols("violent", "tmax10", dfa, hac=HAC_DEFAULT)
print(res.params["tmax10"])           # log-point β
print(fmt_pct(pct_ci(res, "tmax10"))) # published percent + CI
```

**CTRL_M2 term-by-term:**

| Term | Job |
|---|---|
| `C(ym)` | Absorb month×year (season + year shocks) |
| `C(dow)` | Weekend vs weekday |
| `holiday` | US federal holidays (observed) |
| `rain` / `snow_day` | Wet-weather confounds with temp and street activity |

**Percent transform:**

```python
# models.pct_from_b
pct = 100 * np.expm1(beta)          # not 100*beta
# CI endpoints transformed the same way (not delta-method on percent scale)
```

### 4) Components (is “violent” just a label?)

```python
for y in ["battery", "assault", "robbery", "theft", "violent", "property"]:
    use = dfa[dfa[y] > 0]
    r = fit_ols(y, "tmax10", use, hac=9)
    print(f"{y:10s} {fmt_pct(pct_ci(r, 'tmax10'))}")
```

Expect battery ~+6.0% (main mass of violent), theft ~+3.4% (property-like).

### 5) Leads placebo (F1)

```bash
python examples/leads_placebo.py
```

```python
# same idea, inline:
use = dfa.dropna(subset=["tmax10_F1", "tmax10_F2", "tmax10_F3"])
r = fit_ols("violent", "tmax10 + tmax10_F1 + tmax10_F2 + tmax10_F3", use, hac=9)
# same-day large; leads small  → reverse causality less plausible
```

### 6) Distributed lag (hot spells)

```python
from models import fit_dlag

res, terms, cum = fit_dlag("violent", L=1, data=dfa, hac=9)
print(terms)                      # ['tmax10', 'tmax10_L1']
print(fmt_pct(pct_ci(res, "tmax10")))
print(fmt_pct(cum))               # cumulative ~ +6.1%
```

Lab G3 picks `L*` by AIC over L=0…7 (violent L\*=1, property L\*=4).

### 7) Full lab (selection + attacks)

```bash
python run_model_lab.py
# writes data/model_lab_results.json + figures/g*.png + f5_permutation.png
```

```python
import json
from pathlib import Path

lab = json.loads(Path("data/model_lab_results.json").read_text(encoding="utf-8"))
s = lab["summary"]
print(s["baseline_m2_violent_same_day"])
print(s["preferred_hac_maxlags"])       # 9
print(s["dlag_cumulative_violent"])     # ~+6.1%
print(s["kink_tau_violent"])            # 80
print(s["falsification"])               # F1/F2/F5/F6/F7 flags
```

**Lab blocks (read the source functions in `run_model_lab.py`):**

| Block | Function theme | You care because |
|---|---|---|
| G0 | baseline lock | Asserts ~+5.6% still there |
| G1 | HAC lag grid | Default lag 9 not fiat |
| G2 | treatment race + **per 1 within-ym SD** | tmax vs tmean on a fair scale |
| G3 | DLAG AIC | cumulative spell effect |
| G4 | spline df + kink τ | shape / 80°F plateau |
| G5 | LOYO + rolling | no single year owns the result |
| F1–F7 | leads, battery, HI, inference, permutation, influence, FDR | attack surface |

**Permutation (F5) idea:** demean log(violent) and tmax10 within `ym`, shuffle x inside each `ym` 300 times, compare residual slopes. Observed slope is extreme (`p ≈ 0`). See `f5_permutation` in `run_model_lab.py` and `figures/f5_permutation.png`.

### 8) EDA residuals

```bash
python run_eda_diagnostics.py
# figures/eda_leads_residual_corr.png
# figures/eda_type_residual_gradients.png
```

---

## Design choices (why this stack)

| Choice | Why |
|---|---|
| SoQL day×type counts | Laptop scale; forces honest city-day design |
| Month×year FE | Removes seasonality costume |
| Daily **max** temp | Public ruler; daytime/evening peak |
| Log OLS + HAC | Percent effects + serial correlation |
| Poisson/NB check | Not an artifact of logging |
| Leads + permutation | Cheap reverse-causality / null checks |
| No ML for headline β | This is reduced-form ID, not a Kaggle score |

**tmax vs tmean:** tmean can look “larger” per +10°F because means move less. G2 reports **per residual SD** so the rulers are comparable. Public treatment stays daily max.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError` on data/ | You are not in `heat-and-crime/` (or `heat/`) as cwd |
| MANIFEST hash mismatch (notebook) | Intentional after re-download; rewrite MANIFEST after refresh |
| Import errors on `panel` | `sys.path` — run examples from package root as shown |
| Windows venv | `.venv\Scripts\activate` then `pip install -r requirements.txt` |
| Lab slow | Normal (many FE fits); results already in `data/model_lab_results.json` |

---

## Layout

```
heat-and-crime/   (GitHub + local package)
  README.md                 # this walkthrough
  examples/headline_fit.py
  examples/leads_placebo.py
  panel.py
  models.py
  run_model_lab.py
  run_eda_diagnostics.py
  requirements.txt
  data/                     # caches + lab JSON + MANIFEST
  figures/                  # lab + article charts
  index.html                # public article
```

---

## Sources

- Crime: [Chicago Data Portal `ijzp-q8t2`](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- Weather: [Open-Meteo Historical / ERA5](https://open-meteo.com/en/docs/historical-weather-api) at 41.88, −87.63, America/Chicago, °F
