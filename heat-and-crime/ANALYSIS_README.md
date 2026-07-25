# Heat and Crime (Chicago, 2015–2025)

**Live article:** https://along777.github.io/along/heat-and-crime/

City-day fixed effects: daily max temperature and reported crime. Public data. No API keys.

| | |
|---|---|
| Crimes / days | 2,757,885 / 4,018 |
| Same-day violent | **+5.6%** [+5.1, +6.0] per +10°F |
| Property | +3.7% [+3.2, +4.2] |
| Hot-spell cum. (L*=1) | **+6.1%** [+5.6, +6.6] |
| Battery only | +6.0% [+5.5, +6.5] |

**Identification:** within month×year cells; controls for day-of-week, holidays, rain, snow. Percent effect = `100 * expm1(β)` on log daily counts. HAC lag 9.

**Why this project:** classic Freakonomics-style heat/crime question, re-run with modern open APIs and a full attack suite. **Build:** Claude Fable 5 (max) started the blank repo in two prompts. Grok 4.5 finished audit, lab, falsification, and the article. Different tools, different tasks.

---

## Pipeline (what the code actually does)

1. **Ingest** – Socrata SoQL aggregates crime to day × type (`count(*)`), not 2.8M raw rows. Open-Meteo ERA5 weather at the Loop.
2. **Validate** – Independent `count(*)` checks + SHA-256 `data/MANIFEST.json`.
3. **Panel** – `panel.build_panel()`: categories, battery split, lags/leads, heat index, FE keys.
4. **EDA** – `run_eda_diagnostics.py`: residual leads, type gradients.
5. **Baseline** – Notebook M1–M6: FE ladder, counts, shape, gap, summer heatwave.
6. **Lab** – `run_model_lab.py` G0–G7: HAC, DLAG, kink, LOYO, treatment race.
7. **Falsify** – F1–F7: leads, components, heat index, inference, permutation, influence, FDR.
8. **Article** – `index.html` (hand-written; not a notebook export).

---

## Design choices

- **SoQL aggregation** – laptop-reproducible city-day panel.
- **Month×year FE** – kills season/year; uses within-month weather noise (~9°F mean SD).
- **Daily max temp** – public treatment; tmean can look “bigger” per +10°F only because means move less (check per-1SD race in lab).
- **Log OLS + HAC** – percent effects + serial correlation; Poisson/NB as check.
- **No ML for headline β** – reduced-form ID, not a forecasting contest.

---

## Code (three blocks)

**SoQL (crime download concept):**

```python
params = {
    "$select": "date_trunc_ymd(date) AS day, primary_type, count(*) AS n",
    "$where": "date >= '2010-01-01' AND date < '2026-01-01'",
    "$group": "day, primary_type",
    "$limit": 50000,
    "$offset": offset,
}
```

**Headline fit:**

```python
from panel import build_panel
from models import fit_ols, pct_ci, fmt_pct

dfa, _ = build_panel()
res = fit_ols("violent", "tmax10", dfa, hac=9)
print(fmt_pct(pct_ci(res, "tmax10")))
# +5.6% [+5.1, +6.0]
```

**Lab + read results:**

```bash
python run_model_lab.py
python run_eda_diagnostics.py
```

```python
import json
lab = json.load(open("data/model_lab_results.json", encoding="utf-8"))
print(lab["summary"]["baseline_m2_violent_same_day"])
print(lab["summary"]["dlag_cumulative_violent"])
print(lab["summary"]["falsification"])
```

---

## Reproduce

```bash
cd heat
python -m venv .venv   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_model_lab.py
python run_eda_diagnostics.py
# optional: jupyter nbconvert --to notebook --execute --inplace heat_and_crime.ipynb
python -m http.server 8000   # article at http://127.0.0.1:8000/
```

Offline if `data/` is present.

---

## Layout

```
index.html              public article
figures/                charts
panel.py                daily panel builder
models.py               FE / DLAG / FDR helpers
run_model_lab.py        G0–G7 + F1–F7
run_eda_diagnostics.py
heat_and_crime.ipynb    full executed science notebook
data/                   caches, MANIFEST, lab JSON
requirements.txt
```

**Sources:** [Chicago `ijzp-q8t2`](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) · [Open-Meteo ERA5](https://open-meteo.com/en/docs/historical-weather-api)
