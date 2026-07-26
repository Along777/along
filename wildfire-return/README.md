# Return to Fire

The 2017 Tubbs Fire took my family's home. When I started my data science career in 2020, fires
had to be my first subject, a three-part series written by hand on a dataset that ended in 2015,
22 months before the fire that mattered most to me. This project is the return: the same questions,
asked on FPA FOD-Attributes (2.3M fires × 308 columns, 1992–2020), answered the way I would answer
them now, with the original's bugs rebuilt as museum pieces beside the honest versions.

**Article:** https://along777.github.io/along/wildfire-return/

## The process, in rounds

| Round | What happened | Exhibit |
|---|---|---|
| **1 · First shot** | One prompt: old 2020 files in → goals & intent analyzed → constructive critique, then a full 2026 rebuild (~6 hours: fetch, reduce, 4 labs, article, verifier) | [FIRST_SHOT.md](FIRST_SHOT.md) tells the trail; [first_shot.html](first_shot.html) is the article frozen byte-identical, sha256-recorded, never edited |
| **2 · Data expert's cut** | The corpus cross-examined: 29-check streaming audit over 4.68 GB, ingest hardening (`--verify-only`, Range resume), auto-generated data dictionary, a pre-registered cleaning gate that changed **0 rows**, and caught 47,367 AK/HI/PR rows hiding in the "CONUS" data (3 main-article numbers corrected) plus 64 FOD_ID collisions. Closed with a self-review that found CA's **label-selection drift** (10.1% → 41.7% exclusion across the ML splits, now a stated caveat) and shipped the modeling handoff: feature contract in `wildfire.py`, frozen splits, `dup_flags.parquet` sidecar | [data.html](https://along777.github.io/along/wildfire-return/data.html), *The Data Expert's Cut* (§12 Reviewed before handoff); [DATA_DICTIONARY.md](DATA_DICTIONARY.md) |
| **3 · Modeling** | The escalation-risk model, built properly: cache v2 (29 more fire-science columns incl. wind direction), physics features (Hot-Dry-Windy, Fosberg FFWI, wind–terrain alignment, leakage-safe fire history), a three-stack tuning bake-off (sklearn vs Optuna vs LightGBM), spatial-block bootstrap intervals, calibration on the previously-unused validation era, decision-curve analysis, plus **a retraction**: Round 1's "calibrated probabilities" had a Brier skill score of −3.78 | [modeling.html](https://along777.github.io/along/wildfire-return/modeling.html), *The Modeling Stage* |
| **4 · Red team** | A fresh-context hostile reviewer audited the Round-3 code with orders to be merciless: **35 findings**. Four blockers fixed (the headline mixed two models, so an HGB *system of record* was named; calibration re-selected on val, which changed the chosen map; a circular leak tripwire rebuilt with empirically calibrated thresholds, now catching all four showcased leaks; the "scored once" protocol claim amended with a full touch accounting). Plus the experiments a boss would demand: 5-seed stability, geography/yesterday-knowledge/same-day-load ablations, fair zoo re-runs, leave-2020-out, population-tercile lifts, an out-of-sample Tubbs percentile, all in `modeling.html` §15 with a stakeholder FAQ | `run_redteam.py`; *The Modeling Stage* §15 "Red team: the hostile review" |
| **5 · Final report** | Every page swept against the reviewed numbers (13 leftover contradictions fixed in modeling.html; calibration figures re-rendered from the val-selected map), then `index.html` rebuilt as the finished report: the five-round journey published as its own section, the escalation model integrated into the main narrative, the Tubbs row scored in-page, the red-team defense table extended, and the verifier taught to **ban retired numbers**, so a stale claim reappearing anywhere is a build failure. Also fixed: the family-photo captions (one photo before the fire; two after) | this README's expected-output block; `verify_claims.py` (needles + retired-string bans); [the final report](https://along777.github.io/along/wildfire-return/) |
| **6 · Publication** | The two questions the project had never answered, answered with measurements: a train/val/test table (the in-sample gap had never been computed), a label-shuffle null for the escalation model, and a space-time accounting of why ignition *occurrence* is unanswerable from these caches. Plus the exhibit rail linking all four pages, a full plain-language editing pass (every em dash removed, now enforced by the verifier), and an accessibility pass (contrast, focus states, heading order, scoped table headers, lazy images, zero inline styles) | `run_generalization.py`; [PUBLISH.md](PUBLISH.md); *The Modeling Stage* §12 |

## The headlines

| Question | 2020's answer | 2026's answer |
|---|---|---|
| Are US wildfires increasing? | "Yes, and climate change is a strong cause" (one trend line) | **SPLIT**, ignition counts flat (MK p=0.564); burned area up **2.4×**; +3.1 fires ≥5,000 ac per year in the lower 48 (p=0.027) |
| Where? | "Across the U.S. and across California and Florida" | The **West**. Florida's large-fire trend: IRR 0.99/decade, p=0.612, **rejected** |
| Fire weather? | (no climate variable existed in the data) | Top-decile-ERC ignitions in CA: **7.7% → 28.2%** era-over-era; season **+17.8 days/decade** |
| Cause prediction? | 94.7% accuracy, celebrated | Majority baseline was 95.7%. Honest model: macro-F1 **0.868** vs climatology 0.787, temporally held out |
| The Tubbs Fire? | Not in the data (ended 2015) | **FOD_ID 400015986**, 36,807 acres, rmin 6.3%, ERC >90th local percentile. One row among 2,302,521, scored **97th percentile** out-of-sample by the escalation model, calibrated risk 9.4% vs 2.1% base |
| Which ignition escalates? | (no such model existed) | Test AP **0.126 ± 0.0023** across 5 seeds, **6.0×** [4.8–7.5] the base rate; top-1% precision 20.7%; survived a 35-finding red team |
| Are the probabilities honest? | "when it says 12%, about 12% do" (published) | **Retracted**: Brier skill **−3.78** as published. Rebuilt with val-selected isotonic calibration: **+0.053** |
| Is it overfit? | (never measured) | Train AP **0.538** vs test **0.124**, so the in-sample gap is real; but the two held-out eras agree (ROC-AUC **0.813** and **0.817**) and shuffled labels collapse the model to AP **0.0232** at a 0.0206 base rate |
| Can it predict fires *before* they start? | (not asked) | **No.** It models P(escalation \| a fire was reported). Ignition occurrence needs the 7.6M CA cell-days where nothing burned, and no cache here contains them |

## Mental model

```
fetch_raw.py ──> data/raw/*.csv (4.68 GB, gitignored)          [network, once]
fetch_recent.py ─> data/recent_annual.csv (WFIGS 2021-25)      [network, seconds]
reduce_raw.py ──> data/*.parquet + *.csv + MANIFEST.json       [offline, ~10 min]
                   │ (SHA-256 locked; date assertions make the
                   │  2020 Julian bug a crash, not a surprise)
run_eda.py ──────> dataset anatomy         ─┐
run_trend_lab.py > H1-H6 verdicts           ├─> data/*_results.json + figures/*.png
run_ml_lab.py ──> museum + honest models    │
run_maps.py ────> the three maps           ─┘
run_data_audit.py> 29-check corpus audit ──> data_audit.json + audit_*.png + DATA_DICTIONARY.md
features.py ─────> physics features + ablation ladder + tripwire v2   ─┐ (imported by
tuning.py ───────> three-track search spaces + model factories        ─┘  the modeling stage)
run_modeling.py ─> escalation model: zoo, calibration, decisions ─> modeling_results.json + m*.png
run_redteam.py ──> the hostile review (after run_modeling) ─> redteam_results.json + r1_redteam.png
run_final_figures.py > m4/m6 re-rendered from the val-selected calibration map
verify_claims.py > every number on ALL THREE pages == the JSONs + freeze check, or exit 1
```

The ML protocol: train ≤2014, validate 2015–17, **test 2018–20** (the model never sees the era
containing Tubbs), plus 1°×1° spatial-block CV and three baselines (majority, per-cell×month
climatology, logistic). Bare accuracy never headlines anything.

## Follow along

```powershell
# 0 · environment (Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 1 · the emotional payload, offline once caches exist (they ship with the repo)
python examples\find_tubbs.py
```
```
==============================================================
  TUBBS FIRE  --  FOD_ID 400015986  (FPA-FOD v6 + attributes)
==============================================================
  discovered      2017-10-08  (day 281 of the year)
  final size      36,807 acres  (class G)
  ...
  min humidity    6.3 %
  ERC             77  (local climatological percentile: >90%)
```

```powershell
# 2 · the headline trends, offline, instant
python examples\headline_trend.py
```
```
Large fires (>= 1,000 acres), 1992-2020, NB2 GLM with state fixed effects:
  West (11 states):  IRR 1.14/decade (95% CI 1.04-1.26, MK p=0.185)
  East (the rest):   IRR 1.10/decade (95% CI 0.99-1.23, MK p=0.021)
All reported ignitions, any size (partly a reporting series -- see article):
  National:          IRR 1.24/decade (95% CI 1.18-1.31, MK p=0.564)
```

```powershell
# 3 · full rebuild from raw (one ~5 GB download, then offline forever)
python fetch_raw.py --yes            # or --years 2015-2020 to start with ~1 GB
python fetch_recent.py
python reduce_raw.py                 # prints per-year QA and "TUBBS FOUND"
python run_eda.py
python run_trend_lab.py              # prints the H1-H6 verdicts
python run_ml_lab.py                 # ~20 min: museum replica + honest models
python run_maps.py
python run_data_audit.py             # ~12 min: 29-check corpus audit -> data_audit.json,
                                     #   7 audit figures, DATA_DICTIONARY.md
python fetch_raw.py --verify-only    # re-hash all raw files vs Zenodo: "30/30 files verified ok"

# 4 · the modeling stage (round 3)
python tuning.py --probe             # which of sklearn/optuna/lightgbm/shap are importable
python run_modeling.py --trials 25   # ~45 min: bake-off, zoo, calibration, decisions, Tubbs
python run_modeling.py --quick --subsample 15000   # 5-min smoke run of the same pipeline

# 5 · the hostile review (round 4)
python run_redteam.py                # ~10 min: seed stability, clean calibration, geography/
                                     #   yesterday/same-day ablations, fair zoo, tercile lifts,
                                     #   OOS Tubbs, decision-curve redo -> redteam_results.json
python run_final_figures.py          # ~2 min: m4/m6 re-rendered from the val-selected map,
                                     #   hard-asserted equal to redteam_results.json first
python run_generalization.py         # ~1 min: train/val/test AP + ROC-AUC, label-shuffle null,
                                     #   ignition-question accounting -> generalization.json

python verify_claims.py              # all three pages vs the JSONs + freeze check, exit 0
```

Expected tail of `verify_claims.py`:

```
48 in index.html + 23 in data.html + 59 in modeling.html; 6 verdict chips;
18 retired strings absent; all figures present and referenced; zero JS on
every page; cross-links intact; first_shot freeze intact; all 11 manifest
entries match.
```

("Retired strings" are numbers the Round-4 red team superseded. The verifier
fails if a stale headline ever reappears, not just if a current one goes missing.
It also enforces the house style rule that no page may contain an em dash.)

## Design choices

| Choice | Why |
|---|---|
| Per-year Zenodo files, not the 5 GB blob | resume granularity; `--years` lets Tubbs work start after one file |
| Headline trends restricted to ≥1,000-acre fires | small-fire counts partly measure reporting; the article shows the threshold ladder that flips the answer |
| PR/AK/HI excluded from CONUS series, MA/KS/LA/TN sensitivity-tested | PR's reporting is discontinuous; AK/HI records turned out to be hiding in the "CONUS" dataset (the Round-2 audit's biggest catch); step-change states are paperwork, not physics |
| `HistGradientBoosting` as the **system of record**; LightGBM and Optuna as tuner tracks | native NaN + categoricals, zero compiled deps, the pre-committed, fully-analyzed model. The Round-3 bake-off runs all three stacks; LightGBM's higher single test AP is *reported, not selected* (see modeling.html §15) |
| Permutation importance + PDP in the Round-1 lab; grouped permutation + TreeSHAP in the modeling stage | correlated weather features share credit either way, which is why the modeling stage permutes whole families and cross-checks with SHAP (the two disagreeing would be the finding) |
| WFIGS 2021–25 drawn dashed, never tested | different reporting system; like-for-like ≥1,000-ac series only, zero cross-seam claims |
| 2020 museum replica keeps its bugs | the dead MONTH/DOW constants and leaked FIRE_SIZE are the point, scored honestly beside the fix |
| Everything hash-locked (`MANIFEST.json`) | the Julian bug was silent; drift here fails loud |

## Troubleshooting

- **`MANIFEST mismatch`**, a cache changed outside `reduce_raw.py`. Re-run `python reduce_raw.py`.
- **Download interrupted**, re-run `fetch_raw.py --yes`; complete files are md5-skipped.
- **`COLUMN DISCOVERY FAILED`**, the Zenodo dataset changed a column name; the full header prints.
  Update `wildfire.RAW_COLUMNS`, nothing guesses silently.
- **Memory**, the reducer streams 250k-row chunks; peak stays under ~1.5 GB.
- **venv on Windows**, `.venv\Scripts\activate` (not `bin/`).

## Layout

```
wildfire-return/
  index.html            the article (hand-rolled CSS, zero JS, figures/ only)
  data.html             The Data Expert's Cut -- the Round-2 corpus audit article
  modeling.html         The Modeling Stage -- Rounds 3+4: escalation model + red-team review
  first_shot.html       Round 1 frozen byte-identical (see FIRST_SHOT.md)
  FIRST_SHOT.md         the one-prompt trail + the freeze hash
  DATA_DICTIONARY.md    AUTO-GENERATED by run_data_audit.py -- units, missingness, schemas
  wildfire.py           paths, column map, cause taxonomies, units registry, loaders
  models.py             NB2/Theil-Sen/Mann-Kendall + honest-ML protocol helpers
  checks.py             streaming validation primitives for the corpus audit
  fetch_raw.py          Zenodo annual CSVs + Census boundaries (--verify-only, Range resume)
  fetch_recent.py       WFIGS 2021-2025 large-fire aggregates (network, seconds)
  reduce_raw.py         4.68 GB -> ~33 MB hash-locked caches (85 columns since cache v2) + the Tubbs row
  run_eda.py            dataset anatomy: 6 figures + eda_results.json
  run_trend_lab.py      trends, hypotheses H1-H6: 7 figures + trend_lab_results.json
  run_ml_lab.py         museum replica + honest models: 6 figures + ml_lab_results.json
  run_maps.py           Albers CONUS pair, CA wind-season map, FL spring map
  run_data_audit.py     29-check streaming corpus audit -> data_audit.json + audit figures
  features.py           physics feature engineering (HDW, Fosberg FFWI, wind-terrain alignment,
                        place-relative anomalies, leakage-safe fire history) + ablation ladder
  tuning.py             three-track hyperparameter bake-off (sklearn / optuna / lightgbm)
  run_modeling.py       the escalation-risk model: zoo, calibration, decisions, leak ladder, Tubbs
  run_redteam.py        Round 4: 35-finding hostile review's experiments -> redteam_results.json
  run_final_figures.py  m4/m6 re-rendered from the val-selected calibration map
  verify_claims.py      ALL THREE pages vs the JSONs + first_shot freeze check, or exit 1
  requirements.txt      pandas/sklearn/statsmodels/pyarrow/matplotlib + lightgbm/optuna/shap
  examples/             find_tubbs.py · headline_trend.py (offline, instant)
  data/                 committed caches + results JSONs + MANIFEST.json (data/raw/ gitignored)
  figures/              article_*.png · eda_*.png · g*_*.png · f*_*.png · audit_*.png ·
                        m*_*.png · r1_redteam.png · story_*.jpg
```

## Sources

- **FPA FOD-Attributes**, Pourmohamad et al. 2024, [ESSD](https://essd.copernicus.org/articles/16/3045/2024/),
  Zenodo [10.5281/zenodo.8381129](https://zenodo.org/records/8381129) (CC-BY 4.0)
- **FPA-FOD v6**, Short, K.C., USFS RDS-2013-0009.6 (1992–2020)
- **WFIGS Interagency Fire Perimeters**, NIFC Open Data (public domain)
- **State boundaries**, US Census cartographic boundary files, 1:20m (public domain)
- The 2020 originals, unedited: [Part I](https://along777.github.io/along/projects/wildfiresp1.html) ·
  [Part II](https://along777.github.io/along/projects/wildfiresp2.html) ·
  [Part III](https://github.com/Along777/along/blob/master/projects/randomforest.ipynb)
