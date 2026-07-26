# Diet Soda / Artificial Sweeteners & Cardiometabolic Outcomes

Independent, fully public, reproducible analysis of **artificially sweetened beverage (ASB) / diet soda** consumption and **cardiometabolic markers**, with a transparent secondary module on **cancer claims**.

**Public article (open from project root):** [`index.html`](index.html) — finding-first myth lab write-up (cancer + diet lead, charts, models). Rebuild with `python scripts/build_html_report.py`.

**Data source:** [NHANES](https://www.cdc.gov/nchs/nhanes/index.html) (CDC/NCHS) + USDA WWEIA food categories + NCHS public Linked Mortality Files.  
**Principle:** Data engineering first; analysis only after a clean analysis-ready dataset.

> This project uses **only public data**. It cannot prove or disprove long-latency cancer causation the way multi-decade cohorts can. The cancer module is designed for **honest myth-testing + regulatory context**, not sensational claims.

---

## Research questions

1. How do ASB consumers differ from sugar-sweetened beverage (SSB) consumers and non-consumers on **measured** cardiometabolic markers?
2. Do observational associations in the literature replicate in carefully cleaned public NHANES data after adjustment?
3. How sensitive are results to exposure definitions and covariate sets?
4. (Secondary) What do NHANES self-reported cancer history and linked cancer mortality show for ASB intake—and how should that be read next to IARC/JECFA/FDA?

---

## Project status

| Phase | Status |
|-------|--------|
| 0 Scaffold | Done |
| 1A Download + inventory | Done |
| 1B Analysis-ready dataset | Done (`data/processed/analysis_ready.parquet`, n≈19k adults) |
| EDA myth dashboard | Done (`outputs/figures/myth_*.png`) |
| Inference ladder + multiverse | Done (`outputs/tables/model_cardio_ladder.csv`) |
| ML + hypertuning | Done (`outputs/tables/ml_tuning_results.json`) |
| Myth verdicts | Done (`docs/myth_verdicts.md`) |

---

## Structure

```text
diet-soda-analysis/
├── data/raw|interim|processed/
├── notebooks/
├── src/data|features|analysis/
├── outputs/figures|tables/
├── docs/
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd diet-soda-analysis
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run the Phase 1A pipeline

From the project root (`diet-soda-analysis/`):

```bash
# 1. Download NHANES 2011–2018 core XPT files
python -m src.data.download_nhanes

# 2. Download USDA WWEIA food category files (best-effort public URLs)
python -m src.data.download_wweia

# 3. Download NCHS public Linked Mortality Files
python -m src.data.download_mortality

# 4. Build inventory tables + markdown report
python -m src.data.inventory

# 5. Build analysis-ready datasets (Phase 1B)
python -m src.data.build_analysis_dataset

# 6. Myth Lab analytics
python -m src.analysis.run_eda
python -m src.analysis.run_models
python -m src.analysis.run_ml
python -m src.analysis.run_verdicts

# 7. Cancer talking-point module (M4)
python -m src.analysis.run_cancer_module
```

**Key artifacts**

| Output | Path |
|--------|------|
| Analysis-ready table | `data/processed/analysis_ready.parquet` |
| Myth figures | `outputs/figures/myth_*.png` |
| Model ladder | `outputs/tables/model_cardio_ladder.csv` |
| ML results | `outputs/tables/ml_tuning_results.json` |
| **Myth verdicts** | `docs/myth_verdicts.md` |
| Inventory report | `docs/data_inventory_report.md` |

Raw downloads are **gitignored**. Re-run download scripts anytime (idempotent).

---

## Analytic window (planned)

- **Cycles:** NHANES 2011–2012, 2013–2014, 2015–2016, 2017–2018  
- **Exposure (planned):** WWEIA **7102** diet soft drinks vs **7202** soft drinks (keyword rules as sensitivity)  
- **Population (proposed):** non-pregnant adults ≥20 with MEC exam + Day-1 dietary recall  
- **Outcomes:** BMI, waist, BP, HbA1c, fasting glucose/insulin, lipids; secondary MCQ cancer history + LMF cancer mortality  

See `docs/dataset_landscape.md` for other public sources considered (Open Food Facts, BRFSS, CAERS, and cite-only cohorts).  
See `docs/reddit_myth_map.md` for public-discourse claims (R1–R18) vs what we can test.

---

## Scientific caveats (read first)

- Cross-sectional NHANES cannot establish causality.
- 24-hour dietary recall ≠ long-term usual intake.
- Reverse causation is especially plausible for diet soda (people with diabetes/obesity may switch to diet drinks).
- IARC Group 2B for aspartame is **hazard** identification with *limited* evidence; JECFA reaffirmed ADI and found human cancer evidence **not convincing** at usual intakes. See `docs/cancer_evidence_brief.md`.
- **Variance estimation is approximate** (weights used; PSU/strata not fully design-based). See `docs/assumptions.md`.
- Self-audit fixed a serious bug: raw NHANES weights must **not** be used as GLM `freq_weights` (inflates n and fakes p≈0).

---

## License / attribution

NHANES and linked mortality data are public products of CDC/NCHS. USDA WWEIA/FNDDS are public products of USDA ARS. Cite CDC/NCHS and USDA appropriately in any write-up. This repository code is for research/education.
