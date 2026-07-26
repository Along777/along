# Age Myth Findings (Peer-Hardened)

**Peer review applied:** de-duplicated subseries, cluster bootstrap CIs, honest model stack, mid-adult HLD check, expanded charts.

**Scope:** High-quality vital-registration populations in the HMD public summary (not all humans in “the past”).

---

## Executive answer

**No — people did not all die at 30–35.**

Period life expectancy at birth near 30–40 reflects **catastrophic early-life mortality**, not typical adult ages at death.

### Myth A (primary, de-duplicated)

Among country-years with e0 < 40 (**n = 211**, **11 countries**, subseries removed):

| Metric | Value |
|--------|------:|
| Median e0 | 37.00 |
| Median IMR (per 1,000) | 195.9 |
| Median survival birth→65 | 0.270 (**~27%**) |
| Median expected age **if alive at 65** | **75.39** |
| Cluster-bootstrap 95% CI (by country) | [75.12, 75.70] |
| Share with age\|65 ≥ 70 | **100%** |
| Median gap (age\|65 − e0) | 38.3 years |

**Reject Myth A:** `True`.

**Critical reading note:** age\|65 is *conditional on surviving to 65*. With median S(0→65)≈0.27, most births under those period rates **never reach 65**. That is the myth mechanism—not a contradiction.

### Mid-adult check (HLD median, answers “what about age 15?”)

When e0 < 40 in the HLD median panel:

| Metric | Value |
|--------|------:|
| n (region-years) | 111 |
| Median expected age if alive at **15** | **57.61** |
| Share age\|15 ≥ 50 | 0.8828828828828829 |
| Median expected age if alive at **30** | 62.462500000000006 |

Even at age 15—not only 65—conditional expected ages sit **far above 30–35** when e0 is low. (HLD tables are heterogeneous; treat as supporting evidence.)

### Myth B (adult longevity also improved)

De-duplicated countries with pre-1900 and post-2000 e65 (**n = 12**):

| Metric | Value |
|--------|------:|
| Mean Δe65 | **+8.45 years** |
| Cluster bootstrap 95% CI | [7.74, 9.17] |
| Range | 6.36 to 10.66 |
| Mean Δe80 (if available) | 3.8902802755609005 |

**Reject Myth B as an absolute:** adults’ remaining LE rose by roughly **8–9 years** on average—not “only infants improved.”

---

## Correlations (real, partly structural)

| Quantity | Value |
|----------|------:|
| Median within-country corr(e0, IMR) | **-0.9438580012510758** |
| Mean within-country corr | -0.8487530997319996 |
| Pooled corr(IMR, year) | -0.9122956907618426 |

IMR and calendar time are **highly collinear**. Associations are robust across countries but **not causal**.

---

## Models — not overfit in the ML sense; do not headline R²

e0 and IMR are outputs of the **same period mortality schedule**. High R² when predicting e0 from IMR is **expected demography**, not evidence of a clever overfit model.

| Model | R² | β(IMR) | Role |
|-------|---:|-------:|------|
| M0 pooled e0 ~ IMR | 0.918 | -0.2092 | Baseline |
| M1 within e0 ~ IMR | 0.914 | -0.2111 | Drop country means |
| M2 within e0 ~ IMR + year | 0.951 | -0.1211 | Hold linear time |
| M3 first difference Δe0 ~ ΔIMR | 0.507 | -0.1126 | Common trends reduced |
| FE full (sensitivity only) | 0.964 | -0.1211 | **Do not headline** |

Leave-one-country-out stability of within β(IMR\|year): mean -0.1213, sd 0.0035.

---

## Charts (myth-bust pack)

### Core (original)

- `outputs/figures/F1_sweden_myth_killer.png`
- `F1b_multicountry_e0_vs_exp65.png`, `F2`–`F6`, `F_mythB_dumbbell_e65.png`

### Peer pack (new)

| Figure | Path |
|------|------|
| G1 Sweden storyboard | `outputs/figures/peer_pack/G1_sweden_storyboard.png` |
| G2 HLD mid-adult e0 vs age\|15 | `.../G2_hld_mid_adult_e0_vs_age15.png` |
| G2b HLD scatter low e0 | `.../G2b_hld_scatter_low_e0_age15.png` |
| G3 Within-country corr boxplot | `.../G3_within_country_corr_boxplot.png` |
| G4 First differences | `.../G4_first_differences_e0_imr.png` |
| G5 Model R² comparison | `.../G5_model_r2_comparison.png` |
| G6 Myth A forest by country | `.../G6_mythA_forest_by_country.png` |
| G7 Sex sensitivity Sweden | `.../G7_sweden_sex_gap.png` |
| G8 Survival composition | `.../G8_sweden_survival_composition.png` |
| G9 Eurostat e0–age15 gap | `.../G9_eurostat_e0_age15_gap.png` |
| G10 Coverage heatmap | `.../G10_coverage_heatmap.png` |
| Myth B deduped dumbbell | `.../G_mythB_dumbbell_deduped.png` |

---

## Caveats (required)

1. Period LE ≠ lived cohort lifespan.  
2. HMD summary ≠ world population 1700.  
3. Only Sweden has continuous mid-18th-century coverage.  
4. Conditional longevity ≠ “everyone reached that age.”  
5. HLD mid-adult results use median-collapsed heterogeneous tables.  
6. Regressions are **associational**.

---

## Reproduce

```powershell
python -m src.analysis.peer_hardening
python -m src.analysis.figures_peer_pack
python -m src.analysis.write_peer_report
```

---

## Bottom line

> Historical e0 near 30–40 is the arithmetic of dead infants and children.  
> Conditional on adulthood (age 15 or 65), expected ages were already far above 30–35.  
> Adult remaining LE still rose ~8–9 years from pre-1900 to post-2000.  
> Both the crude myth and the “only infants improved” overcorrection fail—in high-quality VR data, with charts and hardened stats.
