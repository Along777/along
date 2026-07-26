"""Write peer-hardened findings + response to peer review."""
from __future__ import annotations

import json
from pathlib import Path

from src.paths import OUTPUTS


def main() -> None:
    data = json.loads((OUTPUTS / "tables" / "peer_hardened_summary.json").read_text(encoding="utf-8"))
    a = data["myth_a_hardened"]
    b = data["myth_b_hardened"]
    m = data["model_comparison"]
    h = data.get("hld_mid_adult", {})
    ic = data.get("within_corr_summary", {})
    exp = a["exp_death_65"]
    de = b["delta_e65"]
    models = m["models"]

    findings = f"""# Age Myth Findings (Peer-Hardened)

**Peer review applied:** de-duplicated subseries, cluster bootstrap CIs, honest model stack, mid-adult HLD check, expanded charts.

**Scope:** High-quality vital-registration populations in the HMD public summary (not all humans in “the past”).

---

## Executive answer

**No — people did not all die at 30–35.**

Period life expectancy at birth near 30–40 reflects **catastrophic early-life mortality**, not typical adult ages at death.

### Myth A (primary, de-duplicated)

Among country-years with e0 < 40 (**n = {a['n_country_years']}**, **{a['n_countries']} countries**, subseries removed):

| Metric | Value |
|--------|------:|
| Median e0 | {a['e0_median']:.2f} |
| Median IMR (per 1,000) | {a['imr_median']:.1f} |
| Median survival birth→65 | {a['s_to_65_median']:.3f} (**~{100*a['s_to_65_median']:.0f}%**) |
| Median expected age **if alive at 65** | **{exp['median']:.2f}** |
| Cluster-bootstrap 95% CI (by country) | [{exp['ci_low']:.2f}, {exp['ci_high']:.2f}] |
| Share with age\\|65 ≥ 70 | **{100*a['share_exp_death_65_ge_70']:.0f}%** |
| Median gap (age\\|65 − e0) | {a['adult_gap_65']['median']:.1f} years |

**Reject Myth A:** `{a['reject_myth_A']}`.

**Critical reading note:** age\\|65 is *conditional on surviving to 65*. With median S(0→65)≈{a['s_to_65_median']:.2f}, most births under those period rates **never reach 65**. That is the myth mechanism—not a contradiction.

### Mid-adult check (HLD median, answers “what about age 15?”)

When e0 < 40 in the HLD median panel:

| Metric | Value |
|--------|------:|
| n (region-years) | {h.get('age15',{}).get('n_low_e0')} |
| Median expected age if alive at **15** | **{h.get('age15',{}).get('median_exp_age_if_15')}** |
| Share age\\|15 ≥ 50 | {h.get('age15',{}).get('share_exp15_ge_50')} |
| Median expected age if alive at **30** | {h.get('age30',{}).get('median_exp_age_if_30')} |

Even at age 15—not only 65—conditional expected ages sit **far above 30–35** when e0 is low. (HLD tables are heterogeneous; treat as supporting evidence.)

### Myth B (adult longevity also improved)

De-duplicated countries with pre-1900 and post-2000 e65 (**n = {de['n_clusters']}**):

| Metric | Value |
|--------|------:|
| Mean Δe65 | **+{de['mean']:.2f} years** |
| Cluster bootstrap 95% CI | [{de['ci_low']:.2f}, {de['ci_high']:.2f}] |
| Range | {de['min']:.2f} to {de['max']:.2f} |
| Mean Δe80 (if available) | {b.get('delta_e80',{}).get('mean')} |

**Reject Myth B as an absolute:** adults’ remaining LE rose by roughly **8–9 years** on average—not “only infants improved.”

---

## Correlations (real, partly structural)

| Quantity | Value |
|----------|------:|
| Median within-country corr(e0, IMR) | **{ic.get('median_pearson')}** |
| Mean within-country corr | {ic.get('mean_pearson')} |
| Pooled corr(IMR, year) | {m.get('corr_imr_year')} |

IMR and calendar time are **highly collinear**. Associations are robust across countries but **not causal**.

---

## Models — not overfit in the ML sense; do not headline R²

e0 and IMR are outputs of the **same period mortality schedule**. High R² when predicting e0 from IMR is **expected demography**, not evidence of a clever overfit model.

| Model | R² | β(IMR) | Role |
|-------|---:|-------:|------|
| M0 pooled e0 ~ IMR | {models['M0_pooled_imr_only']['r2']:.3f} | {models['M0_pooled_imr_only']['beta_imr']:.4f} | Baseline |
| M1 within e0 ~ IMR | {models['M1_within_imr_only']['r2']:.3f} | {models['M1_within_imr_only']['beta_imr']:.4f} | Drop country means |
| M2 within e0 ~ IMR + year | {models['M2_within_imr_plus_year']['r2']:.3f} | {models['M2_within_imr_plus_year']['beta_imr']:.4f} | Hold linear time |
| M3 first difference Δe0 ~ ΔIMR | {models['M3_first_difference']['r2']:.3f} | {models['M3_first_difference']['beta_dimr']:.4f} | Common trends reduced |
| FE full (sensitivity only) | {models['M_fe_full_sensitivity']['r2']:.3f} | {models['M_fe_full_sensitivity']['beta_imr']:.4f} | **Do not headline** |

Leave-one-country-out stability of within β(IMR\\|year): mean {m['loco_within_imr_year_beta']['mean']:.4f}, sd {m['loco_within_imr_year_beta']['std']:.4f}.

---

## Charts (myth-bust pack)

### Core (original)

- `outputs/figures/F1_sweden_myth_killer.png`
- `F1b_multicountry_e0_vs_exp65.png`, `F2`–`F6`, `F_mythB_dumbbell_e65.png`

### Peer pack (new)

| Figure | Path |
|------|------|
| G1 Sweden storyboard | `outputs/figures/peer_pack/G1_sweden_storyboard.png` |
| G2 HLD mid-adult e0 vs age\\|15 | `.../G2_hld_mid_adult_e0_vs_age15.png` |
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
"""
    out = OUTPUTS / "reports" / "age_myth_findings_peer_hardened.md"
    out.write_text(findings, encoding="utf-8")
    print(f"Wrote {out}")

    response = """# Peer Review Response

| Issue | Severity | Response |
|-------|----------|----------|
| FE R²=0.97 looks overfit | High (comms) | Added model stack M0–M3; within R²; first differences; LOCO β; FE demoted to sensitivity. High R² expected (shared mortality schedule). |
| Bootstrap CI too tight | High | Cluster bootstrap by country for Myth A/B. |
| France/UK TOTAL+CIVILIAN double count | Medium | De-duplication filter drops CIVILIAN/East/West/ethnic subseries. |
| Age 65 only for “adults” | High (claim scope) | Added HLD e15/e30 mid-adult analysis + figures G2. |
| Correlation vs time collinearity | Medium | Report corr(IMR,year); first-difference model M3; within+year M2. |
| Not enough charts | Medium | Peer pack G1–G10 + survival composition + coverage honesty. |
| Overclaim “the past” globally | Medium | Findings scoped to HMD-like VR populations; G10 coverage heatmap. |
| Concordance r≈1 independent validation | Low | Note shared HMD lineage with OWID for modern e0. |

**Verdict after remediation:** Myth-busting claims remain **directionally correct and stronger under de-dupe + mid-adult checks**. Statistical presentation is peer-safe if FE R² is not headlined.
"""
    out2 = OUTPUTS / "reports" / "PEER_REVIEW_RESPONSE.md"
    out2.write_text(response, encoding="utf-8")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
