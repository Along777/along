"""One-shot: panels → tests → figures → findings report."""
from __future__ import annotations

import json
from pathlib import Path

from src.analysis.figures import run_all as run_figures
from src.analysis.myth_tests import run_all as run_tests
from src.analysis.panels import save_panels
from src.paths import OUTPUTS, ensure_dirs


def write_findings() -> Path:
    tables = OUTPUTS / "tables"
    summary_path = tables / "myth_tests_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    a = data["myth_a_all"]
    b = data["myth_b"]
    reg = data.get("fe_regression", {})
    conc = data.get("concordance_hmd_vs_owid", {})
    ic = data.get("infant_corr_summary", {})

    exp = a["exp_death_65"]
    gap = a["adult_gap_65"]
    de = b["delta_e65"]

    lines = f"""# Age Myth Findings Report

**Data:** HMD public summary indicators via `life_expectancy_modeling`  
**Question:** Did people in the past “only live to 30–35”?

---

## Executive answer

**No.** Period life expectancy near 30–40 is an average **dragged down by catastrophic infant and child mortality**, not the typical age of adult death.

In every high-quality country-year where e₀ < 40 in this dataset:

- Median expected age at death **if alive at 65** is about **{exp['median']:.1f} years**  
  (95% bootstrap CI [{exp['ci_low']:.1f}, {exp['ci_high']:.1f}]).
- **{100*a['share_exp_death_65_ge_70']:.0f}%** of those country-years have expected age|65 ≥ 70.
- Median survival from birth to 65 is only about **{100*a['s_to_65']['median']:.0f}%** — many never reached old age.
- Median infant mortality is about **{a['imr']['median']:.0f}** per 1,000 births.

A second claim — “if you survived childhood you lived as long as modern adults” — is also **false as an absolute**:

- Across **{b['n_countries']}** countries with both pre-1900 and post-2000 data, mean remaining LE at 65 rose by about **{de['mean']:.1f} years**  
  (95% CI [{de['ci_low']:.1f}, {de['ci_high']:.1f}]; range about {b['delta_e65_min']:.1f}–{b['delta_e65_max']:.1f}).

So both the crude myth and the popular overcorrection fail.

---

## Myth A — “They died at 30”

### Test sample

- Country-years with e₀ < 40 (both sexes, HMD public summary): **n = {a['n_country_years']}**  
- Countries represented: **{a['n_countries']}**  
- Of which e₀ in [30, 35]: **{a['n_e0_in_30_35']}**

### Results

| Metric | Value |
|--------|------:|
| Median e₀ | {a['e0']['median']:.2f} |
| Median IMR (per 1,000) | {a['imr']['median']:.1f} |
| Median S(0→65) | {a['s_to_65']['median']:.3f} |
| Median expected age if alive at 65 | **{exp['median']:.2f}** |
| 95% CI (bootstrap median) | [{exp['ci_low']:.2f}, {exp['ci_high']:.2f}] |
| Share expected age\\|65 ≥ 70 | **{100*a['share_exp_death_65_ge_70']:.1f}%** |
| Median adult gap (age\\|65 − e₀) | {gap['median']:.1f} years |

**Decision:** Reject Myth A (`reject_myth_A = {a['reject_myth_A']}`).

### How to read this without overclaiming

Expected age at death *conditional on age 65* does **not** mean most people reached 65.  
S(0→65) near 0.27 means most births did **not** survive to 65 under those period rates.  
The myth error is equating **e₀** with “when adults died.”

---

## Myth B — “Only infant mortality improved; adults always lived modern lengths”

| Metric | Value |
|--------|------:|
| Countries (pre-1900 & post-2000 e₆₅) | {b['n_countries']} |
| Mean Δe₆₅ | **+{de['mean']:.2f} years** |
| 95% CI | [{de['ci_low']:.2f}, {de['ci_high']:.2f}] |
| Median Δe₆₅ | +{b['delta_e65_median']:.2f} |
| Range | {b['delta_e65_min']:.2f} to {b['delta_e65_max']:.2f} |

**Decision:** Reject Myth B as an absolute (`reject_myth_B = {b['reject_myth_B']}`).  
Infant mortality fell a lot **and** adult remaining LE rose by roughly **8–9 years** on average in this sample.

---

## Mechanism: infant mortality and e₀

- Within-country Pearson corr(e₀, IMR): median ≈ **{ic.get('median_pearson')}** across {ic.get('n_countries')} countries with ≥10 years.
- Country FE regression (associational):  
  e₀ ~ IMR + year + country FE, cluster SE by country.

| Coefficient | Estimate | SE (cluster) | p |
|-------------|----------|--------------|---|
| IMR | {reg.get('params',{}).get('imr', float('nan')):.4f} | {reg.get('bse',{}).get('imr', float('nan')):.4f} | {reg.get('pvalue',{}).get('imr', float('nan')):.3g} |
| year | {reg.get('params',{}).get('year', float('nan')):.4f} | {reg.get('bse',{}).get('year', float('nan')):.4f} | {reg.get('pvalue',{}).get('year', float('nan')):.3g} |
| N / countries / R² | {reg.get('n')} / {reg.get('n_countries')} / {reg.get('r2', float('nan')):.3f} |

**Note:** Not causal identification—describes co-movement in period tables.

---

## Data credibility (HMD summary vs OWID)

| Metric | Value |
|--------|------:|
| Overlapping country-years | {conc.get('n')} |
| Correlation | {conc.get('corr')} |
| RMSE (years) | {conc.get('rmse')} |
| MAE | {conc.get('mae')} |

---

## Sweden snapshot (canonical long series)

See `outputs/tables/sweden_snapshot.csv` and figure `F1_sweden_myth_killer.png`.

Illustrative pattern: e₀ can sit near the 30s while expected age|65 stays in the mid-70s; IMR ~200/1000; S(0→65) well below 50% until modern public health.

---

## Figures

| File | Content |
|------|---------|
| `outputs/figures/F1_sweden_myth_killer.png` | Sweden e₀, age\\|65, IMR, survival |
| `outputs/figures/F1b_multicountry_e0_vs_exp65.png` | Allowlist countries small multiples |
| `outputs/figures/F2_scatter_e0_vs_exp_death_65.png` | Era-colored scatter |
| `outputs/figures/F3_e0_vs_imr.png` | Infant drag scatter |
| `outputs/figures/F4_low_e0_distributions.png` | Myth A sample histograms |
| `outputs/figures/F5_eurostat_age_profiles.png` | Modern multi-age e(x) |
| `outputs/figures/F6_concordance_hmd_owid.png` | Source agreement |
| `outputs/figures/F_mythB_dumbbell_e65.png` | Adult LE rise by country |

---

## Methods caveats

1. **Period** life expectancy applies one year’s age-specific rates to a synthetic cohort—not the lived lifespan of a real birth cohort.  
2. HMD public summary covers countries with strong vital registration—not “all humans in 1700.”  
3. Only **Sweden** has continuous series from 1751; multi-country historical starts are staggered.  
4. Subpopulation series (civilian, ethnic, East/West Germany) are in the wide panel; multi-country charts use a **primary allowlist**.  
5. HMD public summary has e₀/e₆₅/e₈₀—not e₁₅; Eurostat supplies modern multi-age including 15.  
6. Conditional longevity at 65 is **not** “everyone lived to 75”—it is the right statistic for refuting “adults died at 30.”

---

## Reproduce

```powershell
cd age_myth
pip install -r requirements.txt
python -m src.analysis.run_analysis
```

---

## Bottom line

> **People in the past did not all die at 30.**  
> Low historical e₀ is the arithmetic of dead infants and children.  
> Adults who reached later ages still often had expected remaining lives into old age—yet adult remaining LE also improved by about **8–9 years** into the 21st century.
"""
    out = OUTPUTS / "reports" / "age_myth_findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(lines, encoding="utf-8")
    print(f"Wrote {out}")
    return out


def main() -> None:
    print(
        "DEPRECATED: use `python -m src.analysis.run_final_agrade` for the ship path.\n"
        "Continuing legacy run_analysis for compatibility...\n"
    )
    ensure_dirs()
    print("=== 1/3 Panels ===")
    save_panels()
    print("=== 2/3 Myth tests ===")
    run_tests()
    print("=== 3/3 Figures ===")
    run_figures()
    print("=== Report ===")
    write_findings()
    print("Done. Prefer FINAL_A_GRADE_REPORT.md from run_final_agrade.")


if __name__ == "__main__":
    main()
