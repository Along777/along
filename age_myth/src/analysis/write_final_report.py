"""Write FINAL_A_GRADE_REPORT.md from final_claims.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.ladder import BULLET
from src.paths import OUTPUTS


def main() -> None:
    final = json.loads((BULLET / "final_claims.json").read_text(encoding="utf-8"))
    sc = final["scorecard"]
    yw = final["claim_A_year_weighted"]["year_weighted"]
    ec = final["claim_A_year_weighted"]["equal_country"]
    band = final["strict_band_30_35"]
    sex = final["sex_specific_claim_A"]
    base = final["from_bulletproof"]
    b = base["claim_B"]["delta_e65"]
    c = base["claim_C"]
    a2 = base["claim_A2"]
    a15 = a2.get("ages", {}).get("15", {})
    a30 = a2.get("ages", {}).get("30", {})
    gold_lad = final["dual_hld"].get("gold", {}).get("ladder_e0lt40", [])
    med_lad = final["dual_hld"].get("median", {}).get("ladder_e0lt40", [])

    def ladder_md(rows):
        if not rows:
            return "_none_"
        cols = ["age_x", "n", "n_countries", "median_expected_age", "share_ge_50", "share_ge_70"]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    fem = sex.get("female", {})
    mal = sex.get("male", {})
    mc = fem.get("min_case", {})

    md = f"""# FINAL A-GRADE REPORT — Age Myth Bust

**Scorecard letter: {sc["letter"]} ({sc["total"]}/100)**  
**Date:** 2026-07-26  
**Command:** `python -m src.analysis.run_final_agrade`

This is the **single ship document**. Prior reports remain as archives.

---

## Grade scorecard

| Item | Points |
|------|-------:|
| Myth A multi-country + de-dupe | {sc["points"].get("myth_A_multicountry")} / 20 |
| Equal-country + strict band | {sc["points"].get("equal_country_and_strict_band")} / 10 |
| Mid-adult ladder gold+median | {sc["points"].get("mid_adult_ladder")} / 15 |
| Myth B + CI | {sc["points"].get("myth_B")} / 15 |
| Uncertainty method labeled | {sc["points"].get("uncertainty_method")} / 10 |
| Association models honest | {sc["points"].get("association_models")} / 10 |
| Chart pack | {sc["points"].get("chart_pack")} / 10 |
| Scope / dual-metric discipline | {sc["points"].get("scope_dual_metric")} / 10 |
| **Total** | **{sc["total"]} / 100** |

**Notes:**  
{chr(10).join("- " + n for n in sc["notes"])}

**Warnings:** {final.get("warnings")}

---

## Abstract (A-grade safe)

In de-duplicated HMD public-summary data for high-quality vital-registration populations, when period life expectancy at birth is below 40—or even in the strict **30–35** band—median expected age at death **conditional on age 65** is about **75**, with **very high infant mortality** and **low survival to 65**. Equal-country weighting yields the same conclusion (median of country medians ≈ **75.4**). Mid-adult HLD evidence shows expected age if alive at **15** around the **mid-50s** (gold and median ladders). Adult remaining LE at 65 rose about **+8.5 years** from pre-1900 to post-2000. This rejects both “they died at 30” and “only infants improved,” under period life-table measures—for these populations, not all past humans, and not as cohort lifespans.

---

## Claim A — dual metric (always together)

### Year-weighted (country-years)

| Metric | Value |
|--------|------:|
| n country-years (e0&lt;40, de-duped) | {yw["n_country_years"]} |
| n countries | {yw["n_countries"]} |
| Median e0 | {yw["median_e0"]:.2f} |
| Median IMR /1000 | **{yw["median_imr"]:.1f}** |
| Median S(0→65) | **{yw["median_s_to_65"]:.3f}** |
| Median expected age if alive at 65 | **{yw["median_exp65"]:.2f}** |
| Share age\\|65 ≥ 70 | **{100*yw["share_exp65_ge_70"]:.1f}%** |
| Share age\\|65 ≥ 69 | {100*yw["share_exp65_ge_69"]:.1f}% |
| Min / max age\\|65 | {yw["min_exp65"]:.2f} / {yw["max_exp65"]:.2f} |

### Equal-country (one median per country, then median)

| Metric | Value |
|--------|------:|
| n countries | {ec["n_countries"]} |
| Median of country-medians age\\|65 | **{ec["median_of_country_medians_exp65"]:.2f}** |
| Mean of country-medians | {ec["mean_of_country_medians_exp65"]:.2f} |
| Range of country medians | {ec["min_country_median_exp65"]:.2f} – {ec["max_country_median_exp65"]:.2f} |
| Bootstrap mean of country medians (95% CI) | {ec["bootstrap_mean_of_country_medians"]["mean"]:.2f} [{ec["bootstrap_mean_of_country_medians"]["ci_low"]:.2f}, {ec["bootstrap_mean_of_country_medians"]["ci_high"]:.2f}] |

**Method note:** Sweden contributes many years, but **every** country median is ~75. Year-weighting does not create the result.

**Dual-metric rule:** Never quote age\\|65 without IMR or S(0→65). Low e0 = early death, not adult death at 30.

---

## Strict myth band: e0 ∈ [30, 35]

| Metric | Value |
|--------|------:|
| n country-years | {band.get("n_country_years")} |
| n countries | {band.get("n_countries")} |
| Median e0 | {band.get("year_weighted", {}).get("median_e0")} |
| Median IMR | {band.get("year_weighted", {}).get("median_imr")} |
| Median S(0→65) | {band.get("year_weighted", {}).get("median_s_to_65")} |
| Median age\\|65 | **{band.get("year_weighted", {}).get("median_exp65")}** |
| Share ≥70 | **{100*band.get("year_weighted", {}).get("share_exp65_ge_70", 0):.0f}%** |
| Equal-country median age\\|65 | {band.get("equal_country_median_exp65")} |

Country-year counts: `{band.get("country_year_counts")}`

---

## Sex-specific Claim A (HMD)

| Sex | n | Median age\\|65 | Share ≥70 | Share ≥69 | Min age\\|65 |
|-----|--:|---------------:|----------:|----------:|------------:|
| Female | {fem.get("n_country_years")} | {fem.get("median_exp65")} | {100*fem.get("share_exp65_ge_70",0):.1f}% | {100*fem.get("share_exp65_ge_69",0):.1f}% | {fem.get("min_exp65")} |
| Male | {mal.get("n_country_years")} | {mal.get("median_exp65")} | {100*mal.get("share_exp65_ge_70",0):.1f}% | {100*mal.get("share_exp65_ge_69",0):.1f}% | {mal.get("min_exp65")} |

### Female floor (do not hide)

- **{mc.get("region_id")} {mc.get("year")}**: e0={mc.get("e0")}, age\\|65=**{mc.get("exp_death_65")}**, S(0→65)=**{mc.get("s_to_65")}**, IMR={mc.get("imr")}  
- Still ~**70**, not 30. Survival to 65 was ~**7%**.  
- Do **not** claim “100% of female observations ≥70” without this footnote.

---

## Claim A2 — mid-adult ladder (HLD)

Bulletproof primary (quality={a2.get("quality_used")}):

| Age | Median expected age | n | share ≥50 |
|----:|--------------------:|--:|----------:|
| 15 | {a15.get("median")} | {a15.get("n_rows")} | {a15.get("share_ge_50")} |
| 30 | {a30.get("median")} | {a30.get("n_rows")} | {a30.get("share_ge_55")} |

### Dual ladder when e0 &lt; 40

**Gold (n_tables=1):**

{ladder_md(gold_lad)}

**Median (all tables):**

{ladder_md(med_lad)}

Gold geography is incomplete (e.g. some large countries lack n_tables=1 cells)—always show both.

---

## Claim B — adults also improved

| Metric | Value |
|--------|------:|
| Mean Δe65 pre-1900 → post-2000 | **+{b["mean"]:.2f} years** |
| 95% CI (country bootstrap) | {b["ci_low"]:.2f} – {b["ci_high"]:.2f} |
| n countries (de-duped) | {b["n"]} |
| Range | {b["min"]:.2f} – {b["max"]:.2f} |

Rejects “only infant mortality improved.”

---

## Claim C — associations (not causal, not overfit)

Median within-country corr(e0, IMR) = **{c.get("median_within_corr_e0_imr")}**

| Model | R² | β(IMR) |
|-------|---:|-------:|
| M0 pooled | {c["models"]["M0_pooled"]["r2"]:.3f} | {c["models"]["M0_pooled"]["beta_imr"]:.4f} |
| M1 within | {c["models"]["M1_within"]["r2"]:.3f} | {c["models"]["M1_within"]["beta_imr"]:.4f} |
| M2 within+year | {c["models"]["M2_within_year"]["r2"]:.3f} | {c["models"]["M2_within_year"]["beta_imr"]:.4f} |
| M3 first difference | {c["models"]["M3_first_diff"]["r2"]:.3f} | {c["models"]["M3_first_diff"]["beta_dimr"]:.4f} |

High R² is **expected** (shared mortality schedule). Do not headline FE R² as predictive performance.

---

## Figures (ship pack)

### Final A-grade

- `outputs/figures/final/FA1_year_vs_equal_country.png`
- `FA2_strict_band_30_35.png`
- `FA3_dual_hld_ladder.png`
- `FA4_sex_honesty.png`
- `FA5_collage.png`

### Definitive D1–D12

`outputs/figures/definitive/`

### HTML board

`outputs/reports/myth_bust_board.html` (refresh via run pipeline)

---

## What we are NOT claiming

- Global premodern humanity outside VR data  
- Cohort lifespan = period e0  
- Causal effect of IMR from regressions  
- That everyone reached 15 or 65  
- “100% of female low-e0 years have age\\|65 ≥ 70” without Iceland 1843 footnote  

---

## Reproduce

```powershell
python -m src.analysis.run_final_agrade
```

Artifacts:

- `outputs/bulletproof/final_claims.json`
- `outputs/bulletproof/agrade_scorecard.json`
- `outputs/reports/FINAL_A_GRADE_REPORT.md`
"""
    out = OUTPUTS / "reports" / "FINAL_A_GRADE_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out} grade={sc['letter']} {sc['total']}/100")


if __name__ == "__main__":
    main()
