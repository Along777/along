"""Markdown report for bulletproof myth bust."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.ladder import BULLET
from src.paths import OUTPUTS


def main() -> None:
    claims = json.loads((BULLET / "claims.json").read_text(encoding="utf-8"))
    a = claims["claim_A"]
    a2 = claims["claim_A2"]
    b = claims["claim_B"]["delta_e65"]
    c = claims["claim_C"]
    gates = claims["gates"]
    snap = claims["sweden_1800"]
    a15 = a2.get("ages", {}).get("15", {})
    a30 = a2.get("ages", {}).get("30", {})
    a65h = a2.get("ages", {}).get("65", {})

    ladder_path = None
    for name in ("ladder_hld_gold_agg_e0lt40.csv", "ladder_hld_median_agg_e0lt40.csv"):
        if (BULLET / name).exists():
            ladder_path = BULLET / name
            break
    def _md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(str(c) for c in cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    ladder_md = ""
    if ladder_path:
        lad = pd.read_csv(ladder_path)
        ladder_md = _md_table(lad)
    sens = ""
    if (BULLET / "sensitivity_matrix.csv").exists():
        sens = _md_table(pd.read_csv(BULLET / "sensitivity_matrix.csv"))

    md = f"""# MYTH BUST — BULLETPROOF

**Gates:** `all_pass = {gates.get("all_pass")}`  
**Reproduce:** `python -m src.analysis.run_next_level`

---

## Abstract

Using de-duplicated HMD public summary life-table indicators for high-quality vital-registration populations, country-years with period life expectancy at birth below 40 (**n={a["n_country_years"]}**, **{a["n_countries"]} countries**) combine very high infant mortality (median **{a["median_imr"]:.0f}/1000**) and low survival to age 65 (median **{a["median_s_to_65"]:.2f}**) with conditional expected age at death if alive at 65 of **{a["exp_death_65"]["median"]:.1f}** years (country-cluster bootstrap 95% CI **{a["exp_death_65"]["ci_low"]:.1f}–{a["exp_death_65"]["ci_high"]:.1f}**). **{100*a["share_exp65_ge_70"]:.0f}%** of those observations have age|65 ≥ 70. Mid-adult HLD evidence when e0&lt;40 shows median expected age if alive at 15 of **{a15.get("median", float("nan")):.1f}** (quality={a2.get("quality_used")}). Separately, remaining LE at 65 rose by **+{b["mean"]:.1f}** years from pre-1900 to post-2000 (n={b["n"]} countries; CI {b["ci_low"]:.1f}–{b["ci_high"]:.1f}). Associations between e0 and IMR are strong (median within-country r={c["median_within_corr_e0_imr"]:.2f}) but associational—not causal. High R² when predicting e0 from IMR is expected from shared mortality schedules, not ML overfitting.

---

## Claim cards

### A — Birth e0 is not adult death age  [{ "PASS" if gates["A"]["pass"] else "FAIL" }]

| Metric | Value |
|--------|------:|
| n country-years (e0&lt;40, de-duped) | {a["n_country_years"]} |
| n countries | {a["n_countries"]} |
| Median e0 | {a["median_e0"]:.2f} |
| Median IMR | {a["median_imr"]:.1f} |
| Median S(0→65) | {a["median_s_to_65"]:.3f} |
| Median expected age if alive at 65 | **{a["exp_death_65"]["median"]:.2f}** |
| Cluster bootstrap 95% CI | {a["exp_death_65"]["ci_low"]:.2f} – {a["exp_death_65"]["ci_high"]:.2f} |
| Share age\\|65 ≥ 70 | **{100*a["share_exp65_ge_70"]:.1f}%** |
| Median adult gap (age\\|65 − e0) | {a["adult_gap_65"]["median"]:.1f} |

**Must read with S(0→65):** most births never reach 65 under those period rates.

### A2 — Mid-adult hole closed  [{ "PASS" if gates["A2"]["pass"] else "FAIL" }]

| Age x | Median expected age if alive at x | n | share ≥50 |
|------:|----------------------------------:|--:|----------:|
| 15 | {a15.get("median")} | {a15.get("n_rows")} | {a15.get("share_ge_50")} |
| 30 | {a30.get("median")} | {a30.get("n_rows")} | {a30.get("share_ge_55")} |
| 65 (HLD) | {a65h.get("median")} | {a65h.get("n_rows")} | {a65h.get("share_ge_70")} |

HLD quality used: **{a2.get("quality_used")}** (gold = n_tables==1 preferred).

### B — Adults also improved  [{ "PASS" if gates["B"]["pass"] else "FAIL" }]

| Metric | Value |
|--------|------:|
| Mean Δe65 (post-2000 − pre-1900) | **+{b["mean"]:.2f}** |
| 95% CI | {b["ci_low"]:.2f} – {b["ci_high"]:.2f} |
| n countries | {b["n"]} |
| Range | {b["min"]:.2f} – {b["max"]:.2f} |

### C — Infant mechanism (associational)  [{ "PASS" if gates["C"]["pass"] else "FAIL" }]

| Model | R² | β(IMR) |
|-------|---:|-------:|
| M0 pooled | {c["models"]["M0_pooled"]["r2"]:.3f} | {c["models"]["M0_pooled"]["beta_imr"]:.4f} |
| M1 within | {c["models"]["M1_within"]["r2"]:.3f} | {c["models"]["M1_within"]["beta_imr"]:.4f} |
| M2 within+year | {c["models"]["M2_within_year"]["r2"]:.3f} | {c["models"]["M2_within_year"]["beta_imr"]:.4f} |
| M3 first diff | {c["models"]["M3_first_diff"]["r2"]:.3f} | {c["models"]["M3_first_diff"]["beta_dimr"]:.4f} |

Out-of-time (pre/post 1950) same sign on β(IMR): {c.get("out_of_time", {}).get("same_sign")}

---

## Sweden 1800 anchor  [{ "PASS" if gates.get("S", {}).get("pass") else "FAIL" }]

| Metric | Value |
|--------|------:|
| e0 | {snap.get("e0")} |
| IMR | {snap.get("imr")} |
| Expected age if alive at 65 | {snap.get("exp_death_65")} |
| S(0→65) | {snap.get("s_to_65")} |

---

## Age ladder when e0 &lt; 40

{ladder_md}

---

## Sensitivity matrix

{sens}

---

## Figures

All under `outputs/figures/definitive/` (D1–D12).  
Interactive board: `outputs/reports/myth_bust_board.html`.

---

## What we are NOT claiming

{chr(10).join("- " + x for x in claims["scope"]["not_claiming"])}

---

## Reproduce

```powershell
python -m src.analysis.run_next_level
```
"""
    out = OUTPUTS / "reports" / "MYTH_BUST_BULLETPROOF.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
