# Age Myth Findings Report

**Data:** HMD public summary indicators via `life_expectancy_modeling`  
**Question:** Did people in the past “only live to 30–35”?

---

## Executive answer

**No.** Period life expectancy near 30–40 is an average **dragged down by catastrophic infant and child mortality**, not the typical age of adult death.

In every high-quality country-year where e₀ < 40 in this dataset:

- Median expected age at death **if alive at 65** is about **75.5 years**  
  (95% bootstrap CI [75.4, 75.6]).
- **100%** of those country-years have expected age|65 ≥ 70.
- Median survival from birth to 65 is only about **27%** — many never reached old age.
- Median infant mortality is about **193** per 1,000 births.

A second claim — “if you survived childhood you lived as long as modern adults” — is also **false as an absolute**:

- Across **14** countries with both pre-1900 and post-2000 data, mean remaining LE at 65 rose by about **8.5 years**  
  (95% CI [7.9, 9.2]; range about 6.4–10.7).

So both the crude myth and the popular overcorrection fail.

---

## Myth A — “They died at 30”

### Test sample

- Country-years with e₀ < 40 (both sexes, HMD public summary): **n = 247**  
- Countries represented: **13**  
- Of which e₀ in [30, 35]: **45**

### Results

| Metric | Value |
|--------|------:|
| Median e₀ | 37.56 |
| Median IMR (per 1,000) | 193.0 |
| Median S(0→65) | 0.274 |
| Median expected age if alive at 65 | **75.45** |
| 95% CI (bootstrap median) | [75.36, 75.64] |
| Share expected age\|65 ≥ 70 | **100.0%** |
| Median adult gap (age\|65 − e₀) | 38.0 years |

**Decision:** Reject Myth A (`reject_myth_A = True`).

### How to read this without overclaiming

Expected age at death *conditional on age 65* does **not** mean most people reached 65.  
S(0→65) near 0.27 means most births did **not** survive to 65 under those period rates.  
The myth error is equating **e₀** with “when adults died.”

---

## Myth B — “Only infant mortality improved; adults always lived modern lengths”

| Metric | Value |
|--------|------:|
| Countries (pre-1900 & post-2000 e₆₅) | 14 |
| Mean Δe₆₅ | **+8.52 years** |
| 95% CI | [7.88, 9.17] |
| Median Δe₆₅ | +8.44 |
| Range | 6.36 to 10.66 |

**Decision:** Reject Myth B as an absolute (`reject_myth_B = True`).  
Infant mortality fell a lot **and** adult remaining LE rose by roughly **8–9 years** on average in this sample.

---

## Mechanism: infant mortality and e₀

- Within-country Pearson corr(e₀, IMR): median ≈ **-0.9388352153210587** across 50 countries with ≥10 years.
- Country FE regression (associational):  
  e₀ ~ IMR + year + country FE, cluster SE by country.

| Coefficient | Estimate | SE (cluster) | p |
|-------------|----------|--------------|---|
| IMR | -0.1224 | 0.0158 | 9.89e-15 |
| year | 0.1217 | 0.0200 | 1.14e-09 |
| N / countries / R² | 4966 / 50 / 0.968 |

**Note:** Not causal identification—describes co-movement in period tables.

---

## Data credibility (HMD summary vs OWID)

| Metric | Value |
|--------|------:|
| Overlapping country-years | 3721 |
| Correlation | 0.9998954236898089 |
| RMSE (years) | 0.1966260527628543 |
| MAE | 0.06267823165815653 |

---

## Sweden snapshot (canonical long series)

See `outputs/tables/sweden_snapshot.csv` and figure `F1_sweden_myth_killer.png`.

Illustrative pattern: e₀ can sit near the 30s while expected age|65 stays in the mid-70s; IMR ~200/1000; S(0→65) well below 50% until modern public health.

---

## Figures

| File | Content |
|------|---------|
| `outputs/figures/F1_sweden_myth_killer.png` | Sweden e₀, age\|65, IMR, survival |
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
