# MYTH BUST — BULLETPROOF

**Gates:** `all_pass = True`  
**Reproduce:** `python -m src.analysis.run_next_level`

---

## Abstract

Using de-duplicated HMD public summary life-table indicators for high-quality vital-registration populations, country-years with period life expectancy at birth below 40 (**n=211**, **11 countries**) combine very high infant mortality (median **196/1000**) and low survival to age 65 (median **0.27**) with conditional expected age at death if alive at 65 of **75.4** years (country-cluster bootstrap 95% CI **75.1–75.7**). **100%** of those observations have age|65 ≥ 70. Mid-adult HLD evidence when e0&lt;40 shows median expected age if alive at 15 of **55.9** (quality=gold). Separately, remaining LE at 65 rose by **+8.5** years from pre-1900 to post-2000 (n=12 countries; CI 7.7–9.1). Associations between e0 and IMR are strong (median within-country r=-0.94) but associational—not causal. High R² when predicting e0 from IMR is expected from shared mortality schedules, not ML overfitting.

---

## Claim cards

### A — Birth e0 is not adult death age  [PASS]

| Metric | Value |
|--------|------:|
| n country-years (e0&lt;40, de-duped) | 211 |
| n countries | 11 |
| Median e0 | 37.00 |
| Median IMR | 195.9 |
| Median S(0→65) | 0.270 |
| Median expected age if alive at 65 | **75.39** |
| Cluster bootstrap 95% CI | 75.12 – 75.70 |
| Share age\|65 ≥ 70 | **100.0%** |
| Median adult gap (age\|65 − e0) | 38.3 |

**Must read with S(0→65):** most births never reach 65 under those period rates.

### A2 — Mid-adult hole closed  [PASS]

| Age x | Median expected age if alive at x | n | share ≥50 |
|------:|----------------------------------:|--:|----------:|
| 15 | 55.9225 | 44 | 0.8409090909090909 |
| 30 | 60.18 | 49 | 0.8979591836734694 |
| 65 (HLD) | 74.8125 | 44 | 1.0 |

HLD quality used: **gold** (gold = n_tables==1 preferred).

### B — Adults also improved  [PASS]

| Metric | Value |
|--------|------:|
| Mean Δe65 (post-2000 − pre-1900) | **+8.45** |
| 95% CI | 7.74 – 9.14 |
| n countries | 12 |
| Range | 6.36 – 10.66 |

### C — Infant mechanism (associational)  [PASS]

| Model | R² | β(IMR) |
|-------|---:|-------:|
| M0 pooled | 0.918 | -0.2092 |
| M1 within | 0.914 | -0.2111 |
| M2 within+year | 0.951 | -0.1211 |
| M3 first diff | 0.507 | -0.1126 |

Out-of-time (pre/post 1950) same sign on β(IMR): True

---

## Sweden 1800 anchor  [PASS]

| Metric | Value |
|--------|------:|
| e0 | 32.19 |
| IMR | 227.06 |
| Expected age if alive at 65 | 73.73 |
| S(0→65) | 0.2092 |

---

## Age ladder when e0 &lt; 40

| age_x | n | n_countries | median_e0 | median_e_x | median_expected_age | p10_expected_age | p90_expected_age | share_ge_45 | share_ge_50 | share_ge_60 | share_ge_70 | e0_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 49.0 | 18.0 | 35.075 | 35.075 | 35.075 | 30.083 | 39.088 | 0.0 | 0.0 | 0.0 | 0.0 | 40.0 |
| 1.0 | 47.0 | 16.0 | 35.07 | 42.86 | 43.86 | 37.905 | 48.548 | 0.4468085106382978 | 0.0 | 0.0 | 0.0 | 40.0 |
| 5.0 | 49.0 | 18.0 | 35.075 | 46.775000000000006 | 51.775000000000006 | 44.314 | 55.407 | 0.8571428571428571 | 0.6530612244897959 | 0.0 | 0.0 | 40.0 |
| 15.0 | 44.0 | 16.0 | 35.645 | 40.9225 | 55.9225 | 48.781 | 59.059 | 0.9772727272727272 | 0.8409090909090909 | 0.0227272727272727 | 0.0 | 40.0 |
| 20.0 | 49.0 | 18.0 | 35.075 | 37.010000000000005 | 57.010000000000005 | 50.133 | 60.251 | 1.0 | 0.8979591836734694 | 0.1836734693877551 | 0.0 | 40.0 |
| 30.0 | 49.0 | 18.0 | 35.075 | 30.18 | 60.18 | 54.993 | 63.312 | 1.0 | 1.0 | 0.5510204081632653 | 0.0 | 40.0 |
| 50.0 | 49.0 | 18.0 | 35.075 | 17.814999999999998 | 67.815 | 64.922 | 69.686 | 1.0 | 1.0 | 1.0 | 0.0612244897959183 | 40.0 |
| 65.0 | 44.0 | 16.0 | 35.645 | 9.8125 | 74.8125 | 73.4175 | 76.0755 | 1.0 | 1.0 | 1.0 | 1.0 | 40.0 |

---

## Sensitivity matrix

| variant | e0_threshold | n | n_countries | median_exp65 | share_ge_70 | median_s65 | median_imr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deduped_all | 40 | 211 | 11 | 75.39 | 1.0 | 0.2697 | 195.94 |
| deduped_all | 35 | 63 | 8 | 74.95 | 1.0 | 0.2133 | 218.2 |
| primary_allowlist | 40 | 208 | 10 | 75.4 | 1.0 | 0.2694 | 196.17 |
| primary_allowlist | 35 | 62 | 7 | 74.965 | 1.0 | 0.2137 | 218.595 |
| female | 40 | 150 | 10 | 75.37 | 0.9933333333333332 | 0.2744 | 192.135 |
| female | 35 | 46 | 7 | 74.83 | 0.9782608695652174 | 0.21815 | 210.55 |
| male | 40 | 295 | 12 | 75.31 | 1.0 | 0.2605 | 202.07 |
| male | 35 | 94 | 9 | 74.76 | 1.0 | 0.1869 | 224.485 |
| pre_1850_only | 40 | 101 | 6 | 75.38 | 1.0 | 0.2683 | 194.57 |

---

## Figures

All under `outputs/figures/definitive/` (D1–D12).  
Interactive board: `outputs/reports/myth_bust_board.html`.

---

## What we are NOT claiming

- Global premodern humanity outside VR data
- Cohort (lived) lifespan equal to period e0
- Causal effect of IMR on e0 from regressions
- Everyone reached age 65 historically

---

## Reproduce

```powershell
python -m src.analysis.run_next_level
```
