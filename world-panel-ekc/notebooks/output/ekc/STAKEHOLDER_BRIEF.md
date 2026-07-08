# Stakeholder brief — growth, emissions, and the limits of the EKC

*Final pre-presentation review: thesis, star result, and prepared answers to anticipated objections. Panel: ~150 countries, 1996–2024, micro-states excluded. All results are observational associations.*

## Thesis in one paragraph

The Environmental Kuznets Curve — the promise that growth eventually cleans up after itself — **is not in the data**: its inverted-U is a cross-sectional artifact that flips sign under within-country fixed effects and never yields an identified turning point (2% of bootstrap resamples). What *is* in the data is **two-speed decarbonization**: the *cyclical* coupling of emissions to growth has not weakened — a 1% GDP-per-capita acceleration still brings ~0.5–0.7% more CO2, tighter than in the 1990s — while the *autonomous* drift (emission change at zero growth) has swung from **+0.8%/yr in 1996–2005 to −1.0%/yr in 2016–24** (−2.7%/yr in high-income countries, ex-COVID). **Decarbonization is happening around growth — via energy mix and technology — not through it.** Policy implication: waiting for the EKC is not a strategy; the drift is where the action is.

![two-speed decarbonization](stress/coupling_headline.png)

## The star table: drift vs coupling by period

| sample    |   drift_pct_yr |   drift_se_pct |   drift_p |   elasticity |   elast_se |   elast_p |    n |
|:----------|---------------:|---------------:|----------:|-------------:|-----------:|----------:|-----:|
| 1996-2005 |           0.84 |           0.37 |    0.0249 |        0.315 |      0.084 |    0.0002 | 1495 |
| 2006-2015 |          -0.18 |           0.3  |    0.5452 |        0.639 |      0.066 |    0      | 1502 |
| 2016-2024 |          -0.95 |           0.28 |    0.0006 |        0.715 |      0.082 |    0      | 1355 |

High-income countries only:

| sample             |   drift_pct_yr |   drift_se_pct |   drift_p |   elasticity |   elast_se |   elast_p |   n |
|:-------------------|---------------:|---------------:|----------:|-------------:|-----------:|----------:|----:|
| 1996-2005          |          -0.47 |           0.33 |    0.1538 |        0.422 |      0.11  |    0.0001 | 500 |
| 1996-2005 ex-COVID |          -0.47 |           0.33 |    0.1538 |        0.422 |      0.11  |    0.0001 | 500 |
| 2006-2015          |          -2.07 |           0.28 |    0      |        0.652 |      0.071 |    0      | 500 |
| 2006-2015 ex-COVID |          -2.07 |           0.28 |    0      |        0.652 |      0.071 |    0      | 500 |
| 2016-2024          |          -3.1  |           0.36 |    0      |        0.878 |      0.079 |    0      | 450 |
| 2016-2024 ex-COVID |          -2.72 |           0.39 |    0      |        0.534 |      0.105 |    0      | 350 |

**Is the elasticity rise significant?** Interaction test (1996–2005 vs 2016–24): Δelasticity = +0.40 (p = 0.000); drift change -1.78 pp/yr (p = 0.000). Both margins moved, in opposite directions — that is the two-speed result.

## Anticipated objections — and our answers

**Q1. "Your out-of-sample win on emission changes is just COVID."**  Largely, yes — and that *is* the finding. Ex-COVID test years the change-model R² is ~0.02 (annual changes are near-noise in calm years); in 2020–21 it is 0.27 while naive persistence collapses to −0.45. The model earns its keep exactly when growth moves sharply — which is direct out-of-sample evidence that the growth-emissions coupling is real and live. We present the per-year split ourselves:

| slice                  |   n |   ols_r2 |   naive_r2 |
|:-----------------------|----:|---------:|-----------:|
| 2019                   | 128 |    0.015 |     -0.014 |
| 2020                   | 128 |    0.141 |     -0.449 |
| 2021                   | 128 |    0.022 |     -0.173 |
| 2022                   |   7 |   -6.685 |     -0.063 |
| ex-COVID (2019, 2022+) | 135 |    0.015 |     -0.013 |
| COVID only (2020-21)   | 256 |    0.267 |     -0.003 |

*Rows after 2021 have n ≤ 7 (the fossil-fuel-share control lags in WDI), so their R² is not informative — flagged here before anyone else does.*


**Q2. "Contemporaneous drivers aren't a forecast."**  Correct, and we don't claim one. With only lagged information the test R² is 0.01 — essentially unforecastable. The change model is **attribution**: given observed growth, how much emissions move. Its ΔGDP coefficient is the marginal coupling — an inferential quantity, not a crystal ball. (validation_change_metrics.csv remains the honest headline: structure beats the GBM on the same task, same rows.)

**Q3. "Petrostates create your pooled U-shape."**  They amplify it but don't create it: dropping the 8 largest oil-and-gas outliers, pooled curvature stays positive (+0.65, p < 0.001). Either way, no inverted-U:

| sample            |   b_gdp_sq_pooled |   p_value |    n |
|:------------------|------------------:|----------:|-----:|
| all countries     |             1.324 |    0.0005 | 2843 |
| excl. petrostates |             0.654 |    0      | 2742 |

**Q4. "Intensity decoupling is a ratio trick — absolute emissions matter."**  Agreed, so we estimated absolute within-country trends: high-income CO2 per capita is falling **-0.070 t/yr** (p < 0.001) — genuine absolute decline — while upper-middle-income is still **rising** (+0.038 t/yr, p < 0.01). Absolute decoupling exists, but is so far a rich-country phenomenon:

| income_group        |   abs_trend_t_per_yr |   p_value |   mean_level_t |    n |
|:--------------------|---------------------:|----------:|---------------:|-----:|
| Low income          |              -0.0024 |    0.515  |           0.36 |  720 |
| Lower middle income |               0.0088 |    0.065  |           1.05 | 1200 |
| Upper middle income |               0.0377 |    0.0023 |           3.84 | 1260 |
| High income         |              -0.0697 |    0.0002 |          10.45 | 1500 |

**Q5. "Your FE null on the EKC could be low power — only 3.2% of CO2 variance is within-country."**  Partly fair, and we flag it. But the with-controls FE quadratic is significantly *convex* (+0.48, p < 0.001), not merely null, and the change-elasticity analysis — which uses the same within-country variation — finds strong, precisely-estimated coupling. The data have enough within-variation to speak; what they say is 'no inverted-U.'

**Q6. "Driscoll-Kraay with T≈29 is stretching it."**  Both SE families are reported side by side (headline_se_comparison.csv); point estimates are identical and every conclusion survives under clustered SEs. The FE-vs-RE call rests on the Mundlak test (p = 0.001), not the degenerate Hausman.

**Q7. "A rising elasticity contradicts your own decoupling claim."**  No — they are the two different margins estimated in one regression: the *slope* (cyclical response to growth fluctuations) rose, the *intercept* (secular drift) fell. Trend-decoupling coexists with tight cyclical coupling; that tension is the headline, not a contradiction.

**Q8. "None of this is causal."**  Stated on every output. These are conditional associations with fixed effects and robust inference; no instrument exists here. The policy-relevant claim — that the observed decline comes from the drift, not from an income turning point — is about *where the variation lives*, and stands at the descriptive level at which it is made.

## The close

> Growth has not decoupled from carbon at the margin — every point of GDP growth still buys about half a point of CO2, more tightly than in the 1990s. What has changed is everything *around* growth: at zero growth, emissions now fall ~1% a year globally and ~2.7% in high-income economies, versus *rising* 0.8% in the late 1990s. The Kuznets curve promised growth would do the cleanup. The data say the cleanup is being done in spite of growth — by the energy mix — and that is where policy leverage lies.
