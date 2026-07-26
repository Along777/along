# Stakeholder defense — cautious first showing

## Open with (60 seconds)

1. Question: Does historical life expectancy ≈ 30–35 mean adults typically died then?  
2. Scope: High-quality vital registration (HMD), **period** life tables—not all past humans.  
3. Sweden figure D1.  
4. Dual number: when e0 was low, expected age **if alive at 65** ≈ **75**, but only ~**27%** reached 65; IMR ~**200**/1000.

**Tone:** careful correction of a misread statistic—not “we destroyed the myth forever.”

## Never open with

- Self-grade “A 100/100”  
- “People lived to 75”  
- “R² proves infants caused e0”  
- Live full pipeline cold-start  
- Global premodern humanity claims  

## Red-team answers

| Attack | Answer |
|--------|--------|
| “So they lived to 75” | Conditional on age 65; only ~27% got there when e0 was low |
| “Only Sweden” | 11 countries; equal-country medians all ~74.8–76.1 |
| “Whole world?” | HMD VR populations only—best long series, not a world census |
| “Overfit model” | No causal ML model; e0 and IMR share mortality schedules |
| “Self-graded A” | Internal engineering checklist; judge the evidence tables |
| “HLD is junk” | Primary is HMD summary; HLD only supplements ages 15/30 |
| “Prove causality” | Out of scope; we correct how period e0 is interpreted |
| “Where are tests?” | `pytest -q`; claim gates fail the build if core claims regress |

## Six-stop demo order

1. Scope + question  
2. `D1_sweden_storyboard.png`  
3. `FA1_year_vs_equal_country.png`  
4. `FA2_strict_band_30_35.png`  
5. `FA3_dual_hld_ladder.png` or `D2_age_ladder...`  
6. `D10_mythB_dumbbell.png` + limits  

Backup if asked: `FA4_sex_honesty.png` (Iceland 1843), coverage D11.

## Leave-behind numbers (memorize)

- e0&lt;40: n=211, 11 countries; age|65 median **75.4**; S65 **0.27**; IMR **~196**  
- Equal-country median of medians **75.4**  
- Strict e0 in 30–35: age|65 **~75**, S65 **~0.22**  
- Δe65 pre-1900→post-2000: **+8.5 years** (≈12 countries)  
- Ship command: `python -m src.analysis.run_final_agrade`  
- Report: `outputs/reports/FINAL_A_GRADE_REPORT.md`  
