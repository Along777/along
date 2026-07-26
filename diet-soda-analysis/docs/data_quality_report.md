# Data quality report (Phase 1B)

Generated: 2026-07-26T18:07:52.801884+00:00

## Exclusion flow

| step                                 |     n |
|:-------------------------------------|------:|
| All merged rows                      | 39156 |
| Age >= 20                            | 22617 |
| + not pregnant                       | 22370 |
| + MEC weight > 0                     | 21399 |
| + Day-1 dietary exposure             | 19384 |
| + diet recall status OK (or missing) | 19384 |
| Analytic sample (in_analytic)        | 19384 |
| Fasting subsample                    |  8528 |

- Analytic n: **19384**
- Fasting n: **8528**
- Mortality-eligible analytic n: **19333**

## Beverage groups (analytic)

```json
{
  "Neither": 11558,
  "SSB-only": 5934,
  "ASB-only": 1744,
  "Both": 148
}
```

## Missingness in analytic sample (%)

| Variable | pct_missing |
|----------|------------|
| bmi | 1.0 |
| hba1c | 3.7 |
| sbp_mean | 2.4 |
| hdl | 4.9 |
| tg | 54.6 |
| glucose | 53.4 |
| cancer_ever | 0.0 |
| asb_any_d1 | 0.0 |

## Notes

- 2011–2012 insulin from GLU file.
- Multi-cycle weight = WTMEC2YR / 4.
- Cancer mortality from public LMF (perturbed for some records per NCHS).
