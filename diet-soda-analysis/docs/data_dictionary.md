# Data dictionary (analysis-ready)

One row = one NHANES participant. Primary file: `data/processed/analysis_ready.parquet`
(filter `in_analytic == True` already applied in that file).

## Keys & design
| Variable | Description |
|----------|-------------|
| SEQN | Participant ID |
| cycle | NHANES cycle years |
| SDMVPSU, SDMVSTRA | Survey design |
| WTMEC2YR, w_mec | 2-year MEC weight; multi-cycle weight /4 |
| wtsaf2yr, w_fast | Fasting subsample weights |

## Exposure (Day-1 primary)
| Variable | Description |
|----------|-------------|
| asb_any_d1 | Any WWEIA 7102 diet soft drinks Day-1 |
| ssb_any_d1 | Any WWEIA 7202 soft drinks Day-1 |
| bev_group | ASB-only / SSB-only / Both / Neither |
| asb_g_d1, ssb_g_d1 | Grams Day-1 |
| asb_serv_d1 | asb_g_d1 / 355 |
| asb_broad_*, ssb_broad_* | Broader beverage categories |
| asb_fndds_kw_* | Description-keyword sensitivity |
| water_g_d1 | Tap + bottled water grams |
| asb_either_day | Diet soft on Day-1 or Day-2 |

## Outcomes
| Variable | Description |
|----------|-------------|
| bmi, waist | Body measures |
| sbp_mean, dbp_mean | Mean exam BP |
| hba1c, glucose, insulin, homa_ir | Glycemic |
| hdl, tc, tg, ldl | Lipids |
| obesity, hba1c_elevated, hypertension_bp | Binary thresholds |
| cancer_ever | MCQ220 ever cancer |
| mortstat, cancer_death, allcause_death, permth_exm | LMF mortality |

## Covariates
| Variable | Description |
|----------|-------------|
| age, sex, female, race_eth, education, pir | Demographics |
| smoking_status | 0 never / 1 former / 2 current |
| sedentary_min | PAD680 |
| total_kcal_d1 | Day-1 energy |
| diabetes_sr | Self-report diabetes |
| pregnancy_status | RIDEXPRG |

## Sample flags
| Variable | Description |
|----------|-------------|
| in_analytic | Adults ≥20, not pregnant, MEC weight, Day-1 exposure, diet status OK |
| in_fasting | Analytic + fasting weight |
| lmf_eligible | eligstat==1 |
