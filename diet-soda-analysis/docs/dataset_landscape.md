# Dataset landscape

Public and semi-public sources relevant to diet soda / artificial sweeteners, cardiometabolic outcomes, and cancer claims.  
Compiled before Phase 1A build (web / literature / X / Reddit scan).

## Tier A — Core (this project)

| Source | Why |
|--------|-----|
| NHANES 2011–2018 | Individual diet + measured biomarkers + design weights |
| WWEIA Food Categories | Official **7102** diet soft drinks / **7202** soft drinks |
| FNDDS / DRXFCD | Food codes and descriptions |
| NCHS public Linked Mortality Files | Cancer / all-cause mortality follow-up |

## Tier B — Recommended complements

| Source | Why |
|--------|-----|
| Open Food Facts | Product-level sweeteners in diet sodas (myth context) |
| Longer NHANES + LMF | More mortality events if 2011–2018 is sparse |

## Tier C — Optional

| Source | Why | Limit |
|--------|-----|--------|
| BRFSS SSB module | State regular-soda maps | Often excludes diet soda |
| NHIS CCS | SSB frequency | Weak ASB |
| FDA CAERS | Spontaneous food AE reports | No rates; bias |

## Cite only (not fully public microdata)

UK Biobank, NutriNet-Santé, Nurses’ Health Study, HPFS, WHI, MESA, CARDIA, All of Us controlled tiers.

These drive much of the social-media narrative; we benchmark *against* published results, we do not re-download their individual data.
