# Literature map (cite-only benchmarks)

External studies and claims we **do not re-download as microdata**, but must answer in demos.

## Cardiometabolic / selection (existing Myth Lab focus)

| Theme | Typical sources | vs our NHANES tests |
|-------|-----------------|---------------------|
| ASB ↔ obesity / T2D | MESA, cohorts, reverse-causation reviews | M1/M2: BMI robust; HbA1c attenuates after diabetes exclusion |
| ASB ↔ CVD | NutriNet, various cohorts | M3: lipids/BP; selection strong |
| Consumer profile | AHA LCS advisory | M5/M8: diabetes, BMI, age, income |

## Cancer / aspartame (talking-point core)

| Claim people make | Source class | What the evidence actually is | Our public test |
|-------------------|--------------|-------------------------------|-----------------|
| “WHO says aspartame causes cancer” | IARC media 2023 | **2B possible hazard**, limited evidence; JECFA kept ADI | Hazard vs risk figures; ADI cans chart |
| “Artificial sweeteners raise cancer risk” | NutriNet-Santé (r/science) | Cohort **incidence** associations; residual confounding debated | Cite only; cannot re-run |
| “ASB → liver cancer” | UK Biobank / media; IARC HCC discussion | Site-specific, limited human evidence | We lack HCC incidence; only ever-cancer + cancer death |
| “Diet soda cancer death” | Meta-analyses | **Cancer mortality often null** while all-cause/CVD mixed | Cox + logistic on LMF cancer death |
| Mouse cancer | Toxicology headlines | High dose vs human soda | Dose/ADI communication |
| NHANES LCS & disease | Fulgoni et al. type papers | Cross-section: LCS users sicker/heavier | Matches our M5 selection story |

## Method papers (engineering)

| Topic | Why we care |
|-------|-------------|
| LCSB classification inconsistency (Swithers / Bonanno / NHANES) | WWEIA 7102 primary; multiverse exposures |
| ML + survey weights (MacNell et al.) | Weighted vs unweighted ML |
| CDC NHANES weighting tutorials | Multi-cycle weights; design SEs still approximate here |

## How to use this file in a demo

1. Point at the **viral claim** row.  
2. Say what study class it comes from.  
3. Show **our** corresponding figure/test.  
4. State the limit in one sentence.  
