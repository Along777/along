# Data Inventory Report (Phase 1A)

**Generated (UTC):** 2026-07-25T23:20:06.450676+00:00

This report is produced **before** heavy cleaning or analysis. It documents what was downloaded and basic coverage.

## Scope

- Cycles: 2011-2012, 2013-2014, 2015-2016, 2017-2018
- Planned NHANES file stems: DEMO, DR1IFF, DR2IFF, DRXFCD, DR1TOT, DR2TOT, BMX, BPX, GHB, GLU, INS, HDL, TCHOL, TRIGLY, SMQ, PAQ, DIQ, MCQ, ALQ, BPQ

## NHANES file load summary

- Successfully loaded: **79 / 80**

### Missing or failed files

| cycle     | stem   | error            |
|:----------|:-------|:-----------------|
| 2011-2012 | INS    | missing_or_empty |

**Note:** For 2011–2012, fasting **insulin is in `GLU_G`** (`LBXIN`), not a separate `INS_G` file. CDC documentation titles that file “Plasma Fasting Glucose & Insulin.” Treat as expected, not a data gap.

## SEQN overlap (unique participants)

Counts are unique `SEQN` within each file; `*_in_demo` is intersection with DEMO.

| cycle     |   demo_n |   DEMO_n |   DEMO_in_demo |   DR1IFF_n |   DR1IFF_in_demo |   BMX_n |   BMX_in_demo |   BPX_n |   BPX_in_demo |   GHB_n |   GHB_in_demo |   GLU_n |   GLU_in_demo |   MCQ_n |   MCQ_in_demo |   SMQ_n |   SMQ_in_demo |   DIQ_n |   DIQ_in_demo |   DR1TOT_n |   DR1TOT_in_demo |   DEMO_DR1IFF_BMX_GHB |   DEMO_DR1IFF_BMX_GHB_MCQ |
|:----------|---------:|---------:|---------------:|-----------:|-----------------:|--------:|--------------:|--------:|--------------:|--------:|--------------:|--------:|--------------:|--------:|--------------:|--------:|--------------:|--------:|--------------:|-----------:|-----------------:|----------------------:|--------------------------:|
| 2011-2012 |     9756 |     9756 |           9756 |       8519 |             8519 |    9338 |          9338 |    9338 |          9338 |    6549 |          6549 |    3239 |          3239 |    9364 |          9364 |    6790 |          6790 |    9364 |          9364 |       9338 |             9338 |                  5953 |                      5953 |
| 2013-2014 |    10175 |    10175 |          10175 |       8661 |             8661 |    9813 |          9813 |    9813 |          9813 |    6979 |          6979 |    3329 |          3329 |    9770 |          9770 |    7168 |          7168 |    9770 |          9770 |       9813 |             9813 |                  6343 |                      6343 |
| 2015-2016 |     9971 |     9971 |           9971 |       8505 |             8505 |    9544 |          9544 |    9544 |          9544 |    6744 |          6744 |    3191 |          3191 |    9575 |          9575 |    7001 |          7001 |    9575 |          9575 |       9544 |             9544 |                  6212 |                      6212 |
| 2017-2018 |     9254 |     9254 |           9254 |       7640 |             7640 |    8704 |          8704 |    8704 |          8704 |    6401 |          6401 |    3036 |          3036 |    8897 |          8897 |    6724 |          6724 |    8897 |          8897 |       8704 |             8704 |                  5786 |                      5786 |

## Soft drink code probe (preliminary)

Uses FNDDS code prefix `924` (carbonated soft drinks) and optional diet-keyword match on DRXFCD descriptions.

| cycle     |   dr1iff_rows |   soft_drink_code_924_rows |   soft_drink_code_924_people |   diet_soft_drink_keyword_people |   diet_keyword_among_924_people |
|:----------|--------------:|---------------------------:|-----------------------------:|---------------------------------:|--------------------------------:|
| 2011-2012 |        126503 |                       4814 |                         3425 |                                3 |                             704 |
| 2013-2014 |        131394 |                       4782 |                         3353 |                              684 |                             686 |
| 2015-2016 |        121481 |                       4129 |                         3037 |                              488 |                             489 |
| 2017-2018 |        112683 |                       3582 |                         2586 |                              370 |                             376 |

Keyword matching is **fragile** (especially 2011–2012 descriptions). Prefer official WWEIA categories below.

## WWEIA official exposure probe (Day-1 unique SEQN)

Mapped DR1IFF food codes via USDA WWEIA Food Category Excel files (`category_number` **7102** = Diet soft drinks, **7202** = Soft drinks). See `outputs/tables/wweia_beverage_probe.csv`.

| cycle     | diet soft drinks (7102) people | regular soft drinks (7202) people | both (Day-1) | people with Day-1 diet |
|:----------|-------------------------------:|----------------------------------:|-------------:|-----------------------:|
| 2011-2012 |                            694 |                              2741 |           80 |                   8519 |
| 2013-2014 |                            678 |                              2658 |           75 |                   8661 |
| 2015-2016 |                            481 |                              2496 |           29 |                   8505 |
| 2017-2018 |                            363 |                              2132 |           34 |                   7640 |

**Across 2011–2018 (raw, unweighted, all ages with Day-1 recall):** ~2,200 diet-soft-drink consumers and ~10,000 regular soft-drink consumers (not mutually exclusive across cycles; person IDs do not repeat across cycles). Adult ≥20 analytic n will be lower after exclusions.

## Cancer history variable (spot check)

`MCQ220` (ever told cancer/malignancy) present in all cycles. Example 2017–2018 adults ≥20: Yes≈588, No≈4979 (raw counts; not survey-weighted).

## Key variable presence (1 = present in cycle)

| stem   | variable   |   2011-2012 |   2013-2014 |   2015-2016 |   2017-2018 |
|:-------|:-----------|------------:|------------:|------------:|------------:|
| DEMO   | SEQN       |           1 |           1 |           1 |           1 |
| DEMO   | RIDAGEYR   |           1 |           1 |           1 |           1 |
| DEMO   | RIAGENDR   |           1 |           1 |           1 |           1 |
| DEMO   | RIDRETH3   |           1 |           1 |           1 |           1 |
| DEMO   | DMDEDUC2   |           1 |           1 |           1 |           1 |
| DEMO   | INDFMPIR   |           1 |           1 |           1 |           1 |
| DEMO   | SDMVPSU    |           1 |           1 |           1 |           1 |
| DEMO   | SDMVSTRA   |           1 |           1 |           1 |           1 |
| DEMO   | WTMEC2YR   |           1 |           1 |           1 |           1 |
| DEMO   | WTINT2YR   |           1 |           1 |           1 |           1 |
| BMX    | SEQN       |           1 |           1 |           1 |           1 |
| BMX    | BMXBMI     |           1 |           1 |           1 |           1 |
| BMX    | BMXWAIST   |           1 |           1 |           1 |           1 |
| BPX    | SEQN       |           1 |           1 |           1 |           1 |
| BPX    | BPXSY1     |           1 |           1 |           1 |           1 |
| BPX    | BPXDI1     |           1 |           1 |           1 |           1 |
| GHB    | SEQN       |           1 |           1 |           1 |           1 |
| GHB    | LBXGH      |           1 |           1 |           1 |           1 |
| GLU    | SEQN       |           1 |           1 |           1 |           1 |
| GLU    | LBXGLU     |           1 |           1 |           1 |           1 |
| GLU    | WTSAF2YR   |           1 |           1 |           1 |           1 |
| INS    | SEQN       |           0 |           1 |           1 |           1 |
| INS    | LBXIN      |           0 |           1 |           1 |           1 |
| HDL    | SEQN       |           1 |           1 |           1 |           1 |
| HDL    | LBDHDD     |           1 |           1 |           1 |           1 |
| TCHOL  | SEQN       |           1 |           1 |           1 |           1 |
| TCHOL  | LBXTC      |           1 |           1 |           1 |           1 |
| TRIGLY | SEQN       |           1 |           1 |           1 |           1 |
| TRIGLY | LBXTR      |           1 |           1 |           1 |           1 |
| TRIGLY | LBDLDL     |           1 |           1 |           1 |           1 |
| DR1IFF | SEQN       |           1 |           1 |           1 |           1 |
| DR1IFF | DR1IFDCD   |           1 |           1 |           1 |           1 |
| DR1IFF | DR1IGRMS   |           1 |           1 |           1 |           1 |
| DR1TOT | SEQN       |           1 |           1 |           1 |           1 |
| DR1TOT | DR1TKCAL   |           1 |           1 |           1 |           1 |
| MCQ    | SEQN       |           1 |           1 |           1 |           1 |
| MCQ    | MCQ220     |           1 |           1 |           1 |           1 |
| SMQ    | SEQN       |           1 |           1 |           1 |           1 |
| SMQ    | SMQ020     |           1 |           1 |           1 |           1 |
| DIQ    | SEQN       |           1 |           1 |           1 |           1 |
| DIQ    | DIQ010     |           1 |           1 |           1 |           1 |
| PAQ    | SEQN       |           1 |           1 |           1 |           1 |

## Mortality files (public LMF)

| filename                              |   bytes | exists   | note                                   |
|:--------------------------------------|--------:|:---------|:---------------------------------------|
| NHANES_2011_2012_MORT_2019_PUBLIC.dat |  475574 | True     | Fixed-width LMF; SEQN join in Phase 1B |
| NHANES_2013_2014_MORT_2019_PUBLIC.dat |  494268 | True     | Fixed-width LMF; SEQN join in Phase 1B |
| NHANES_2015_2016_MORT_2019_PUBLIC.dat |  484288 | True     | Fixed-width LMF; SEQN join in Phase 1B |
| NHANES_2017_2018_MORT_2019_PUBLIC.dat |  449623 | True     | Fixed-width LMF; SEQN join in Phase 1B |

## WWEIA category files

| relative_path                                                           |   bytes | suffix   |
|:------------------------------------------------------------------------|--------:|:---------|
| data\raw\wweia_categories\2011-2012\WWEIA1112_foodcat_FNDDS.xlsx        |  382084 | .xlsx    |
| data\raw\wweia_categories\2013-2014\WWEIA1314_foodcat_FNDDS.xlsx        |  419313 | .xlsx    |
| data\raw\wweia_categories\2015-2016\WWEIA1516_foodcat_FNDDS.xlsx        |  363726 | .xlsx    |
| data\raw\wweia_categories\2017-2018\WWEIA1718_foodcat_FNDDS.xlsx        |  307221 | .xlsx    |
| data\raw\wweia_categories\2017-2018\wweia_food_categories_2017-2018.pdf |  242507 | .pdf     |

## Tier B/C candidates (not downloaded in Phase 1A core)

| Tier | Source | Recommendation |
|------|--------|----------------|
| B | Open Food Facts | **Yes** for product sweetener context (cancer myth module) |
| B | Longer NHANES for LMF power | Decide after cancer-death counts in 1B |
| C | BRFSS SSB | Optional SSB geography only |
| C | FDA CAERS | Optional pedagogy only |

## Next steps (awaiting feedback)

1. Confirm analytic population (proposed: non-pregnant adults ≥20 with Day-1 diet + MEC).
2. Confirm exposure: WWEIA 7102/7202 primary when maps available; keyword sensitivity.
3. Approve Phase 1B cleaning pipeline.
4. Choose Tier B/C add-ons.

## Assumptions (inventory only)

- No exclusions applied yet beyond file availability.
- No survey weights applied in inventory counts (raw n).
- Soft-drink probe is approximate until WWEIA maps are validated.
