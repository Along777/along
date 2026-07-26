# Diet Soda Myth Lab — Verdicts

Independent public-data stress tests using NHANES 2011–2018 + WWEIA + public mortality.

Analytic n ≈ 19,384. Models: weighted least squares / GLM (see `model_cardio_ladder.csv`).

> **Read before citing p-values:** Continuous models use normalized multi-cycle MEC weights but **not** full NCHS design-based SEs (PSU/strata). Prefer **coefficients and CIs as approximate**. Binary/mortality GLMs use normalized weights (never raw MEC as frequency weights). Triglyceride models are on **log(TG)** scale.

**Verdict ladder:** BUSTED · NUANCED · SUPPORTED (assoc.) · UNTESTABLE HERE

**Reddit alignment:** Full claim inventory in [`docs/reddit_myth_map.md`](reddit_myth_map.md) (R1–R18). Highest-frequency public myths: cancer/WHO (R5), weight (R1), diabetes/insulin (R2–R3), gut microbiome (R4), dose (R10/R13).

---

## M1: Diet soda makes you fat / causes obesity

**Verdict:** NUANCED (assoc. remains; causation not shown)

- **Viral claim vs our test:** Media: diet soda → weight gain. Test: WLS BMI ASB-only vs Neither crude→adjusted→no-diabetes.
- **Crude:** β=2.533 (95% CI 2.221, 2.846), p=1.89e-56, n=19189
- **Adjusted:** β=2.504 (95% CI 2.179, 2.829), p=2.74e-51, n=17446
- **Sensitivity:** β=2.262 (95% CI 1.903, 2.622), p=7.86e-35, n=15018 | sub: β=1.888 (95% CI 1.478, 2.298), p=2.46e-19, n=6951
- **ML:** ΔR² BMI with ASB features: 0.007465939532267263
- **Takeaway:** ASB-only drinkers have higher BMI even after demographics/lifestyle and after excluding known diabetes (~+2 BMI units). That is a real cross-sectional association—not proof diet soda causes weight gain. Heavier people also select into diet soda (see M5/M8).
- **Limits:** Cross-sectional 24h diet; residual confounding; SEs approximate (not full survey design).

---

## M2: Diet soda wrecks blood sugar / causes diabetes

**Verdict:** NUANCED

- **Viral claim vs our test:** Media: ASB → diabetes. Test: HbA1c models + exclude known diabetes.
- **Crude:** β=0.302 (95% CI 0.259, 0.344), p=2.81e-43, n=18672
- **Adjusted:** β=0.303 (95% CI 0.260, 0.345), p=4.11e-44, n=17000
- **Sensitivity:** β=0.045 (95% CI 0.018, 0.073), p=0.00103, n=14598
- **ML:** Diabetes rank in ASB classifier top features: see feature_importance CSV
- **Takeaway:** Any crude HbA1c gap must be stress-tested by excluding people already diagnosed with diabetes—who disproportionately drink diet soda.
- **Limits:** No incident diabetes follow-up in core design.

---

## M3: Diet soda is as bad as regular soda for BP/heart markers

**Verdict:** NUANCED

- **Viral claim vs our test:** Fair-fight comparison ASB vs SSB on SBP/HDL/log-TG.
- **Crude:** SBP β=0.127 (95% CI -0.650, 0.903), p=0.749, n=18927 | HDL β=-1.615 (95% CI -2.355, -0.875), p=1.91e-05, n=18428
- **Adjusted:** SBP β=-0.800 (95% CI -1.524, -0.075), p=0.0305, n=17233 | log-TG β=0.093 (95% CI 0.054, 0.132), p=2.92e-06, n=8010 (log mg/dL, not raw TG)
- **Sensitivity:** S7 ASB vs SSB SBP: β=-1.271 (95% CI -2.119, -0.424), p=0.00329, n=6837
- **ML:** n/a
- **Takeaway:** SSB and ASB are not interchangeable profiles; compare both against Neither and each other with the same covariates.
- **Limits:** Single BP exam; meds partially unmodeled; TG models use log transform.

---

## M4: Diet soda gives you cancer

**Verdict:** BUSTED as a slogan; UNTESTABLE for low-HR long-latency causation

- **Viral claim vs our test:** Reddit/X: “WHO says aspartame causes cancer” / “one can = cancer” / NutriNet headlines. Our pack: IARC 2B vs JECFA ADI (C1–C2), age-stratified ever-cancer (C3), Cox cancer death (C4–C5).
- **Crude:** Ever-cancer by group (unweighted): {'ASB-only': 0.146, 'Both': 0.095, 'Neither': 0.105, 'SSB-only': 0.069} — confounded by age. Age-stratified rates in cancer_ever_by_age_bev.csv + figure C3
- **Adjusted:** cancer_ever logit (normalized weights): β=0.148 (95% CI 0.005, 0.290), p=0.0419, n=17612 | Prefer age-band models in cancer_ever_model_specs.csv
- **Sensitivity:** LMF deaths: all-cause=1198, cancer=285. Cox: cancer_death Cox HR=0.77 (95% CI 0.51–1.15), p=0.199, events=285; allcause_death Cox HR=0.98 (95% CI 0.81–1.17), p=0.793, events=1190. Cancer deaths=285; approx MDES HR>~1.3933562032436633 (illustrative power).
- **ML:** n/a — see docs/cancer_module_report.md and cancer_evidence_brief.md
- **Takeaway:** Slogan “diet soda gives you cancer” is not supported: (1) IARC 2B is hazard with limited evidence, JECFA kept ADI many cans/day order-of-magnitude; (2) crude ever-cancer % is inflated by who drinks diet soda (older/sicker)—age stratification shrinks the scare but does not erase every gap; residual confounding remains; (3) cancer-death Cox HR is non-significant with wide CIs and limited events (underpowered for small risks). We cannot prove lifelong safety or site-specific HCC risk from NHANES.
- **Limits:** 24h diet; self-report ever-cancer; public LMF; no incidence registry; Cox unweighted primary.

---

## M5: Diet soda drinkers are just like regular soda drinkers

**Verdict:** BUSTED

- **Viral claim vs our test:** Assume same people. Test: SMD love plot + classifier.
- **Crude:** Group n: {'Neither': 11558, 'SSB-only': 5934, 'ASB-only': 1744, 'Both': 148}
- **Adjusted:** Top |SMD| ASB vs Neither: [{'covariate': 'Diabetes (SR)', 'smd': 0.4222389307503126}, {'covariate': 'BMI', 'smd': 0.3625740572445499}, {'covariate': 'PIR', 'smd': 0.201458193673535}, {'covariate': 'Age', 'smd': 0.1865394602497514}, {'covariate': 'Smoking code', 'smd': 0.0599369241674233}]
- **Sensitivity:** ASB classifier AUC weighted=0.6631533150818105, unweighted=0.6801684335167986
- **ML:** Top features: [{'feature': 'diabetes_sr', 'importance_mean': 0.04439347073931868}, {'feature': 'pir', 'importance_mean': 0.03310383952058307}, {'feature': 'waist', 'importance_mean': 0.02552714912305072}, {'feature': 'bmi', 'importance_mean': 0.023563621487797476}, {'feature': 'age', 'importance_mean': 0.018519549537553692}]
- **Takeaway:** Diet-soda consumers differ systematically (age, sex, diabetes, BMI, SES). Treating them as random is wrong.
- **Limits:** Profile ≠ moral judgment; confounding structure.

---

## M6: Any amount is poison / one can is harmless always

**Verdict:** NUANCED

- **Viral claim vs our test:** Dose models: BMI ~ diet-soft servings (continuous).
- **Crude:** BMI ~ asb_serv (minimal age/sex in S6 still has full lifestyle set): β=0.966 (95% CI 0.859, 1.073), p=1.23e-69, n=17446 | binary ASB-only S0: β=2.533 (95% CI 2.221, 2.846), p=1.89e-56, n=19189
- **Adjusted:** BMI ~ asb_serv with lifestyle covariates is S6_dose above; binary ASB S3: β=2.504 (95% CI 2.179, 2.829), p=2.74e-51, n=17446 | HbA1c dose (secondary): β=0.094 (95% CI 0.080, 0.108), p=1.55e-39, n=17000
- **Sensitivity:** Heavy zero-inflation: most adults drink 0 diet soft drinks on recall day. Do not mix BMI and HbA1c in one crude/adjusted pair.
- **ML:** n/a
- **Takeaway:** Binary any/none and continuous dose can disagree; single 24h recall misclassifies usual intake.
- **Limits:** Day-1 recall noise at low doses.

---

## M7: Results are p-hacked / only one definition works

**Verdict:** PROCESS PASS (transparency)

- **Viral claim vs our test:** Specification curve across S0–S7.
- **Crude:** See multiverse_bmi.csv and myth_m7_spec_curve_bmi.png
- **Adjusted:** BMI ASB-only specs run: 5
- **Sensitivity:** Exposure multiverse E1–E5 documented in exposure_definitions.md
- **ML:** n/a
- **Takeaway:** We publish the curve, not a single p-value.
- **Limits:** Multiverse still researcher-designed.

---

## M8: Labs prove diet soda is the driver

**Verdict:** NUANCED — predictive ASB signal is small vs health selection features

- **Viral claim vs our test:** Predict ASB use from labs/demographics; predict BMI with/without ASB.
- **Crude:** BMI R² without ASB=0.07430217294008468, with=0.08176811247235194
- **Adjusted:** ΔR²=0.007465939532267263
- **Sensitivity:** Weighted vs unweighted AUC: 0.6631533150818105 vs 0.6801684335167986
- **ML:** {'min_samples_leaf': 50, 'max_iter': 100, 'max_depth': 3, 'learning_rate': 0.05, 'l2_regularization': 0.1}
- **Takeaway:** If BMI/diabetes predict diet-soda use better than soda predicts BMI, the causal arrow in headlines is often backwards.
- **Limits:** Predictive ≠ causal.

---

## M9: Diet soda destroys the gut microbiome (Reddit high-frequency)

**Verdict:** UNTESTABLE HERE

- **Viral claim vs our test:** r/science, r/diabetes, r/HydroHomies often claim artificial sweeteners wreck the microbiome → insulin resistance. NHANES has **no stool metagenomics**.
- **Crude / adjusted / ML:** n/a — no exposure–microbiome measures in this public dataset.
- **Takeaway:** Microbiome harm is a real *research question* (some feeding studies exist). It is **not** something this Myth Lab can confirm or bust with NHANES exam/lab files. Saying nothing would look like we ignored the feed; saying we “disproved it” would be dishonest.
- **Limits:** Would need controlled feeding + microbiome sequencing, not a one-day diet recall survey.
- **See:** `docs/reddit_myth_map.md` R4.

---

## FAQ (Reddit-adjacent truths)

| Question | Short answer |
|----------|----------------|
| PKU / phenylalanine? | People with **PKU** must limit aspartame — real clinical issue, rare, not the general “cancer” myth. |
| Industry funded this? | **No.** Pipeline uses free public NHANES/CDC/USDA data and open code. |
| Mouse “cancer at low dose”? | Check **human-equivalent dose**; many headlines map to absurd can counts — see ADI chart (cancer C2). |
| Should I drink diet soda? | Not medical advice. Relative to **sugar-sweetened** soda, substitution is a different question than “is zero risk.” Water is still the boring winner. |

## Cancer evidence context (not NHANES microdata)

- **IARC (2023):** aspartame Group **2B** (possibly carcinogenic) — *limited* evidence.
- **JECFA:** ADI **0–40 mg/kg/day** reaffirmed; human cancer evidence *not convincing* at usual intakes.
- **FDA:** continues to allow aspartame under approved conditions.
- Group 2B ≠ “diet soda gives you cancer” in plain language; hazard ≠ risk at dietary doses.
- Reddit often cites **NutriNet-Santé** (incidence) or “30 cans / monkey” dose folklore — see `reddit_myth_map.md` R5/R13.

## Anti-pattern checklist

- [x] Multi-cycle pool (not one Kaggle year)
- [x] WWEIA official diet soft drink codes (not keyword-only)
- [x] Soda-type exclusivity (not “only beverage in the universe”)
- [x] Reverse-causation exclusions
- [x] No causal ML claims
- [x] Sampling weights **not** abused as GLM frequency counts (normalized only)
- [~] Full NCHS design-based SE (PSU/strata) — **not implemented**; treat p-values as approximate

## Known statistical caveats (post self-audit)

1. Continuous models: WLS with normalized multi-cycle MEC weights — good for **point estimates**, optimistic for **SEs** (no cluster design).
2. Binary/mortality: binomial GLM with **normalized** weights or unweighted; raw `WTMEC2YR` as `freq_weights` was a bug and was fixed.
3. Triglycerides: fasting lab; prefer `*_fast` rows in model table when present.
