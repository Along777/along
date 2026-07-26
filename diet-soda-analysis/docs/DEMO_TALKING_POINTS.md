# Coworker demo — 5 minutes (data-checked)

Cross-checked against analytic n=19,384 and model tables. Prefer these phrases over older “collapses / correct weighting” shortcuts.

## Open these

1. **`index.html`** — public article: 30-second take (“What I learned”) → Cancer → Diet → Who drinks → Models · first person · Grok 4.5 note  
2. `docs/myth_verdicts.md` — full myth ladder  
3. `outputs/figures/myth_m5_smd_loveplot.png` + `myth_m8_asb_feature_importance.png`  
4. `outputs/figures/myth_m2_hba1c_violin.png` + `myth_m1_bmi_violin.png`  
5. `docs/assumptions.md` — if they ask about weights/p-values  
6. Cancer deep-dive: `docs/cancer_module_report.md` + `cancer_c1`…`c5` figures  
7. Rebuild article: `python scripts/build_html_report.py`  

## Script (safe)

1. **Data:** NHANES 2011–2018, WWEIA diet soft drinks (**7102**) vs regular (**7202**), **n = 19,384** adults (ASB-only 1,744 / SSB-only 5,934 / Both 148 / Neither 11,558).  
2. **Who drinks it (M5/M8):** Not a random sample — vs non-drinkers roughly **~31% vs ~14% diabetes**, BMI **~31.5 vs ~28.9**, **slightly older** (~55 vs ~51), higher income (PIR). Top ML predictors of diet-soda use: diabetes, income, waist/BMI, age.  
3. **Diabetes myth (M2):** Crude HbA1c gap for ASB-only ≈ **+0.30**; after excluding known diabetes ≈ **+0.05** — **mostly shrinks** (selection/reverse causation), not magic zero. Still not proof ASB causes diabetes.  
4. **Weight myth (M1):** BMI association **stays ~+2.3 to +2.5** after covariates / after dropping known diabetes — **real association, not proven causation**. ASB features barely improve BMI prediction (ΔR² ≈ 0.007).  
5. **Cancer slogan (M4):**  
   - IARC **2B hazard** ≠ JECFA **ADI / usual-intake risk** (figures C1–C2).  
   - Crude ever-cancer % higher in ASB drinkers is **inflated by who drinks it**; age stratification **helps** but **does not erase every gap** (e.g. still higher in 60+).  
   - **Cancer-death Cox** HR ≈ **0.77** (0.51–1.15), p≈0.20 — **not significant**, with only **~27 cancer deaths in ASB-only** → **underpowered null**, not proof of safety.  
6. **Microbiome (M9):** Common Reddit claim — **untestable in NHANES** (no stool data).  
7. **Caveats (say out loud):** Day-1 diet ≠ lifetime intake; **SEs approximate** (weights yes, full PSU/strata design no); TG models are **log-TG**; not medical advice.

## Do not claim

| Avoid | Prefer |
|-------|--------|
| “HbA1c collapses / disappears” | “**Mostly** shrinks (0.30 → 0.05) when we exclude known diabetes” |
| “Mortality NS after correct weighting” | “**Cancer-death Cox not significant**; ~**27** events in the diet-soda group” |
| “Age explains the cancer rates” | “Age/comorbidity **inflate** crude rates; residual gaps can remain” |
| “Diet soda causes obesity” | “Associated with higher BMI; causation not shown” |
| “We disproved cancer forever” | “Simple slogan not supported; small long-latency risks untestable here” |
| “p=1e−56 is design-correct” | “Point estimates useful; p-values optimistic” |
| “MDES HR > 1.39 is definitive” | “Only large effects detectable; unbalanced exposure + few ASB deaths” |

## Cancer deep-dive (if they care)

1. **Hazard ≠ risk** (C1): IARC 2B vs JECFA ADI.  
2. **Cans to ADI** (C2): many cans/day order-of-magnitude — not “one can = cancer.”  
3. **Age stratification** (C3): crude scare shrinks; residual gaps may remain.  
4. **Cox cancer death** (C4–C5): follow-up months; HR NS; **~27 ASB cancer deaths**.  
5. Never: proved safety. Always: slogan not supported + underpowered for small risks.

## Reddit myths (what they will ask)

Full map: `docs/reddit_myth_map.md`.

| They say | You say |
|----------|---------|
| WHO banned diet soda / causes cancer | IARC 2B ≠ JECFA ADI; open C1–C2 |
| Diet soda causes diabetes | M2: HbA1c gap **mostly shrinks** after excluding known diabetes |
| Makes you fat | M1: association ~+2 BMI; causation no; selection M5/M8 |
| Destroys microbiome | M9: **UNTESTABLE** in NHANES — no stool data |
| Mouse study / any can is poison | Dose translation + ADI chart |
| Industry covered it up | Public NHANES + open code; no industry funding |

## Sanity numbers (if challenged)

| Fact | Value |
|------|--------|
| Analytic n | 19,384 |
| ASB-only n | 1,744 |
| BMI ASB-only vs Neither (adjusted) | ~+2.5 kg/m² |
| HbA1c crude → no known DM | ~+0.30 → ~+0.05 |
| Ever-cancer crude ASB / Neither | ~14.6% / ~10.5% |
| Cancer deaths total / in ASB-only | 285 / ~27 |
| Cox cancer-death HR (ASB-only) | 0.77 (0.51–1.15), p≈0.20 |
| BMI ΔR² adding ASB features | ~0.007 |
