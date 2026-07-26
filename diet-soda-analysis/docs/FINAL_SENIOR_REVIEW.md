# Final senior review — Diet Soda Myth Lab

**Reviewer role:** senior biostatistics / epi methods (portfolio demo readiness)  
**Scope:** coworker pipeline + myth narrative + figure pack + HTML readiness  
**Analytic n:** 19,384 adults (NHANES 2011–2018; WWEIA 7102 vs 7202; public LMF)  
**Date:** 2026-07-26  

---

## Executive judgment

| Dimension | Grade | One-line |
|-----------|-------|----------|
| **Data engineering** | **A−** | Multi-cycle NHANES + WWEIA + LMF; exclusive soda-type exposure; normalized weights; documented exclusions |
| **Myth design (what people talk about)** | **A** | Cancer/WHO + weight + diabetes + selection + microbiome named; Reddit map R1–R18 |
| **Biostat honesty** | **A− / B+** | Reverse-causation stress tests; underpowered-null language; SE caveat front-loaded; still no design-based SEs |
| **Cancer communication pack** | **A** | Hazard≠risk, ADI cans, age strata, KM, Cox forest — correct public-science stack |
| **Diet leads (weight/diabetes)** | **A** | +2.5 BMI sticks; HbA1c 0.30→0.05 after excl. known DM — demo-safe numbers |
| **Inference / causal claims** | **B−** | Correct *language* mostly; design still cross-sectional 24h + short mortality FU |
| **HTML readiness** | **GO** | `scripts/build_html_report.py` leads with selection → diet myths → cancer hero |

**Ship the HTML.** Do not ship “we proved safety” or design-perfect p-values.

---

## What the coworker built (and what makes sense)

### Architecture (sensible)

```
raw NHANES XPT + WWEIA + LMF
        ↓
build_analysis_dataset → analysis_ready*.parquet
        ↓
run_eda  → myth figures M1–M8, missingness, SMD
run_models → model_cardio_ladder (S0–S7, logits, mort)
run_ml → ASB classifier + BMI ΔR²
run_cancer_module → C1–C5 + Cox + power note
run_verdicts → myth_verdicts.md + CSV
build_html_report → myth_lab_report.html
```

This is the right order: **exposure truth → selection → outcome ladders → mortality/cancer pack → locked language**.

### Exposure (correct for a myth lab)

- Primary: Day-1 **WWEIA 7102** (diet soft drinks) vs **7202** (soft drinks).
- `bev_group` exclusive on **soda type only** (Neither / ASB-only / SSB-only / Both).
- Counts match live parquet: **Neither 11,558 · SSB-only 5,934 · ASB-only 1,744 · Both 148**.

### Weighting (fixed and documented)

- Multi-cycle MEC: `WTMEC2YR / 4`.
- Binary GLMs: **normalized** weights (raw MEC-as-`freq_weights` bug fixed and called out in `assumptions.md`).
- Continuous: WLS with normalized weights.
- **Still missing:** SDMVPSU × SDMVSTRA design-based variance → p-values **optimistic**. Demo language correctly demotes p-values.

---

## Myth-by-myth biostats review

### M5 / M8 — Who drinks diet soda (LEAD #1) — **BUSTED** · strongest story

| Claim | Evidence | Sound? |
|-------|----------|--------|
| ASB drinkers ≠ general sample | Diabetes ~**31% vs 14%**; BMI ~**31.5 vs 28.9**; age ~**55 vs 51**; higher PIR | **Yes** |
| Selection dominates ML | Top features: diabetes, PIR, waist, BMI, age; AUC ~0.66–0.68 | **Yes** (predictive, not causal) |
| BMI ΔR² from ASB features | ~**0.007** | **Yes** — tiny incremental prediction |

**Figures that must ship:** `myth_m5_smd_loveplot.png`, `myth_m8_asb_feature_importance.png`  
**Optional depth:** `myth_m5_bev_by_sex.png`  
**Why it leads the report:** Without this, every disease association is misread as “soda did it.”

---

### M2 — Diabetes / blood sugar (LEAD #2) — **NUANCED**

| Step | ASB-only β (HbA1c) | Interpretation |
|------|--------------------|----------------|
| S0 crude | +0.302 | Scary crude gap |
| S3 adjusted | +0.303 | Demographics don’t erase it |
| **S5 no known DM** | **+0.045** | **Mostly selection / reverse causation** |
| S6 dose | +0.094 / serving | Dose among zero-inflated day | 

**Lock language:** “mostly shrinks (0.30 → 0.05)” — **not** “collapses / disappears.” Residual +0.05 is still detectable with approximate SEs; do not overclaim zero.

**Figures:** `myth_m2_hba1c_violin.png`, `myth_m1_reverse_causation_scatter.png`  
**Gap (honest):** no incident T2D; no clamp / CGM → R2 “cephalic insulin” stays **UNTESTABLE HERE**.

---

### M1 — Weight / obesity (LEAD #2b) — **NUANCED (assoc.)**

| Step | BMI β ASB-only vs Neither |
|------|---------------------------|
| S0 | +2.53 |
| S3 lifestyle | +2.50 |
| S5 no DM | +2.26 |
| S7 ASB vs SSB | +1.89 |
| S6 dose / serving | +0.97 |

**Sound:** Association is **stubborn** after the same reverse-causation exclusion that gutted HbA1c. That is scientifically interesting and demo-honest: *not* “diet soda causes obesity,” *yes* “heavier people and higher BMI cluster with ASB use; causation not identified.”

**Figures:** `myth_m1_bmi_violin.png` (+ reverse-causation scatter shared with M2)  
**ML punchline:** ΔR² ≈ 0.007 — labs/selection >> soda features for predicting BMI.

---

### M4 — Cancer (HERO) — **BUSTED as slogan; UNTESTABLE for small long-latency risks**

This is the highest-frequency public claim (Reddit R5/R13). Coworker pack is **correctly layered**:

| Layer | Asset | Biostats judgment |
|-------|-------|-------------------|
| Hazard ≠ risk | C1 + evidence brief | **Essential** — IARC 2B ≠ JECFA ADI / FDA |
| Dose reality | C2 ADI cans (~10–16 cans/d order-of-magnitude @ 50–80 kg) | **Correct frame** for “one can = cancer” |
| Crude scare | Ever-cancer ~14.6% ASB vs ~10.5% Neither | **Confounded by age/comorbidity** |
| Age strata | C3 / C3b | Shrinks scare; **residual gaps remain** (e.g. 60+) — do **not** say “age explains all” |
| Ever-cancer logit | Crude OR-ish β 0.36 → full ~0.20 | Attenuates; still not causal (lifetime cancer vs yesterday’s diet) |
| Cancer **death** Cox | HR **0.77** (0.51–1.15), p≈0.20, events=285 | **NS**; ~**27** ASB-only cancer deaths |
| Power | MDES HR ≳1.39 (illustrative) | Directionally honest; **still optimistic** (assumes ~balanced exposure; ASB is ~9%) |

**Figures (ship all as hero pack):**  
`cancer_c1` · `cancer_c2` · `cancer_c3` · `cancer_c3b` · `cancer_c4` · `cancer_c5`  
Secondary: `myth_m4_cancer_crude.png` (optional; C3b is stronger for demos)

**Never say:** proved safety / “no cancer risk” / WHO banned soda.  
**Always say:** slogan not supported; underpowered null; site-specific incidence (e.g. HCC) not in NHANES.

**Small n-discrepancies (benign):** Cox complete-case n≈19,312 vs mort-eligible ~19,327; all-cause deaths 1,198 (sum) vs 1,190 (Cox complete case). Prefer Cox table numbers when reporting model results; prefer sum when saying “events in linked file.”

---

### M3 — Heart / BP / lipids — **NUANCED**

- SBP crude ~null; after covariates ASB-only slightly **lower** SBP vs Neither (and vs SSB in S7) — interesting, **not** “heart protective proven.”
- HDL lower in ASB crude/adjusted; shrinks toward null with BMI (S4) — composition again.
- **log(TG)** only — document every time (already in ladder notes).

**Figure:** `myth_m3_sbp_violin.png`  
**Limit:** single exam BP; meds partially unmodeled.

---

### M6 — Dose / “any amount poison” — **NUANCED**

- Heavy zero-inflation (~90% zero on Day-1).
- Continuous β mixes any-vs-none with dose among drinkers.
- Pair with **C2 ADI** for Reddit poison claims.

**Figure:** `myth_m6_asb_dose_hist.png`

---

### M7 — Multiverse / p-hack defense — **PROCESS PASS**

- Spec curve + cycle stability for BMI exist.
- Researcher-designed multiverse still (honest limits text).

**Figures:** `myth_m7_spec_curve_bmi.png`, `myth_m7_cycle_stability_bmi.png`

---

### M9 — Microbiome — **UNTESTABLE HERE** (must keep)

Highest-frequency claim you **cannot** test. Naming it as untestable is **stronger science** than fake-nulling it. HTML + verdicts already do this.

---

## Chart / figure audit (myth alignment)

| Figure | Myth role | Ship in HTML? | Notes |
|--------|-----------|---------------|-------|
| myth_m5_smd_loveplot | Lead: selection | **Yes** | Core |
| myth_m8_asb_feature_importance | Lead: reverse arrow | **Yes** | Core |
| myth_m2_hba1c_violin | Diet: diabetes | **Yes** | |
| myth_m1_reverse_causation_scatter | Diet: RC visual | **Yes** | |
| myth_m1_bmi_violin | Diet: weight | **Yes** | |
| cancer_c1 … c5 | Cancer hero | **Yes all** | |
| myth_m3_sbp_violin | Heart | Yes | |
| myth_m6_asb_dose_hist | Dose | Yes | |
| myth_m7_spec + cycle | Process | Yes | |
| myth_missingness | Data QC | Yes | |
| myth_m4_cancer_crude | Cancer crude | Optional | C3b preferred |
| myth_m5_bev_by_sex | Selection detail | Optional | |
| myth_corr_heatmap | EDA | Optional / appendix | |

**Narrative order that matches human talk (and HTML):**  
1. Who drinks (M5/M8) → 2. Diabetes + weight (M2/M1) → 3. Cancer pack (M4) → 4. Heart/dose/robustness/microbiome → 5. Reddit map.

---

## Locked numbers (challenge table)

| Fact | Value | Source |
|------|-------|--------|
| Analytic n | 19,384 | analysis_ready |
| ASB-only n | 1,744 | bev_group |
| Diabetes SR ASB / Neither | ~31% / ~14% | means |
| Mean BMI ASB / Neither | ~31.5 / ~28.9 | means |
| Mean age ASB / Neither | ~55 / ~51 | means |
| BMI β S3 / S5 | +2.50 / +2.26 | model ladder |
| HbA1c β S0 / S5 | +0.30 / +0.05 | model ladder |
| Ever-cancer ASB / Neither | ~14.6% / ~10.5% | means |
| Cancer deaths total / ASB-only | 285 / **27** | LMF + group |
| Cox cancer-death HR | 0.77 (0.51–1.15), p≈0.20 | cancer_cox_results |
| BMI ΔR² (ASB features) | ~0.007 | ml_tuning_results |
| ASB classifier AUC (w / uw) | ~0.66 / ~0.68 | ml |

---

## Language hygiene (do / don’t)

| Avoid | Prefer |
|-------|--------|
| HbA1c “collapses / disappears” | **Mostly shrinks** 0.30 → 0.05 after excl. known DM |
| “Age explains the cancer rates” | Age/comorbidity **inflate** crude rates; residual gaps can remain |
| “Mortality NS after correct weighting” | Cancer-death Cox **NS**; ~27 events in ASB-only |
| “We disproved cancer forever” | Slogan not supported; small long-latency risks untestable |
| “p = 1e−56 is design-correct” | Point estimates useful; p-values **optimistic** |
| “MDES >1.39 is definitive” | Only large effects detectable; rare exposure + few ASB deaths |
| “Diet soda causes obesity” | Associated with higher BMI; causation not shown |

*(One residual soft wording was fixed in `reddit_myth_map.md` R3: “collapses” → “mostly shrinks.”)*

---

## Remaining gaps (post-HTML backlog — not blockers)

1. **Design-based SEs** (PSU/strata) — would re-rank p-values; coefficients likely similar.  
2. **Cox / power footnotes** — report events-by-arm on C5; MDES with **unbalanced** exposure (~9% ASB).  
3. **Ever-cancer residual association** — optional age-band models already tabulated; keep residual language.  
4. **Meds / BP treatment** — improves M3 credibility if ever extended.  
5. **Optional figures** in appendix only (corr heatmap, crude M4, sex split).  
6. **Incident outcomes** — out of scope for this public demo (would need external cohorts).

---

## HTML readiness checklist

- [x] Facts load from parquet + ladder + Cox (not hard-coded fairy tales)  
- [x] Biostats banner (weights / SE / log-TG / 24h diet)  
- [x] Lead: who drinks (M5/M8 figures)  
- [x] Lead: diet myths M2 then M1 with β lines from tables  
- [x] Hero: cancer C1–C5 + HR + ~27 ASB cancer deaths  
- [x] Reddit map table  
- [x] Microbiome untestable card  
- [x] Reproduce block  
- [ ] Run `python scripts/build_html_report.py` → `outputs/myth_lab_report.html`  
- [ ] Open HTML; confirm images render (base64)  

**Verdict:** **GO for HTML.** Narrative, numbers, and figure pack are aligned for a coworker-facing myth lab with cancer + diet as public-lead stories.

---

## One-paragraph coworker handoff

Open with **who drinks diet soda** (diabetes, BMI, age, income — love plot + ML). Then **diabetes myth**: HbA1c gap mostly shrinks when known diabetes is excluded. Then **weight myth**: BMI stays ~+2.5 — association without causation. Then **cancer hero**: IARC 2B is hazard not risk; ADI is many cans/day order-of-magnitude; crude cancer % is inflated by who drinks it; cancer-death Cox HR ~0.77 NS with only ~27 ASB cancer deaths — underpowered null, not lifelong safety. Name **microbiome** as untestable. Caveat approximate SEs and day-1 diet every time someone asks about p-values.
