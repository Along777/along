# Reddit diet-soda myth map

Public discourse scan (r/science, r/nutrition, r/diabetes, r/askscience, r/skeptic, r/loseit, r/keto, r/HydroHomies, r/HubermanLab, r/worldnews, etc.).  
**Reddit is not evidence** — it tells us which claims a coworker or reader will bring up.

| ID | Claim people make | Our verdict style | Where in Myth Lab | What would be needed to test properly |
|----|-------------------|-------------------|-------------------|----------------------------------------|
| R1 | Diet soda makes you fat / blocks weight loss | **NUANCED** — higher BMI in drinkers; reverse selection; not proven causation | M1, figures BMI violin / SMD / reverse-causation scatter | RCTs of SSB→ASB substitution; longitudinal weight change |
| R2 | Sweetness alone spikes insulin (“cephalic insulin”) | **UNTESTABLE HERE** for acute response; fasting insulin only in subsample | M2 note | Meal-test / CGM experiments |
| R3 | Causes / worsens type 2 diabetes | **NUANCED** — HbA1c gap **mostly shrinks** (~0.30 → ~0.05) after excluding known diabetes | M2 | Incident T2D cohorts with repeated diet |
| R4 | Destroys gut microbiome → metabolic harm | **UNTESTABLE HERE** (no stool/microbiome in NHANES) | M9 below | Metagenomic trials + controlled feeding |
| R5 | Causes cancer / “WHO said so” / NutriNet | **BUSTED as slogan**; **UNTESTABLE** for small long-latency risks | M4, cancer C1–C5 | Incidence cohorts; site-specific (e.g. HCC) |
| R6 | Addiction / cravings / “worse than sugar for brain” | Out of scope (behavior) | FAQ | Validated craving scales, RCTs |
| R7 | Ruins teeth (acid) even if zero sugar | Optional / not primary | M10 if oral data added | Dental exams + acid exposure models |
| R8 | Weakens bones / osteoporosis | Not tested | — | DXA + phosphorus intake (optional) |
| R9 | Heart / BP “as bad as regular soda” | **NUANCED** fair-fight ASB vs SSB | M3 | CVD incidence cohorts |
| R10 | One can is poison vs any amount fine | **NUANCED** + ADI dose chart | M6, cancer C2 | Usual-intake dose–response with low error |
| R11 | Only real risk is PKU / phenylalanine | **TRUE for PKU** (rare genetic); not a general cancer story | FAQ | Clinical PKU guidance |
| R12 | Industry hid the harm | Process: we use **public NHANES**, transparent code | README / assumptions | Independent funding disclosure (ours: none) |
| R13 | Mouse/monkey absurd doses prove human cancer | **BUSTED as human dose translation** | C2 + cancer brief | Dose scaling literacy |
| R14 | “I switched to diet soda and labs improved” | Anecdote ≠ causal; fits substitution story | M5/M7 narrative | Within-person pre/post with controls |
| R15 | Brain fog / MS / seizures from aspartame | Folklore; not supported as established | Do not chase | Neuroepidemiology if ever |
| R16 | Fertility / pregnancy harm | Pregnant excluded from sample | Sample design | Specialized pregnancy cohorts |
| R17 | Kidney damage | Not tested | Optional later | eGFR/ACR + longitudinal |
| R18 | Physics (floats / freezes faster) | Not a health claim | Fun aside only | Density/freezing point |

## Highest-frequency clusters (demo order)

1. **Cancer / WHO / aspartame** (R5, R13) → `docs/cancer_module_report.md`, figures `cancer_c1`–`cancer_c5`  
2. **Weight + diabetes + insulin** (R1–R3) → M1/M2  
3. **Gut microbiome** (R4) → M9 UNTESTABLE (name it so we don’t look like we ignored it)  
4. **Dose / one can** (R10, R13) → C2 ADI cans + M6  
5. **Addiction / cravings** (R6) → one sentence: out of scope  

## One-liners (Reddit-fluent)

- “If your feed said WHO banned diet soda—that’s not what IARC + JECFA jointly said.”  
- “If your feed said diet soda causes diabetes, ask who already at risk switched to diet soda—that’s our M2 pattern.”  
- “Microbiome scares are real *research questions*; this public survey cannot measure your gut bugs.”  
- “Mouse-dose headlines often aren’t one-can human doses—see the ADI chart.”  

## Coverage scorecard

| We test with NHANES data | We only communicate / cite | We skip on purpose |
|--------------------------|----------------------------|--------------------|
| R1 weight, R3 diabetes markers, R5 cancer death/history, R9 cardiometabolic, R10 dose | R4 microbiome lit, R11 PKU, R12 funding, R13 dose translation, R5 NutriNet/UKB | R6 addiction, R15 neuro folklore, R18 physics |
