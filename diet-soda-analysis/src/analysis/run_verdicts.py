"""Compile myth verdicts from model/EDA/ML artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.config import get_paths, load_config


def _coef_line(res: pd.DataFrame, outcome: str, step: str, term: str = "asb_only") -> str:
    r = res[(res["outcome"] == outcome) & (res["step"] == step) & (res["term"] == term)]
    if r.empty:
        return "n/a"
    row = r.iloc[0]
    return f"β={row['coef']:.3f} (95% CI {row['ci_low']:.3f}, {row['ci_high']:.3f}), p={row['pval']:.3g}, n={int(row['n'])}"


def run_verdicts(cfg=None) -> None:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    res = pd.read_csv(paths["tables"] / "model_cardio_ladder.csv")
    ml = {}
    mlp = paths["tables"] / "ml_tuning_results.json"
    if mlp.exists():
        ml = json.loads(mlp.read_text(encoding="utf-8"))
    exp = pd.read_csv(paths["tables"] / "analytic_bev_group_counts.csv")
    smd = pd.read_csv(paths["tables"] / "smd_asb_vs_neither.csv") if (paths["tables"] / "smd_asb_vs_neither.csv").exists() else None

    # Event counts mortality
    mort = pd.read_parquet(paths["processed"] / "analysis_ready_mortality.parquet")
    n_dead = int(mort["allcause_death"].fillna(0).sum()) if "allcause_death" in mort else 0
    n_cdead = int(mort["cancer_death"].fillna(0).sum()) if "cancer_death" in mort else 0

    # Cancer crude
    ar = pd.read_parquet(paths["processed"] / "analysis_ready.parquet")
    cancer_by = ar.groupby("bev_group")["cancer_ever"].mean()

    verdicts = []

    # M1
    m1 = {
        "myth_id": "M1",
        "claim": "Diet soda makes you fat / causes obesity",
        "viral_claim_vs_test": "Media: diet soda → weight gain. Test: WLS BMI ASB-only vs Neither crude→adjusted→no-diabetes.",
        "crude": _coef_line(res, "bmi", "S0"),
        "adjusted": _coef_line(res, "bmi", "S3"),
        "sensitivity": _coef_line(res, "bmi", "S5_no_dm") + " | sub: " + _coef_line(res, "bmi", "S7_sub", "asb_vs_ssb"),
        "ml": f"ΔR² BMI with ASB features: {ml.get('bmi_delta_r2', 'n/a')}",
        "verdict": "NUANCED (assoc. remains; causation not shown)",
        "takeaway": "ASB-only drinkers have higher BMI even after demographics/lifestyle and after excluding known diabetes (~+2 BMI units). That is a real cross-sectional association—not proof diet soda causes weight gain. Heavier people also select into diet soda (see M5/M8).",
        "limits": "Cross-sectional 24h diet; residual confounding; SEs approximate (not full survey design).",
    }
    verdicts.append(m1)

    # M2
    m2 = {
        "myth_id": "M2",
        "claim": "Diet soda wrecks blood sugar / causes diabetes",
        "viral_claim_vs_test": "Media: ASB → diabetes. Test: HbA1c models + exclude known diabetes.",
        "crude": _coef_line(res, "hba1c", "S0"),
        "adjusted": _coef_line(res, "hba1c", "S3"),
        "sensitivity": _coef_line(res, "hba1c", "S5_no_dm"),
        "ml": f"Diabetes rank in ASB classifier top features: see feature_importance CSV",
        "verdict": "NUANCED",
        "takeaway": "Any crude HbA1c gap must be stress-tested by excluding people already diagnosed with diabetes—who disproportionately drink diet soda.",
        "limits": "No incident diabetes follow-up in core design.",
    }
    verdicts.append(m2)

    # M3 — TG is log-scale
    tg_label = "log_tg" if (res["outcome"] == "log_tg").any() else "tg"
    m3 = {
        "myth_id": "M3",
        "claim": "Diet soda is as bad as regular soda for BP/heart markers",
        "viral_claim_vs_test": "Fair-fight comparison ASB vs SSB on SBP/HDL/log-TG.",
        "crude": "SBP " + _coef_line(res, "sbp_mean", "S0") + " | HDL " + _coef_line(res, "hdl", "S0"),
        "adjusted": "SBP "
        + _coef_line(res, "sbp_mean", "S3")
        + " | log-TG "
        + _coef_line(res, tg_label, "S3")
        + " (log mg/dL, not raw TG)",
        "sensitivity": "S7 ASB vs SSB SBP: " + _coef_line(res, "sbp_mean", "S7_sub", "asb_vs_ssb"),
        "ml": "n/a",
        "verdict": "NUANCED",
        "takeaway": "SSB and ASB are not interchangeable profiles; compare both against Neither and each other with the same covariates.",
        "limits": "Single BP exam; meds partially unmodeled; TG models use log transform.",
    }
    verdicts.append(m3)

    # M4 — full talking-point pack (see run_cancer_module + cancer_module_report.md)
    cox_path = paths["tables"] / "cancer_cox_results.csv"
    age_path = paths["tables"] / "cancer_ever_by_age_bev.csv"
    pwr_path = paths["tables"] / "cancer_power_mdes.json"
    cox_txt = "run python -m src.analysis.run_cancer_module"
    if cox_path.exists():
        cox = pd.read_csv(cox_path)
        bits = []
        for _, row in cox[cox.get("term", pd.Series()) == "asb_only"].iterrows() if "term" in cox.columns else []:
            bits.append(
                f"{row['outcome']} HR={row.get('hr', float('nan')):.2f} "
                f"({row.get('hr_lo', float('nan')):.2f}–{row.get('hr_hi', float('nan')):.2f}), p={row.get('pval', float('nan')):.3g}"
            )
        # simpler loop
        bits = []
        for _, row in cox.iterrows():
            if row.get("term") == "asb_only" and "hr" in row:
                bits.append(
                    f"{row['outcome']} Cox HR={float(row['hr']):.2f} "
                    f"(95% CI {float(row['hr_lo']):.2f}–{float(row['hr_hi']):.2f}), p={float(row['pval']):.3g}, events={row.get('events')}"
                )
        cox_txt = "; ".join(bits) if bits else cox_txt
    pwr_txt = ""
    if pwr_path.exists():
        pwr = json.loads(pwr_path.read_text(encoding="utf-8"))
        pw = pwr.get("power", pwr)
        pwr_txt = (
            f"Cancer deaths={pw.get('cancer_events', pwr.get('cancer_events'))}; "
            f"approx MDES HR>~{pw.get('approx_mdes_hr_above_1', pwr.get('power', {}).get('approx_mdes_hr_above_1'))} "
            f"(illustrative power)."
        )
    age_note = "See cancer_c3_ever_cancer_by_age.png"
    if age_path.exists():
        age_note = "Age-stratified rates in cancer_ever_by_age_bev.csv + figure C3"

    m4 = {
        "myth_id": "M4",
        "claim": "Diet soda gives you cancer",
        "viral_claim_vs_test": (
            "Reddit/X: “WHO says aspartame causes cancer” / “one can = cancer” / NutriNet headlines. "
            "Our pack: IARC 2B vs JECFA ADI (C1–C2), age-stratified ever-cancer (C3), Cox cancer death (C4–C5)."
        ),
        "crude": f"Ever-cancer by group (unweighted): {cancer_by.round(3).to_dict()} — confounded by age. {age_note}",
        "adjusted": "cancer_ever logit (normalized weights): "
        + (_coef_line(res, "cancer_ever", "S3_logit") if (res["step"] == "S3_logit").any() else "n/a")
        + " | Prefer age-band models in cancer_ever_model_specs.csv",
        "sensitivity": f"LMF deaths: all-cause={n_dead}, cancer={n_cdead}. Cox: {cox_txt}. {pwr_txt}",
        "ml": "n/a — see docs/cancer_module_report.md and cancer_evidence_brief.md",
        "verdict": "BUSTED as a slogan; UNTESTABLE for low-HR long-latency causation",
        "takeaway": (
            "Slogan “diet soda gives you cancer” is not supported: (1) IARC 2B is hazard with limited evidence, "
            "JECFA kept ADI many cans/day order-of-magnitude; (2) crude ever-cancer % is inflated by who drinks diet soda "
            "(older/sicker)—age stratification shrinks the scare but does not erase every gap; residual confounding remains; "
            "(3) cancer-death Cox HR is non-significant with wide CIs and limited events (underpowered for small risks). "
            "We cannot prove lifelong safety or site-specific HCC risk from NHANES."
        ),
        "limits": "24h diet; self-report ever-cancer; public LMF; no incidence registry; Cox unweighted primary.",
    }
    verdicts.append(m4)

    # M5
    top_smd = smd.sort_values("smd", key=abs, ascending=False).head(5) if smd is not None else None
    m5 = {
        "myth_id": "M5",
        "claim": "Diet soda drinkers are just like regular soda drinkers",
        "viral_claim_vs_test": "Assume same people. Test: SMD love plot + classifier.",
        "crude": f"Group n: {exp.set_index('bev_group')['n'].to_dict()}",
        "adjusted": f"Top |SMD| ASB vs Neither: {top_smd.to_dict(orient='records') if top_smd is not None else 'n/a'}",
        "sensitivity": f"ASB classifier AUC weighted={ml.get('asb_classifier_auc_weighted')}, unweighted={ml.get('asb_classifier_auc_unweighted')}",
        "ml": f"Top features: {ml.get('top_features', [])[:5]}",
        "verdict": "BUSTED",
        "takeaway": "Diet-soda consumers differ systematically (age, sex, diabetes, BMI, SES). Treating them as random is wrong.",
        "limits": "Profile ≠ moral judgment; confounding structure.",
    }
    verdicts.append(m5)

    # M6 — same outcome (BMI) for dose crude vs covariate-adjusted dose model
    m6 = {
        "myth_id": "M6",
        "claim": "Any amount is poison / one can is harmless always",
        "viral_claim_vs_test": "Dose models: BMI ~ diet-soft servings (continuous).",
        "crude": "BMI ~ asb_serv (minimal age/sex in S6 still has full lifestyle set): "
        + _coef_line(res, "bmi", "S6_dose", "asb_serv_d1")
        + " | binary ASB-only S0: "
        + _coef_line(res, "bmi", "S0"),
        "adjusted": "BMI ~ asb_serv with lifestyle covariates is S6_dose above; binary ASB S3: "
        + _coef_line(res, "bmi", "S3")
        + " | HbA1c dose (secondary): "
        + _coef_line(res, "hba1c", "S6_dose", "asb_serv_d1"),
        "sensitivity": "Heavy zero-inflation: most adults drink 0 diet soft drinks on recall day. Do not mix BMI and HbA1c in one crude/adjusted pair.",
        "ml": "n/a",
        "verdict": "NUANCED",
        "takeaway": "Binary any/none and continuous dose can disagree; single 24h recall misclassifies usual intake.",
        "limits": "Day-1 recall noise at low doses.",
    }
    verdicts.append(m6)

    # M7
    m7 = {
        "myth_id": "M7",
        "claim": "Results are p-hacked / only one definition works",
        "viral_claim_vs_test": "Specification curve across S0–S7.",
        "crude": "See multiverse_bmi.csv and myth_m7_spec_curve_bmi.png",
        "adjusted": f"BMI ASB-only specs run: {len(res[(res.outcome=='bmi')&(res.term=='asb_only')])}",
        "sensitivity": "Exposure multiverse E1–E5 documented in exposure_definitions.md",
        "ml": "n/a",
        "verdict": "PROCESS PASS (transparency)",
        "takeaway": "We publish the curve, not a single p-value.",
        "limits": "Multiverse still researcher-designed.",
    }
    verdicts.append(m7)

    # M8
    m8 = {
        "myth_id": "M8",
        "claim": "Labs prove diet soda is the driver",
        "viral_claim_vs_test": "Predict ASB use from labs/demographics; predict BMI with/without ASB.",
        "crude": f"BMI R² without ASB={ml.get('bmi_r2_without_asb')}, with={ml.get('bmi_r2_with_asb')}",
        "adjusted": f"ΔR²={ml.get('bmi_delta_r2')}",
        "sensitivity": f"Weighted vs unweighted AUC: {ml.get('asb_classifier_auc_weighted')} vs {ml.get('asb_classifier_auc_unweighted')}",
        "ml": str(ml.get("best_params")),
        "verdict": "NUANCED — predictive ASB signal is small vs health selection features",
        "takeaway": "If BMI/diabetes predict diet-soda use better than soda predicts BMI, the causal arrow in headlines is often backwards.",
        "limits": "Predictive ≠ causal.",
    }
    verdicts.append(m8)

    vdf = pd.DataFrame(verdicts)
    vdf.to_csv(paths["tables"] / "myth_verdict_summary.csv", index=False)

    # Markdown report
    lines = [
        "# Diet Soda Myth Lab — Verdicts",
        "",
        "Independent public-data stress tests using NHANES 2011–2018 + WWEIA + public mortality.",
        "",
        f"Analytic n ≈ {len(ar):,}. Models: weighted least squares / GLM (see `model_cardio_ladder.csv`).",
        "",
        "> **Read before citing p-values:** Continuous models use normalized multi-cycle MEC weights but **not** full NCHS design-based SEs (PSU/strata). Prefer **coefficients and CIs as approximate**. Binary/mortality GLMs use normalized weights (never raw MEC as frequency weights). Triglyceride models are on **log(TG)** scale.",
        "",
        "**Verdict ladder:** BUSTED · NUANCED · SUPPORTED (assoc.) · UNTESTABLE HERE",
        "",
        "---",
        "",
    ]
    for v in verdicts:
        lines += [
            f"## {v['myth_id']}: {v['claim']}",
            "",
            f"**Verdict:** {v['verdict']}",
            "",
            f"- **Viral claim vs our test:** {v['viral_claim_vs_test']}",
            f"- **Crude:** {v['crude']}",
            f"- **Adjusted:** {v['adjusted']}",
            f"- **Sensitivity:** {v['sensitivity']}",
            f"- **ML:** {v['ml']}",
            f"- **Takeaway:** {v['takeaway']}",
            f"- **Limits:** {v['limits']}",
            "",
            "---",
            "",
        ]
    lines += [
        "## Cancer evidence context (not NHANES microdata)",
        "",
        "- **IARC (2023):** aspartame Group **2B** (possibly carcinogenic) — *limited* evidence.",
        "- **JECFA:** ADI **0–40 mg/kg/day** reaffirmed; human cancer evidence *not convincing* at usual intakes.",
        "- **FDA:** continues to allow aspartame under approved conditions.",
        "- Group 2B ≠ “diet soda gives you cancer” in plain language; hazard ≠ risk at dietary doses.",
        "",
        "## Anti-pattern checklist",
        "",
        "- [x] Multi-cycle pool (not one Kaggle year)",
        "- [x] WWEIA official diet soft drink codes (not keyword-only)",
        "- [x] Soda-type exclusivity (not “only beverage in the universe”)",
        "- [x] Reverse-causation exclusions",
        "- [x] No causal ML claims",
        "- [x] Sampling weights **not** abused as GLM frequency counts (normalized only)",
        "- [~] Full NCHS design-based SE (PSU/strata) — **not implemented**; treat p-values as approximate",
        "",
        "## Known statistical caveats (post self-audit)",
        "",
        "1. Continuous models: WLS with normalized multi-cycle MEC weights — good for **point estimates**, optimistic for **SEs** (no cluster design).",
        "2. Binary/mortality: binomial GLM with **normalized** weights or unweighted; raw `WTMEC2YR` as `freq_weights` was a bug and was fixed.",
        "3. Triglycerides: fasting lab; prefer `*_fast` rows in model table when present.",
        "",
    ]
    paths["docs"].mkdir(parents=True, exist_ok=True)
    (paths["docs"] / "myth_verdicts.md").write_text("\n".join(lines), encoding="utf-8")
    (paths["docs"] / "cancer_evidence_brief.md").write_text(
        """# Cancer evidence brief (aspartame / diet soda)

## What agencies said (2023)

| Body | Role | Conclusion (plain language) |
|------|------|------------------------------|
| **IARC** | Hazard identification | Aspartame **Group 2B** — *possibly* carcinogenic; evidence limited (incl. signals discussed for liver cancer) |
| **JECFA** | Risk / ADI | ADI **0–40 mg/kg body weight/day** kept; association evidence **not convincing** for cancer at usual use |
| **FDA** | US food additive | Aspartame allowed under approved conditions; disagrees that available studies justify treating aspartame as a likely human carcinogen at labeled use |
| **NCI** | Public summary | Emphasizes limited evidence and distinction between hazard classes and everyday risk |

## How this maps to Myth M4

- Viral posts collapse **2B hazard** into **“causes cancer.”** That is not what IARC+JECFA jointly communicate.
- Example dose intuition often cited: many adults would need **many cans/day** of diet soda alone to exceed ADI (depends on aspartame mg/can and body weight)—still not a free pass for unlimited intake, but it undercuts “one sip = cancer.”
- **Our NHANES tests** cannot replace multi-decade cohorts (NutriNet, etc.). They *can* show whether simple cross-sectional scare patterns survive age/smoking adjustment and whether mortality events are too few for dramatic claims.

## References to expand

- IARC/WHO joint release on aspartame (July 2023)
- FDA aspartame consumer page
- NCI artificial sweeteners fact sheet
""",
        encoding="utf-8",
    )
    print("Wrote myth_verdicts.md and myth_verdict_summary.csv")


def main():
    run_verdicts()


if __name__ == "__main__":
    main()
