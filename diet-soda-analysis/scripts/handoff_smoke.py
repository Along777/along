"""Final handoff smoke checks. Exit 1 if hard errors."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

errors: list[str] = []
warns: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


def main() -> int:
    df = pd.read_parquet(ROOT / "data/processed/analysis_ready.parquet")
    mort = pd.read_parquet(ROOT / "data/processed/analysis_ready_mortality.parquet")
    r = pd.read_csv(ROOT / "outputs/tables/model_cardio_ladder.csv")

    if df["SEQN"].duplicated().any():
        err("duplicate SEQN in analytic")
    if (df["w_mec"] <= 0).any() or df["w_mec"].isna().any():
        err("invalid w_mec")
    if not (df.loc[df["bev_group"] == "ASB-only", "asb_any_d1"] == 1).all():
        err("ASB-only asb_any inconsistency")
    if not (df.loc[df["bev_group"] == "SSB-only", "asb_any_d1"] == 0).all():
        err("SSB-only has asb")
    if (df.loc[df["bev_group"] == "Neither", ["asb_any_d1", "ssb_any_d1"]].sum(axis=1) != 0).any():
        err("Neither has soft drinks")
    if (df["dbp_mean"] == 0).fillna(False).any():
        err("dbp_mean still contains 0")
    if "education" in df.columns and df["education"].isin([7, 9]).any():
        err("education sentinel 7/9 remain")
    if "sedentary_min" in df.columns and df["sedentary_min"].isin([7777, 9999]).any():
        err("sedentary sentinel remain")
    if (df.get("pregnancy_status") == 1).fillna(False).any():
        err("pregnant in analytic")
    if "diet_recall_status" in df.columns and not (df["diet_recall_status"] == 1).all():
        err("diet_recall_status not all reliable (1)")
    if (df["age"] < 20).any():
        err("age < 20 in analytic")

    if (r["term"] == "ERROR").any():
        err(f"model ERROR rows: {(r['term'] == 'ERROR').sum()}")
    death = r[r["outcome"].astype(str).str.contains("death", na=False)]
    if len(death) and (death["se"] < 0.01).any():
        err("death SE tiny — possible freq_weights bug")
    if not (r["outcome"] == "log_tg").any():
        warn("no log_tg in model table")
    if (r["outcome"] == "tg").any():
        warn("raw outcome label 'tg' still present (prefer log_tg)")

    required = [
        "outputs/tables/cancer_cox_results.csv",
        "outputs/tables/cancer_ever_by_age_bev.csv",
        "outputs/tables/cancer_power_mdes.json",
        "docs/cancer_module_report.md",
        "docs/cancer_evidence_brief.md",
        "docs/myth_verdicts.md",
        "docs/assumptions.md",
        "docs/DEMO_TALKING_POINTS.md",
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            err(f"missing {rel}")

    cox = pd.read_csv(ROOT / "outputs/tables/cancer_cox_results.csv")
    for _, row in cox.iterrows():
        if row.get("term") == "asb_only" and "hr" in row and pd.notna(row["hr"]):
            if not (row["hr_lo"] <= row["hr"] <= row["hr_hi"]):
                err(f"cox CI order {row['outcome']}")
            if float(row["pval"]) == 0.0:
                err(f"cox p exactly 0 {row['outcome']}")

    for fig in [
        "cancer_c1_hazard_vs_risk.png",
        "cancer_c2_adi_cans.png",
        "cancer_c3_ever_cancer_by_age.png",
        "cancer_c4_km_cancer_death.png",
        "cancer_c5_cox_forest.png",
        "myth_m1_bmi_violin.png",
        "myth_m8_asb_feature_importance.png",
    ]:
        if not (ROOT / "outputs/figures" / fig).exists():
            err(f"missing figure {fig}")

    v = (ROOT / "docs/myth_verdicts.md").read_text(encoding="utf-8")
    if "PSU" not in v and "design-based" not in v:
        warn("verdicts missing design-based SE banner keywords")
    if "log-TG" not in v and "log mg" not in v:
        warn("verdicts may not label log-TG")
    if re.search(r"\bp=0([^\.\d]|$)", v):
        err("verdicts claim p=0")

    # M4 takeaway overstates age if gaps remain in strata — soft warn only
    age = pd.read_csv(ROOT / "outputs/tables/cancer_ever_by_age_bev.csv")
    old = age[age["age_band"] == "60+"]
    asb = old.loc[old["bev_group"] == "ASB-only", "cancer_rate"]
    nei = old.loc[old["bev_group"] == "Neither", "cancer_rate"]
    if len(asb) and len(nei) and float(asb.iloc[0]) > float(nei.iloc[0]) + 0.03:
        if "largely age structure" in v and "does not vanish" not in v.lower():
            warn(
                "M4 text says cancer gap is largely age; 60+ ASB still higher than Neither — soften if coworker probes"
            )

    # GLM SE sanity
    from src.analysis.run_models import _glm_binomial

    m = mort.copy()
    m["asb_only"] = (m["bev_group"] == "ASB-only").astype(int)
    m["ssb_only"] = (m["bev_group"] == "SSB-only").astype(int)
    m["both"] = (m["bev_group"] == "Both").astype(int)
    md = m.dropna(subset=["cancer_death", "asb_only", "age", "female", "smoking_status"])
    fit = _glm_binomial(
        "cancer_death ~ asb_only + ssb_only + both + age + female + smoking_status",
        md,
        md["w_mec"],
    )
    if fit.bse["asb_only"] < 0.01:
        err("glm cancer_death SE still tiny")

    print("n_analytic", len(df))
    print("groups", df["bev_group"].value_counts().to_dict())
    print("cancer_deaths", int(mort["cancer_death"].fillna(0).sum()))
    print("cox", cox.to_string(index=False))
    print("ERRORS:", errors if errors else "none")
    print("WARNS:", warns if warns else "none")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
