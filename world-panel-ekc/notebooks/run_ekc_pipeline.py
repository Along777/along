#!/usr/bin/env python3
"""
run_ekc_pipeline.py
===================
Orchestrator for the Environmental Kuznets Curve pipeline. Runs every stage in
the ekc_pipeline package and writes all outputs (tables, figures, report) into
output/ekc/.

Prereq: build_panel.py must have produced output/panel_model_ready.csv.
Requires: pandas, numpy, matplotlib, statsmodels, linearmodels, scikit-learn.

Run:
    python run_ekc_pipeline.py
    python run_ekc_pipeline.py path/to/panel_model_ready.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

from ekc_pipeline import config as C
from ekc_pipeline import data as D
from ekc_pipeline import models, diagnostics, turning_point, validation, report


def main():
    if len(sys.argv) > 1:
        C.IN_PATH = Path(sys.argv[1])
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] loading panel + building EKC features")
    df = D.load_panel()
    df = D.add_ekc_features(df)
    core = D.build_sample(df, "core")
    print(f"      core sample: {len(core):,} obs, {core['iso3'].nunique()} countries, "
          f"{int(core['year'].min())}-{int(core['year'].max())}")

    print("[2/7] specification ladder")
    spec_tbl, fitted = models.spec_ladder(df)
    print(spec_tbl[["spec", "n_obs", C.GDP, C.GDP_SQ]].to_string(index=False))

    print("[3/7] diagnostics (Mundlak / Hausman / serial corr / cross-sectional dependence / VIF)")
    diag_tbl, vif = diagnostics.run_all(df)
    print(diag_tbl[["test", "statistic", "p_value", "verdict"]].to_string(index=False))

    print("[4/7] turning point + block bootstrap")
    tp_tbl, tp_samples = turning_point.estimate(df)
    print(tp_tbl[["spec", "b_gdp_sq", "turning_point_usd", "tp_in_sample"]].to_string(index=False))

    print("[5/7] governance moderation + heterogeneity + robustness")
    gov_moderation, gov_pooled = models.governance_moderation(df)
    print(gov_moderation.to_string(index=False))
    het_income = models.heterogeneity_by_income(df)
    decoupling = models.decoupling_by_income(df)
    het_region = models.heterogeneity_by_region(df)
    robustness = models.robustness_battery(df)
    print(decoupling.to_string(index=False))

    print("[6/7] out-of-sample validation (levels persistence + change task)")
    val = validation.run(df)
    print("levels:"); print(val["level_metrics"].to_string(index=False))
    if val["change_metrics"] is not None:
        print("changes:"); print(val["change_metrics"].to_string(index=False))

    print("[7/7] figures + report")
    report.fig_target_distribution(df)
    report.fig_within_between(df)
    report.fig_ekc_curves_by_income(df, het_income)
    report.fig_governance_interaction(df, gov_pooled)
    report.fig_decoupling(df)
    report.fig_turning_point_bootstrap(tp_samples)
    report.fig_feature_importance(val["level_imp"])
    report.fig_oos_predictions(val["level_preds"])
    report.fig_oos_change_predictions(val["change_preds"])
    report.write_report(spec_tbl, diag_tbl, vif, tp_tbl, het_income, decoupling,
                        het_region, robustness, gov_moderation,
                        fitted.get("se_comparison"), val,
                        C.OUT_DIR / "EKC_REPORT.md")

    print(f"\nDone. Outputs in {C.OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
