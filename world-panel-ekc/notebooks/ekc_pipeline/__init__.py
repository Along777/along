"""
ekc_pipeline
============
A modular econometric + ML pipeline studying the Environmental Kuznets Curve
(EKC), the moderating role of governance quality, and income-conditional
carbon decoupling, built on the country-year panel from build_panel.py.

Stages (run via ../run_ekc_pipeline.py):
    config       constants and paths
    data         load / feature-build / sample / temporal split
    models       nested specification ladder + heterogeneity + robustness
    diagnostics  Hausman, VIF, serial correlation, cross-sectional dependence
    turning_point turning-point estimate + block-bootstrap CI
    validation   temporal holdout: FE model vs gradient-boosting benchmark
    report       figures + EKC_REPORT.md
"""
