"""
config.py
=========
Constants, paths, and feature lists for the EKC pipeline. No logic here --
everything downstream imports from this module so the analysis is defined in
one place.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #
IN_PATH = Path("output/panel_model_ready.csv")
OUT_DIR = Path("output/ekc")

# --------------------------------------------------------------------------- #
# variables                                                                    #
# --------------------------------------------------------------------------- #
# Headline target: CO2 emissions per capita (tonnes). Alternatives are used in
# the robustness battery to check the story isn't an artifact of one measure.
TARGET = "co2_pc"
ALT_TARGETS = ["co2_per_1000gdp", "co2_total_mt"]  # co2_total_mt -> log1p in code

# EKC core regressors. The income terms are MEAN-CENTERED (see
# data.add_ekc_features): using (log_gdp - mean) and its square instead of the
# raw level and square collapses the linear/quadratic collinearity (VIF ~370 ->
# ~1) and makes the linear coefficient interpretable as the slope at mean
# income. Centering does NOT change the squared-term coefficient, the fitted
# values, or the turning-point location -- it is a conditioning fix. GDP_MEAN is
# filled in at feature-build time so the turning point can be mapped back to the
# raw income scale.
GDP = "log_gdp_pc_c"                       # centered log GDP per capita
GDP_SQ = "log_gdp_pc_c_sq"                 # centered, squared
GDP_RAW = "log_gdp_pc"                     # uncentered (for plotting on $ scale)
GDP_MEAN = None                            # set by data.add_ekc_features()
MODERATOR = "regulatory_quality"          # headline governance moderator
MODERATOR_ALT = "rule_of_law"             # robustness swap
GDP_X_GOV = "log_gdp_pc_x_gov"            # interaction term name (centered GDP x gov)

# Controls added at rung 4 of the specification ladder.
CONTROLS = ["renew_share", "urban_pct", "trade_pct_gdp",
            "fossil_fuel_pct", "dependency_ratio"]

# Carbon-intensity decoupling uses this within-country time trend.
INTENSITY = "co2_per_1000gdp"
TREND = "trend"

# --------------------------------------------------------------------------- #
# samples                                                                      #
# --------------------------------------------------------------------------- #
# Petrostates: extreme CO2-per-capita outliers whose emissions are driven by
# oil/gas extraction rather than the income-emissions mechanism the EKC is
# about. Excluded in a robustness check (not the baseline) to show how much
# curvature they drive.
PETROSTATES = ["QAT", "BHR", "KWT", "ARE", "OMN", "SAU", "TTO", "BRN"]

# Ordered for consistent plotting / reporting.
INCOME_ORDER = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]

# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #
SPLIT_YEAR = 2018            # train on year <= SPLIT_YEAR, test on year > SPLIT_YEAR
# Inner temporal validation cutoff for GBM hyperparameter selection: fit on
# year <= TUNE_YEAR, score on TUNE_YEAR < year <= SPLIT_YEAR. Keeps tuning
# strictly out-of-sample (no leakage from the final test years).
TUNE_YEAR = 2014
GBM_PARAMS = dict(               # fixed params (defaults; overridden by tuning)
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)
# Small time-series-aware grid searched over the inner fold.
GBM_GRID = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3],
}
PERM_IMPORTANCE_REPS = 20        # repeats for held-out permutation importance

# --------------------------------------------------------------------------- #
# inference                                                                     #
# --------------------------------------------------------------------------- #
BOOTSTRAP_REPS = 500         # block (country) bootstrap reps for turning-point CI
BOOTSTRAP_SEED = 42
