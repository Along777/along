"""
data.py
=======
Load the model-ready panel, build EKC-specific features, construct the analysis
samples, and split temporally for out-of-sample validation.

Baseline sample always excludes micro-states/territories (population < 1M,
`is_micro_state`), consistent with the data audit -- their per-capita emissions
are noisy and they distort small-N group statistics.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# load + features                                                              #
# --------------------------------------------------------------------------- #

def load_panel(path=None) -> pd.DataFrame:
    path = path or C.IN_PATH
    if not path.exists():
        sys.exit(f"Input not found: {path}. Run build_panel.py first.")
    df = pd.read_csv(path)
    if "is_micro_state" in df.columns:
        df = df[~df["is_micro_state"]].copy()
    return df


def add_ekc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the MEAN-CENTERED income polynomial and the income x governance
    interaction, and record the centering mean on the config module so the
    turning point can be mapped back to the raw ($) income scale.

    Centering (using log_gdp - mean and its square) is a conditioning fix: it
    removes the mechanical linear/quadratic collinearity (VIF ~370 -> ~1)
    without changing the squared-term coefficient, the fitted values, or the
    turning-point location.
    """
    df = df.copy()
    # centre on the mean of the estimable rows (where both target and income
    # exist) so the mean is not skewed by rows dropped downstream
    mask = df[C.GDP_RAW].notna() & df[C.TARGET].notna()
    gdp_mean = float(df.loc[mask, C.GDP_RAW].mean())
    C.GDP_MEAN = gdp_mean

    df[C.GDP] = df[C.GDP_RAW] - gdp_mean          # centered log GDP
    df[C.GDP_SQ] = df[C.GDP] ** 2                  # centered, squared
    if C.MODERATOR in df.columns:
        df[C.GDP_X_GOV] = df[C.GDP] * df[C.MODERATOR]
    return df


def turning_point_to_usd(vertex_centered: float) -> float:
    """Map a quadratic vertex expressed in CENTERED log-GDP units back to raw
    GDP-per-capita dollars: exp(mean + vertex_centered)."""
    if C.GDP_MEAN is None or not np.isfinite(vertex_centered):
        return np.nan
    return float(np.exp(C.GDP_MEAN + vertex_centered))


def within_between(df: pd.DataFrame, col: str) -> dict:
    """Decompose the variance of `col` into within-country and between-country
    components -- shows how much identifying variation fixed effects use up.

    Uses a size-weighted (observation-level) sum-of-squares decomposition,
    which is the correct split for an unbalanced panel: total SS = between SS
    (group means vs grand mean, weighted by group size) + within SS (deviations
    from own group mean).
    """
    d = df[["iso3", col]].dropna()
    if d.empty:
        return {"overall_var": np.nan, "between_var": np.nan,
                "within_var": np.nan, "within_share": np.nan}
    grand = d[col].mean()
    grp_mean = d.groupby("iso3")[col].transform("mean")
    ss_total = float(((d[col] - grand) ** 2).sum())
    ss_within = float(((d[col] - grp_mean) ** 2).sum())
    ss_between = float(((grp_mean - grand) ** 2).sum())
    return {"overall_var": ss_total, "between_var": ss_between,
            "within_var": ss_within,
            "within_share": ss_within / ss_total if ss_total else np.nan}


# --------------------------------------------------------------------------- #
# samples                                                                       #
# --------------------------------------------------------------------------- #

def build_sample(df: pd.DataFrame, scope: str = "core") -> pd.DataFrame:
    """Return a clean, complete-case frame for a given modeling scope.

    scope:
        "core"     -> target + gdp + gdp^2 + moderator
        "full"     -> core + all controls
        "no_petro" -> full, excluding petrostates
    """
    need = [C.TARGET, C.GDP, C.GDP_SQ, C.MODERATOR, C.GDP_X_GOV]
    if scope in ("full", "no_petro"):
        need = need + C.CONTROLS
    sub = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    if scope == "no_petro":
        sub = sub[~sub["iso3"].isin(C.PETROSTATES)]
    return sub


def temporal_split(df: pd.DataFrame, split_year=None):
    """Train on year <= split_year, test on year > split_year."""
    split_year = split_year or C.SPLIT_YEAR
    train = df[df["year"] <= split_year].copy()
    test = df[df["year"] > split_year].copy()
    return train, test
