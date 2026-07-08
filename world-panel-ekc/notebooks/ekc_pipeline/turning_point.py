"""
turning_point.py
================
Estimate the EKC turning point (the income level at which CO2 per capita stops
rising and starts falling) and put a confidence interval on it via a **block
bootstrap** that resamples whole countries -- respecting the panel structure
(observations within a country are dependent).

The headline question this answers: is the inverted-U turning point actually
inside the observed income range, and is it statistically distinguishable from
"emissions rise monotonically with income"?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from . import config as C
from . import data as D
from .models import to_panel, _turning_point


def _fit_quadratic(sub: pd.DataFrame, x_cols):
    panel = to_panel(sub, [C.TARGET] + x_cols)
    y = panel[C.TARGET]
    X = sm.add_constant(panel[x_cols])
    return PanelOLS(y, X, entity_effects=True, time_effects=True,
                    check_rank=False).fit()


def _block_bootstrap_tp(sub: pd.DataFrame, x_cols, reps, seed):
    """Resample countries with replacement; refit; collect turning points ($).

    Pre-splits the panel into per-country frames once (avoids re-scanning the
    whole frame for every resampled country on every rep -- the dominant cost)."""
    rng = np.random.default_rng(seed)
    by_iso = {iso: g.copy() for iso, g in sub.groupby("iso3")}
    isos = np.array(list(by_iso))
    tps = []
    for _ in range(reps):
        pick = rng.choice(isos, size=len(isos), replace=True)
        parts = []
        for i, iso in enumerate(pick):
            d = by_iso[iso].copy()
            d["iso3"] = f"{iso}__{i}"   # distinct entity per draw of a duplicate
            parts.append(d)
        boot = pd.concat(parts, ignore_index=True)
        try:
            tp = _turning_point(_fit_quadratic(boot, x_cols))
        except Exception:
            tp = np.nan
        tps.append(tp)
    return np.array(tps, dtype=float)


def estimate(df: pd.DataFrame, reps=None, seed=None):
    """Turning point + bootstrap CI for two specs: FE quadratic without and
    with controls. Returns (table, bootstrap_samples_dict)."""
    reps = reps or C.BOOTSTRAP_REPS
    seed = seed or C.BOOTSTRAP_SEED
    full = D.build_sample(df, "full")

    specs = {
        "FE_quadratic": [C.GDP, C.GDP_SQ],
        "FE_quadratic_controls": [C.GDP, C.GDP_SQ] + C.CONTROLS,
    }
    # observed income range in raw dollars (C.GDP is now CENTERED log-GDP, so
    # exp() of it would be wrong -- use the raw log column)
    gdp_lo = float(np.exp(full[C.GDP_RAW].min()))
    gdp_hi = float(np.exp(full[C.GDP_RAW].max()))

    # a turning point is only economically meaningful if it lands near the
    # observed income range; allow a generous 5x margin either side for the CI
    plaus_lo, plaus_hi = gdp_lo / 5, gdp_hi * 5

    rows, samples = [], {}
    for name, x_cols in specs.items():
        res = _fit_quadratic(full, x_cols)
        point = _turning_point(res)
        boot = _block_bootstrap_tp(full, x_cols, reps, seed)
        finite = boot[np.isfinite(boot)]
        in_range = finite[(finite >= plaus_lo) & (finite <= plaus_hi)]
        samples[name] = in_range              # only plottable values
        # need a reasonable number of in-range fits for a CI to be meaningful;
        # a single stray value would otherwise give a degenerate lo==hi interval
        if len(in_range) >= 10:
            lo, hi = np.percentile(in_range, [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows.append({
            "spec": name,
            "b_gdp_sq": round(float(res.params[C.GDP_SQ]), 4),
            "gdp_sq_p": round(float(res.pvalues[C.GDP_SQ]), 4),
            "turning_point_usd": round(point, 0) if np.isfinite(point) else np.nan,
            "ci_lo_usd": round(lo, 0) if np.isfinite(lo) else np.nan,
            "ci_hi_usd": round(hi, 0) if np.isfinite(hi) else np.nan,
            # fraction of bootstraps producing an inverted-U with a peak inside
            # the plausible income range -- low => the "peak" is an artifact
            "inverted_u_boot_share": round(float(len(in_range) / max(len(boot), 1)), 3),
            "gdp_range_lo": round(gdp_lo, 0), "gdp_range_hi": round(gdp_hi, 0),
            "tp_in_sample": bool(np.isfinite(point) and gdp_lo <= point <= gdp_hi),
        })
    table = pd.DataFrame(rows)
    table.to_csv(C.OUT_DIR / "turning_points.csv", index=False)
    return table, samples
