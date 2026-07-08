"""
models.py
=========
The econometric core: a nested specification ladder for the EKC, robust
inference (Driscoll-Kraay), heterogeneity by income group and region, and a
robustness battery.

Design note on inference
------------------------
The headline models report **Driscoll-Kraay** HAC standard errors
(`cov_type="kernel"`), which are robust to both serial correlation *and*
cross-sectional dependence -- the two problems the diagnostics module confirms
are present in this panel. Clustered-by-country SEs are reported alongside for
comparison but are not the headline, because they do not handle the strong
cross-sectional dependence (global emission cycles, shared shocks) that a
country panel of CO2 exhibits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from . import config as C
from . import data as D


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #

def to_panel(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Long df -> MultiIndex(iso3, year) frame for linearmodels PanelOLS."""
    sub = df[["iso3", "year"] + cols].dropna()
    return sub.set_index(["iso3", "year"])


def _fit(panel, y_col, x_cols, entity=False, time=False, cov="kernel"):
    y = panel[y_col]
    X = sm.add_constant(panel[x_cols])
    mod = PanelOLS(y, X, entity_effects=entity, time_effects=time,
                   drop_absorbed=True, check_rank=False)
    if cov == "kernel":
        return mod.fit(cov_type="kernel", kernel="bartlett")
    if cov == "clustered":
        return mod.fit(cov_type="clustered", cluster_entity=True)
    return mod.fit(cov_type="unadjusted")


def _row(name, res, keep, se_type="", has_entity=True):
    """Pull a compact result row (coef + stars) for the spec-ladder table.

    rsq_within is only meaningful for models with entity effects; for the
    pooled rung it is a misleading artifact, so we suppress it there.
    """
    rsq_w = (round(float(res.rsquared_within), 3)
             if has_entity and hasattr(res, "rsquared_within") else np.nan)
    out = {"spec": name, "se_type": se_type, "n_obs": int(res.nobs),
           "rsq_within": rsq_w}
    for k in keep:
        if k in res.params.index:
            coef, p = res.params[k], res.pvalues[k]
            stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
            out[k] = f"{coef:.3f}{stars}"
            out[f"{k}_p"] = round(float(p), 3)
        else:
            out[k] = ""
            out[f"{k}_p"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# specification ladder                                                          #
# --------------------------------------------------------------------------- #

def spec_ladder(df: pd.DataFrame):
    """Five nested specifications, from naive pooled quadratic to two-way FE
    with controls and the governance interaction. Returns (table, fitted_dict)
    where fitted_dict holds the estimated models for downstream plots.

    The headline finding lives in the first two rows: the pooled quadratic's
    curvature (sign of log_gdp_pc_sq) versus the fixed-effects curvature.
    """
    full = D.build_sample(df, "full")
    keep = [C.GDP, C.GDP_SQ, C.MODERATOR, C.GDP_X_GOV]
    rows, fitted = [], {}

    # 1) pooled OLS quadratic (no FE) -- the "textbook" EKC. Clustered-by-
    #    country SEs (Driscoll-Kraay needs entity effects to be meaningful here).
    p1 = to_panel(full, [C.TARGET, C.GDP, C.GDP_SQ])
    r1 = _fit(p1, C.TARGET, [C.GDP, C.GDP_SQ], entity=False, time=False, cov="clustered")
    rows.append(_row("1_pooled_quadratic", r1, keep, "clustered", has_entity=False))
    fitted["pooled"] = r1

    # 2) + entity (country) fixed effects. Driscoll-Kraay SEs from here on.
    p2 = to_panel(full, [C.TARGET, C.GDP, C.GDP_SQ])
    r2 = _fit(p2, C.TARGET, [C.GDP, C.GDP_SQ], entity=True, time=False)
    rows.append(_row("2_entity_FE", r2, keep, "Driscoll-Kraay")); fitted["entity_fe"] = r2

    # 3) + year fixed effects (two-way FE)
    r3 = _fit(p2, C.TARGET, [C.GDP, C.GDP_SQ], entity=True, time=True)
    rows.append(_row("3_twoway_FE", r3, keep, "Driscoll-Kraay")); fitted["twoway_fe"] = r3

    # 4) + controls
    x4 = [C.GDP, C.GDP_SQ] + C.CONTROLS
    p4 = to_panel(full, [C.TARGET] + x4)
    r4 = _fit(p4, C.TARGET, x4, entity=True, time=True)
    rows.append(_row("4_twoway_FE_controls", r4, keep, "Driscoll-Kraay"))
    fitted["twoway_controls"] = r4

    # 5) + governance interaction (headline)
    x5 = [C.GDP, C.GDP_SQ, C.MODERATOR, C.GDP_X_GOV] + C.CONTROLS
    p5 = to_panel(full, [C.TARGET] + x5)
    r5 = _fit(p5, C.TARGET, x5, entity=True, time=True)
    rows.append(_row("5_twoway_FE_governance", r5, keep, "Driscoll-Kraay"))
    fitted["governance"] = r5

    # refit rung 5 with clustered SEs and emit an explicit headline SE
    # comparison, so readers can see the point estimates are identical and only
    # the inference (SEs / p-values) changes between clustered and Driscoll-Kraay
    r5_cl = _fit(p5, C.TARGET, x5, entity=True, time=True, cov="clustered")
    fitted["governance_clustered"] = r5_cl
    se_cmp = pd.DataFrame({
        "coef": r5.params.round(4),
        "se_driscoll_kraay": r5.std_errors.round(4),
        "p_driscoll_kraay": r5.pvalues.round(4),
        "se_clustered": r5_cl.std_errors.round(4),
        "p_clustered": r5_cl.pvalues.round(4),
    })
    se_cmp.to_csv(C.OUT_DIR / "headline_se_comparison.csv")
    fitted["se_comparison"] = se_cmp

    table = pd.DataFrame(rows)
    table.to_csv(C.OUT_DIR / "spec_ladder.csv", index=False)
    return table, fitted


# --------------------------------------------------------------------------- #
# governance moderation: between-country vs within-country                      #
# --------------------------------------------------------------------------- #

def governance_moderation(df: pd.DataFrame):
    """Estimate the log_gdp x regulatory_quality interaction two ways so the
    report can contrast them:
      - pooled (year FE only): captures BETWEEN-country variation. Here
        higher-governance countries sit on a lower emissions-income curve.
      - two-way FE: captures WITHIN-country variation. The moderation largely
        vanishes -- it is a cross-sectional pattern, not a within-country law,
        which is the same fragility theme as the EKC shape itself.
    Returns (table, pooled_result) -- the pooled model is used to draw the
    (meaningful) governance-curve figure.
    """
    full = D.build_sample(df, "full")
    x = [C.GDP, C.GDP_SQ, C.MODERATOR, C.GDP_X_GOV] + C.CONTROLS
    panel = to_panel(full, [C.TARGET] + x)
    pooled = _fit(panel, C.TARGET, x, entity=False, time=True, cov="clustered")
    fe = _fit(panel, C.TARGET, x, entity=True, time=True)  # Driscoll-Kraay
    rows = []
    for label, res in [("pooled_between (year FE only)", pooled),
                       ("twoway_FE (within-country)", fe)]:
        coef, p = res.params[C.GDP_X_GOV], res.pvalues[C.GDP_X_GOV]
        stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
        rows.append({"model": label, "n_obs": int(res.nobs),
                     "gdp_x_gov_coef": f"{coef:.3f}{stars}",
                     "gdp_x_gov_p": round(float(p), 4)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.OUT_DIR / "governance_moderation.csv", index=False)
    return tbl, pooled


# --------------------------------------------------------------------------- #
# heterogeneity                                                                 #
# --------------------------------------------------------------------------- #

def _turning_point(res, gdp_c_min=None, gdp_c_max=None, margin=2.0):
    """Turning point (in raw GDP-per-capita $) of a quadratic in CENTERED
    log-GDP; np.nan if not an interior maximum or the vertex is implausibly far
    from the observed income range.

    The regressors are centered (log_gdp - mean), so the vertex in centered
    units is -b1/(2 b2); it is mapped back to dollars via exp(mean + vertex).
    When the observed centered income range is supplied the plausibility band is
    data-derived (range +/- `margin`); otherwise a generous absolute guard is
    used to prevent overflow from an unidentified (near-flat) quadratic.
    """
    if C.GDP not in res.params.index or C.GDP_SQ not in res.params.index:
        return np.nan
    b1, b2 = res.params[C.GDP], res.params[C.GDP_SQ]
    if b2 >= 0:                       # convex => no inverted-U max
        return np.nan
    vertex_c = -b1 / (2 * b2)          # vertex in centered log-GDP units
    if not np.isfinite(vertex_c):
        return np.nan
    if gdp_c_min is not None and gdp_c_max is not None:
        if vertex_c < gdp_c_min - margin or vertex_c > gdp_c_max + margin:
            return np.nan
    elif abs(vertex_c) > 10:           # ~exp(10) beyond any real income spread
        return np.nan
    return D.turning_point_to_usd(vertex_c)


def heterogeneity_by_income(df: pd.DataFrame):
    full = D.build_sample(df, "full")
    rows = []
    for grp in C.INCOME_ORDER:
        sub = full[full["income_group"] == grp]
        panel = to_panel(sub, [C.TARGET, C.GDP, C.GDP_SQ])
        if len(panel) < 40 or panel.reset_index()["iso3"].nunique() < 5:
            continue
        res = _fit(panel, C.TARGET, [C.GDP, C.GDP_SQ], entity=True, time=True)
        lo, hi = panel[C.GDP].min(), panel[C.GDP].max()
        rows.append({
            "income_group": grp, "n_obs": int(res.nobs),
            "b_gdp": round(float(res.params[C.GDP]), 3),
            "b_gdp_sq": round(float(res.params[C.GDP_SQ]), 3),
            "gdp_sq_p": round(float(res.pvalues[C.GDP_SQ]), 3),
            "turning_point_usd_pt": _turning_point(res, lo, hi),  # point est., no CI
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.OUT_DIR / "heterogeneity_income.csv", index=False)
    return tbl


def decoupling_by_income(df: pd.DataFrame):
    """Within-country trend in carbon intensity (co2_per_1000gdp) by income
    group. time_effects OFF because `trend` IS the year signal -- year FE would
    absorb it (bug caught in the earlier topic_models pass)."""
    rows = []
    for grp in C.INCOME_ORDER:
        sub = df.dropna(subset=[C.INTENSITY, C.TREND, "income_group"])
        sub = sub[sub["income_group"] == grp]
        panel = to_panel(sub, [C.INTENSITY, C.TREND])
        if len(panel) < 40 or panel[C.TREND].nunique() < 3:
            continue
        res = _fit(panel, C.INTENSITY, [C.TREND], entity=True, time=False, cov="clustered")
        mean_level = sub[C.INTENSITY].mean()
        rows.append({
            "income_group": grp, "n_obs": int(res.nobs),
            "trend_coef": round(float(res.params[C.TREND]), 5),
            "p_value": round(float(res.pvalues[C.TREND]), 4),
            "pct_per_year": round(float(res.params[C.TREND] / mean_level * 100), 3)
            if mean_level else np.nan,
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.OUT_DIR / "decoupling_by_income.csv", index=False)
    return tbl


def heterogeneity_by_region(df: pd.DataFrame):
    full = D.build_sample(df, "full")
    rows = []
    for grp, sub in full.groupby("region"):
        panel = to_panel(sub, [C.TARGET, C.GDP, C.GDP_SQ])
        if len(panel) < 40 or panel.reset_index()["iso3"].nunique() < 5:
            continue
        res = _fit(panel, C.TARGET, [C.GDP, C.GDP_SQ], entity=True, time=True)
        lo, hi = panel[C.GDP].min(), panel[C.GDP].max()
        rows.append({
            "region": grp, "n_obs": int(res.nobs),
            "b_gdp_sq": round(float(res.params[C.GDP_SQ]), 3),
            "gdp_sq_p": round(float(res.pvalues[C.GDP_SQ]), 3),
            "turning_point_usd_pt": _turning_point(res, lo, hi),  # point est., no CI
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.OUT_DIR / "heterogeneity_region.csv", index=False)
    return tbl


# --------------------------------------------------------------------------- #
# robustness battery                                                            #
# --------------------------------------------------------------------------- #

def robustness_battery(df: pd.DataFrame):
    """Re-estimate the headline curvature under several perturbations; if the
    EKC story is real it should survive, and if it's an artifact these should
    show it. Each row reports the squared-term coef, its p-value, and the
    implied turning point."""
    rows = []

    def add(label, sub, target, x_cols):
        panel = to_panel(sub, [target] + x_cols)
        if len(panel) < 40:
            return
        res = _fit(panel, target, x_cols, entity=True, time=True)
        lo, hi = panel[C.GDP].min(), panel[C.GDP].max()
        rows.append({
            "variant": label, "target": target, "n_obs": int(res.nobs),
            "b_gdp_sq": round(float(res.params[C.GDP_SQ]), 3),
            "gdp_sq_p": round(float(res.pvalues[C.GDP_SQ]), 3),
            "turning_point_usd_pt": _turning_point(res, lo, hi),  # point est., no CI
        })

    base_x = [C.GDP, C.GDP_SQ] + C.CONTROLS
    full = D.build_sample(df, "full")

    add("baseline_full", full, C.TARGET, base_x)
    add("exclude_petrostates", D.build_sample(df, "no_petro"), C.TARGET, base_x)

    # alternative dependent variables
    intens = full.dropna(subset=[C.INTENSITY])
    add("alt_DV_carbon_intensity", intens, C.INTENSITY, base_x)
    tot = full.copy()
    tot["log_co2_total"] = np.log1p(tot["co2_total_mt"])
    add("alt_DV_log_total_co2", tot.dropna(subset=["log_co2_total"]),
        "log_co2_total", base_x)

    # subperiods
    add("subperiod_pre2010", full[full["year"] < 2010], C.TARGET, base_x)
    add("subperiod_post2010", full[full["year"] >= 2010], C.TARGET, base_x)

    # alternative governance moderator (swap into the interaction)
    alt = df.copy()
    alt["log_gdp_pc_x_gov"] = alt[C.GDP] * alt[C.MODERATOR_ALT]
    alt_full = alt.dropna(subset=[C.TARGET, C.GDP, C.GDP_SQ, C.MODERATOR_ALT,
                                   "log_gdp_pc_x_gov"] + C.CONTROLS)
    add("alt_moderator_rule_of_law", alt_full, C.TARGET,
        [C.GDP, C.GDP_SQ, C.MODERATOR_ALT, "log_gdp_pc_x_gov"] + C.CONTROLS)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.OUT_DIR / "robustness.csv", index=False)
    return tbl
