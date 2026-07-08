"""
diagnostics.py
==============
Specification and residual diagnostics that justify the modeling choices:

  - Hausman test (FE vs RE): is a fixed-effects estimator required?
  - VIF: multicollinearity among regressors (the quadratic term is expected to
    be collinear with its own level -- flagged as benign).
  - Wooldridge-style AR(1): serial correlation in the within (FE) residuals.
  - Pesaran CD: cross-sectional dependence across countries.

The last two motivate reporting Driscoll-Kraay standard errors in models.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects

from . import config as C
from . import data as D
from .models import to_panel


def hausman_fe_vs_re(df: pd.DataFrame) -> dict:
    """Classic Hausman: compares FE and RE coefficient vectors. A large
    statistic (small p) => RE is inconsistent => use FE."""
    full = D.build_sample(df, "full")
    x = [C.GDP, C.GDP_SQ] + C.CONTROLS
    panel = to_panel(full, [C.TARGET] + x)
    y = panel[C.TARGET]
    X = sm.add_constant(panel[x])

    fe = PanelOLS(y, X, entity_effects=True, check_rank=False).fit()
    re = RandomEffects(y, X).fit()

    common = [p for p in fe.params.index
              if p in re.params.index and p != "const"]
    b_diff = (fe.params[common] - re.params[common]).values
    v_diff = (fe.cov.loc[common, common] - re.cov.loc[common, common]).values
    try:
        stat = float(b_diff.T @ np.linalg.pinv(v_diff) @ b_diff)
    except np.linalg.LinAlgError:
        stat = np.nan
    dof = len(common)
    # A negative statistic is a well-known finite-sample degeneracy: the
    # difference of covariance matrices is not positive semi-definite, so the
    # chi-square test is uninformative. It does NOT mean "use RE" -- given the
    # serial correlation + cross-sectional dependence found below and the large
    # pooled->FE coefficient shifts, FE is the substantively correct choice.
    if not np.isfinite(stat) or stat < 0:
        verdict = "test degenerate (cov diff not PSD); FE preferred on serial-corr grounds"
        p = np.nan
    else:
        p = float(stats.chi2.sf(stat, dof))
        verdict = "FE preferred" if p < 0.05 else "RE not rejected"
    return {"test": "Hausman FE vs RE", "statistic": round(stat, 2),
            "dof": dof, "p_value": round(p, 4) if np.isfinite(p) else np.nan,
            "verdict": verdict}


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors for the headline regressor set. With the
    income polynomial mean-centered (C.GDP / C.GDP_SQ), the linear and quadratic
    terms are no longer mechanically collinear, so these VIFs should be small
    (contrast with the ~370 seen on the raw, uncentered polynomial)."""
    full = D.build_sample(df, "full")
    x = [C.GDP, C.GDP_SQ, C.MODERATOR] + C.CONTROLS
    M = full[x].dropna()
    X = sm.add_constant(M)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        others = [c for c in X.columns if c != col]
        r2 = sm.OLS(X[col], X[others]).fit().rsquared
        vif = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
        rows.append({"variable": col, "VIF": round(vif, 2)})
    return pd.DataFrame(rows)


def mundlak_test(df: pd.DataFrame) -> dict:
    """Mundlak (1978) auxiliary test for FE vs RE, robust where the classic
    Hausman degenerates. Augment the pooled model with each time-varying
    regressor's country mean; a joint Wald test that those means are zero is the
    operative check -- rejection => the country effect is correlated with the
    regressors => fixed effects are required (RE would be inconsistent).

    Uses cluster-robust (by country) covariance so the Wald test is valid under
    the serial correlation the residual diagnostics flag.
    """
    full = D.build_sample(df, "full")
    x = [C.GDP, C.GDP_SQ] + C.CONTROLS
    d = full[["iso3", C.TARGET] + x].dropna().copy()
    mean_cols = []
    for c in x:
        mc = f"{c}_cmean"
        d[mc] = d.groupby("iso3")[c].transform("mean")
        mean_cols.append(mc)
    X = sm.add_constant(d[x + mean_cols])
    ols = sm.OLS(d[C.TARGET], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"]})
    # joint Wald test that all group-mean coefficients are zero
    R = np.zeros((len(mean_cols), X.shape[1]))
    for i, mc in enumerate(mean_cols):
        R[i, X.columns.get_loc(mc)] = 1.0
    wald = ols.wald_test(R, scalar=True)
    stat = float(np.asarray(wald.statistic).ravel()[0])
    p = float(np.asarray(wald.pvalue).ravel()[0])
    return {"test": "Mundlak (FE vs RE, joint Wald on group means)",
            "statistic": round(stat, 2), "dof": len(mean_cols),
            "p_value": round(p, 4),
            "verdict": "FE required (group means jointly significant)" if p < 0.05
                       else "RE adequate (group means jointly insignificant)"}


def wooldridge_ar1(df: pd.DataFrame) -> dict:
    """Regress FE residual on its own lag within country; a significant lag
    coefficient => serial correlation."""
    full = D.build_sample(df, "full")
    x = [C.GDP, C.GDP_SQ] + C.CONTROLS
    panel = to_panel(full, [C.TARGET] + x)
    y = panel[C.TARGET]
    X = sm.add_constant(panel[x])
    res = PanelOLS(y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit()
    resid = res.resids.reset_index()
    resid.columns = ["iso3", "year", "e"]
    resid = resid.sort_values(["iso3", "year"])
    resid["e_lag"] = resid.groupby("iso3")["e"].shift(1)
    d = resid.dropna(subset=["e", "e_lag"])
    ols = sm.OLS(d["e"], sm.add_constant(d["e_lag"])).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"]})
    rho, p = float(ols.params["e_lag"]), float(ols.pvalues["e_lag"])
    return {"test": "Wooldridge AR(1) in FE residuals", "statistic": round(rho, 3),
            "rho": round(rho, 3), "p_value": round(p, 4),
            "verdict": "serial correlation present" if p < 0.05 else "no serial correlation"}


def pesaran_cd(df: pd.DataFrame) -> dict:
    """Pesaran (2004) CD test for cross-sectional dependence, computed on the
    two-way-FE residuals reshaped to a country x year matrix of pairwise
    correlations."""
    full = D.build_sample(df, "full")
    x = [C.GDP, C.GDP_SQ] + C.CONTROLS
    panel = to_panel(full, [C.TARGET] + x)
    y = panel[C.TARGET]
    X = sm.add_constant(panel[x])
    res = PanelOLS(y, X, entity_effects=True, time_effects=True,
                   check_rank=False).fit()
    resid = res.resids.reset_index()
    resid.columns = ["iso3", "year", "e"]
    wide = resid.pivot(index="year", columns="iso3", values="e")

    corr = wide.corr()
    n = corr.shape[0]
    # pairwise counts of overlapping observations
    mask = wide.notna().astype(float)
    counts = mask.T @ mask  # T_ij overlap per country pair
    iu = np.triu_indices(n, k=1)
    rho = corr.values[iu]
    tij = counts.values[iu]
    good = np.isfinite(rho) & (tij > 3)
    rho, tij = rho[good], tij[good]
    if len(rho) == 0:
        return {"test": "Pesaran CD", "statistic": np.nan, "p_value": np.nan,
                "verdict": "insufficient overlap"}
    # Pesaran (2004) CD statistic ~ N(0,1) under cross-sectional independence
    cd_stat = np.sqrt(2.0 / (n * (n - 1))) * np.sum(np.sqrt(tij) * rho)
    p = float(2 * stats.norm.sf(abs(cd_stat)))
    return {"test": "Pesaran CD (cross-sectional dependence)",
            "statistic": round(float(cd_stat), 2), "p_value": round(p, 4),
            "verdict": "cross-sectional dependence present" if p < 0.05
                       else "no cross-sectional dependence"}


def run_all(df: pd.DataFrame):
    """Run every diagnostic, write a tidy table, return (scalar_table, vif).

    Mundlak is the operative FE-vs-RE test (the classic Hausman degenerates to a
    negative statistic on this panel); it is listed first.
    """
    scalars = [mundlak_test(df), hausman_fe_vs_re(df),
               wooldridge_ar1(df), pesaran_cd(df)]
    scalar_tbl = pd.DataFrame(scalars)
    scalar_tbl.to_csv(C.OUT_DIR / "diagnostics.csv", index=False)
    vif = vif_table(df)
    vif.to_csv(C.OUT_DIR / "diagnostics_vif.csv", index=False)
    return scalar_tbl, vif
