"""
Shared FE estimators and effect formatting for heat–crime analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as spstats

from panel import CTRL_M2, HAC_DEFAULT


def pct_from_b(b: float, lo: float, hi: float, p: float) -> dict:
    return {
        "pct": float(100 * np.expm1(b)),
        "lo": float(100 * np.expm1(lo)),
        "hi": float(100 * np.expm1(hi)),
        "p": float(p),
        "beta": float(b),
    }


def pct_ci(res, term: str) -> dict:
    b = float(res.params[term])
    lo, hi = res.conf_int().loc[term]
    return pct_from_b(b, float(lo), float(hi), float(res.pvalues[term]))


def fmt_pct(d: dict | tuple) -> str:
    if isinstance(d, dict):
        return f"{d['pct']:+.1f}% [{d['lo']:+.1f}, {d['hi']:+.1f}]"
    return f"{d[0]:+.1f}% [{d[1]:+.1f}, {d[2]:+.1f}]"


def fit_ols(
    yvar: str,
    xterms: str,
    data: pd.DataFrame,
    hac: int = HAC_DEFAULT,
    controls: str = CTRL_M2,
):
    return smf.ols(f"np.log({yvar}) ~ {xterms} + {controls}", data=data).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(hac)}
    )


def fit_ols_cluster(
    yvar: str,
    xterms: str,
    data: pd.DataFrame,
    group: str,
    controls: str = CTRL_M2,
):
    return smf.ols(f"np.log({yvar}) ~ {xterms} + {controls}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[group]}
    )


def standardized_effect(res, term: str, x_sd: float) -> dict:
    """% effect per 1 SD of the treatment (using log-linear approx on beta * sd)."""
    b = float(res.params[term])
    se = float(res.bse[term])
    b_sd = b * x_sd
    se_sd = se * x_sd
    return pct_from_b(
        b_sd,
        b_sd - 1.96 * se_sd,
        b_sd + 1.96 * se_sd,
        float(res.pvalues[term]),
    )


def fit_dlag(yvar: str, L: int, data: pd.DataFrame, hac: int = HAC_DEFAULT):
    terms = ["tmax10"] + [f"tmax10_L{k}" for k in range(1, L + 1)]
    use = data.dropna(subset=terms).copy()
    res = fit_ols(yvar, " + ".join(terms), use, hac=hac)
    betas = [float(res.params[t]) for t in terms]
    cum_b = float(sum(betas))
    idx = [list(res.params.index).index(t) for t in terms]
    V = res.cov_params().values
    ones = np.zeros(len(res.params))
    ones[idx] = 1.0
    se = float(np.sqrt(ones @ V @ ones))
    z = cum_b / se if se > 0 else 0.0
    cum = pct_from_b(
        cum_b,
        cum_b - 1.96 * se,
        cum_b + 1.96 * se,
        float(2 * spstats.norm.sf(abs(z))),
    )
    return res, terms, cum


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values (positive FDR control)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[order[i]] = min(prev, 1.0)
    return adj.tolist()
