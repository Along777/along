from __future__ import annotations

"""Shared estimators for Return to Fire.

Trend half: negative-binomial GLM trends reported as incidence-rate ratios per
decade, Theil-Sen slopes, and Mann-Kendall tests -- every headline series gets
both a parametric and a nonparametric read before it is believed.

ML half: the honest-protocol helpers (temporal split, spatial blocks,
climatology baseline, metrics that are not bare accuracy) plus a faithful
reconstruction of the 2020 notebook's random forest for the museum table.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, recall_score)

import wildfire as wf

RNG = 42


# ---------------------------------------------------------------------------
# Trend estimators
# ---------------------------------------------------------------------------
@dataclass
class Trend:
    irr_decade: float          # NB2 incidence-rate ratio per +10 years
    irr_lo: float
    irr_hi: float
    p_nb: float
    theil_slope: float         # nonparametric slope per year (units of the series)
    theil_lo: float
    theil_hi: float
    mk_tau: float              # Mann-Kendall via Kendall's tau against year
    mk_p: float
    n_obs: int

    def to_dict(self) -> dict:
        return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 6))
                for k, v in self.__dict__.items()} | {"n_obs": int(self.n_obs)}


def nb_trend(df: pd.DataFrame, count_col: str = "n", year_col: str = "fire_year",
             state_fe: bool = True) -> Trend:
    """NB2 GLM count trend with optional state fixed effects + robustness reads.

    IRR per decade = exp(10 * beta_year). The nonparametric Theil-Sen slope and
    Mann-Kendall test run on the year-summed series so outlier years (2020)
    cannot quietly drive the headline.
    """
    d = df[[count_col, year_col] + (["state"] if state_fe else [])].dropna().copy()
    d["year_c"] = d[year_col] - d[year_col].mean()
    formula = f"{count_col} ~ year_c" + (" + C(state)" if state_fe and d["state"].nunique() > 1 else "")
    try:
        fit = smf.negativebinomial(formula, d).fit(disp=0, maxiter=200)
        beta, se = fit.params["year_c"], fit.bse["year_c"]
        p_nb = float(fit.pvalues["year_c"])
    except Exception:
        # NB MLE can fail on short/degenerate series; Poisson with robust SE is the fallback
        fit = smf.glm(formula, d, family=sm.families.Poisson()).fit(cov_type="HC1")
        beta, se = fit.params["year_c"], fit.bse["year_c"]
        p_nb = float(fit.pvalues["year_c"])
    irr = np.exp(10 * beta)
    lo, hi = np.exp(10 * (beta - 1.96 * se)), np.exp(10 * (beta + 1.96 * se))

    annual = d.groupby(year_col)[count_col].sum()
    ts = stats.theilslopes(annual.values, annual.index.values)
    tau, mk_p = stats.kendalltau(annual.index.values, annual.values)
    return Trend(irr, lo, hi, p_nb, ts.slope, ts.low_slope, ts.high_slope,
                 tau, mk_p, len(d))


def theil_only(values: pd.Series) -> dict:
    """Theil-Sen + Mann-Kendall for a year-indexed series (e.g. acres, p95 size)."""
    v = values.dropna()
    ts = stats.theilslopes(v.values, v.index.values)
    tau, p = stats.kendalltau(v.index.values, v.values)
    return {"theil_slope": float(ts.slope), "theil_lo": float(ts.low_slope),
            "theil_hi": float(ts.high_slope), "mk_tau": float(tau), "mk_p": float(p),
            "n_years": int(len(v))}


def season_span(df: pd.DataFrame, lo_pct: float = 10, hi_pct: float = 90) -> pd.DataFrame:
    """Per-year fire-season span: the doy_std window holding the middle 80% of fires."""
    g = df.groupby("fire_year")["doy_std"]
    out = pd.DataFrame({"doy_lo": g.quantile(lo_pct / 100), "doy_hi": g.quantile(hi_pct / 100),
                        "n": g.size()})
    out["span_days"] = out["doy_hi"] - out["doy_lo"]
    return out


def era_shift(series_early: pd.Series, series_late: pd.Series) -> dict:
    """Distribution shift between eras: medians + KS test."""
    a, b = series_early.dropna(), series_late.dropna()
    ks = stats.ks_2samp(a, b)
    return {"median_early": float(a.median()), "median_late": float(b.median()),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "n_early": int(len(a)), "n_late": int(len(b))}


# ---------------------------------------------------------------------------
# ML protocol helpers
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Train <=2014, val 2015-2017, test 2018-2020: the model never sees the Tubbs era."""
    y = df["fire_year"]
    return {"train": y <= wf.ML_TRAIN_MAX_YEAR,
            "val": y.between(*wf.ML_VAL_YEARS),
            "test": y.between(*wf.ML_TEST_YEARS)}


def spatial_blocks(df: pd.DataFrame, deg: float = 1.0) -> pd.Series:
    """1-degree grid block ids for GroupKFold -- kills coordinate memorization."""
    return (np.floor(df["lat"] / deg).astype("Int64").astype(str) + "_"
            + np.floor(df["lon"] / deg).astype("Int64").astype(str))


def climatology_baseline(train: pd.DataFrame, test: pd.DataFrame, label_col: str,
                         keys: tuple[str, ...] = ("county", "month")) -> pd.Series:
    """The groupby bar any model must clear: majority label per county x month."""
    lookup = (train.groupby(list(keys), observed=True)[label_col]
              .agg(lambda s: s.mode().iloc[0]))
    global_majority = train[label_col].mode().iloc[0]
    idx = pd.MultiIndex.from_frame(test[list(keys)].astype(lookup.index.dtypes.to_dict()
                                                           if hasattr(lookup.index, "dtypes") else None))
    pred = pd.Series(lookup.reindex(idx).values, index=test.index)
    return pred.fillna(global_majority)


def make_honest_clf(**kw):
    """HistGradientBoosting with native NaN + categorical handling; sklearn-only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    defaults = dict(categorical_features="from_dtype", class_weight="balanced",
                    max_iter=300, learning_rate=0.08, early_stopping=True,
                    validation_fraction=0.1, random_state=RNG)
    defaults.update(kw)
    return HistGradientBoostingClassifier(**defaults)


def museum_replica(n_estimators: int) -> RandomForestClassifier:
    """The 2020 configuration, faithfully: default depth, no class_weight, one
    hyperparameter touched. (random_state pinned so the museum piece is at
    least reproducible -- the original set none on the model.)"""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=RNG, n_jobs=-1)


def metrics_table(y_true, y_pred, proba_pos=None, labels: list | None = None) -> dict:
    """Everything the 2020 notebook did not compute."""
    out = {
        "n": int(len(y_true)),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_recall": {str(k): float(v) for k, v in
                             zip(labels or sorted(pd.unique(y_true)),
                                 recall_score(y_true, y_pred, average=None,
                                              labels=labels or sorted(pd.unique(y_true))))},
        "majority_share": float(pd.Series(y_true).value_counts(normalize=True).iloc[0]),
    }
    if proba_pos is not None:
        pos_label = labels[-1] if labels else sorted(pd.unique(y_true))[-1]
        y_bin = (np.asarray(y_true) == pos_label).astype(int)
        out["pr_auc"] = float(average_precision_score(y_bin, proba_pos))
        out["brier"] = float(brier_score_loss(y_bin, proba_pos))
    return out


def confusion(y_true, y_pred, labels: list) -> list[list[float]]:
    """Row-normalized confusion matrix as plain lists for JSON."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return [[round(float(x), 4) for x in row] for row in cm]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_irr(t: Trend) -> str:
    return f"IRR {t.irr_decade:.2f}/decade (95% CI {t.irr_lo:.2f}-{t.irr_hi:.2f}, MK p={t.mk_p:.3g})"


def fmt_pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


# ===========================================================================
# Round 3: the statistical toolkit. Round 1 shipped bare point estimates with
# no intervals, no paired tests, and no multiplicity control; every comparison
# was an eyeballed bar height. These are the instruments that fix that.
# ===========================================================================

def _block_members(blocks: np.ndarray | None) -> list[np.ndarray] | None:
    """Row indices per spatial block, computed ONCE. (Rebuilding this inside the
    resample loop turns a 20-second bootstrap into a 40-minute one; ask me how
    I know.)"""
    if blocks is None:
        return None
    order = np.argsort(blocks, kind="mergesort")
    sorted_blocks = np.asarray(blocks)[order]
    edges = np.flatnonzero(np.r_[True, sorted_blocks[1:] != sorted_blocks[:-1], True])
    return [order[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]


def _resample_idx(n: int, members: list[np.ndarray] | None,
                  rng: np.random.Generator) -> np.ndarray:
    """One bootstrap resample. With blocks, resample BLOCKS (spatially correlated
    rows travel together) -- a row bootstrap would understate uncertainty because
    nearby fires are not independent draws."""
    if members is None:
        return rng.integers(0, n, n)
    picked = rng.integers(0, len(members), len(members))
    return np.concatenate([members[i] for i in picked])


def bootstrap_ci(y_true, scores, metric_fn, n: int = 1000, blocks=None,
                 alpha: float = 0.05, seed: int = RNG) -> dict:
    """Percentile bootstrap CI for any metric_fn(y, s) -> float."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    members = _block_members(None if blocks is None else np.asarray(blocks))
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, s))
    vals = []
    for _ in range(n):
        idx = _resample_idx(len(y), members, rng)
        if len(np.unique(y[idx])) < 2:      # degenerate resample: skip
            continue
        vals.append(metric_fn(y[idx], s[idx]))
    vals = np.asarray(vals, dtype=float)
    return {"point": point,
            "lo": float(np.quantile(vals, alpha / 2)),
            "hi": float(np.quantile(vals, 1 - alpha / 2)),
            "se": float(vals.std(ddof=1)), "n_boot": int(len(vals))}


def paired_bootstrap_diff(y_true, scores_a, scores_b, metric_fn, n: int = 1000,
                          blocks=None, alpha: float = 0.05, seed: int = RNG) -> dict:
    """Difference metric(a) - metric(b) with a CI, using the SAME resample for
    both models (paired) so shared test-set noise cancels."""
    y = np.asarray(y_true)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    members = _block_members(None if blocks is None else np.asarray(blocks))
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, a) - metric_fn(y, b))
    diffs = []
    for _ in range(n):
        idx = _resample_idx(len(y), members, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx]))
    d = np.asarray(diffs, dtype=float)
    # add-one correction: (1+k)/(B+1) -- a resampling p-value can never be
    # exactly 0, only "< 1/(B+1)" (Round-4 red team; the FDR step was being
    # fed impossible p=0.0 values before this)
    B = len(d)
    p = 2 * min((1 + float((d <= 0).sum())) / (B + 1),
                (1 + float((d >= 0).sum())) / (B + 1))
    return {"diff": point, "lo": float(np.quantile(d, alpha / 2)),
            "hi": float(np.quantile(d, 1 - alpha / 2)),
            "p": min(1.0, p), "n_boot": int(B)}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar on the discordant pairs -- the right test for two
    classifiers scored on the SAME rows."""
    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b_only = int(np.sum(a_ok & ~b_ok))   # a right, b wrong
    c_only = int(np.sum(~a_ok & b_ok))   # b right, a wrong
    n = b_only + c_only
    p = float(stats.binomtest(b_only, n, 0.5).pvalue) if n else 1.0
    return {"a_right_b_wrong": b_only, "b_right_a_wrong": c_only,
            "n_discordant": n, "p": p}


def bh_fdr(pvals, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg. ~20 uncorrected p-values existed across the project
    before this; statsmodels was already installed and unused."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(list(pvals), dtype=float)
    if p.size == 0:
        return {"reject": [], "qvals": [], "alpha": alpha}
    rej, q, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return {"reject": [bool(x) for x in rej], "qvals": [float(x) for x in q],
            "alpha": alpha}


def ece(y_true, proba, bins: int = 15) -> float:
    """Expected calibration error over equal-count bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(total)


def calibration_slope_intercept(y_true, proba, eps: float = 1e-6) -> dict:
    """Cox calibration: regress the outcome on logit(p). Perfect calibration is
    slope 1, intercept 0. Slope < 1 means over-confident spread; a large negative
    intercept means systematically over-stated risk."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(proba, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    # C=inf is the unpenalised fit; penalty=None is deprecated as of sklearn 1.8
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit, np.asarray(y_true))
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def brier_skill_score(y_true, proba, reference: float | None = None) -> float:
    """BSS vs a constant base-rate forecast. Negative = worse than saying
    'every fire has the average risk'. This is the number that exposed Round 1's
    balanced-class-weight probabilities as uncalibrated (BSS = -3.76)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    ref = float(y.mean()) if reference is None else reference
    bs = float(np.mean((p - y) ** 2))
    bs_ref = float(np.mean((ref - y) ** 2))
    return float(1 - bs / bs_ref) if bs_ref > 0 else float("nan")


def precision_at_k(y_true, scores, k: float) -> float:
    """Precision among the top k fraction by score -- the capacity-constrained
    question: 'if we can only act on k% of ignitions, how often are we right?'"""
    y = np.asarray(y_true, dtype=int)
    n_take = max(1, int(round(len(y) * k)))
    top = np.argsort(-np.asarray(scores, dtype=float))[:n_take]
    return float(y[top].mean())


def recall_at_k(y_true, scores, k: float) -> float:
    """Share of all positives captured inside the top k fraction."""
    y = np.asarray(y_true, dtype=int)
    n_take = max(1, int(round(len(y) * k)))
    top = np.argsort(-np.asarray(scores, dtype=float))[:n_take]
    pos = y.sum()
    return float(y[top].sum() / pos) if pos else float("nan")


def topk_table(y_true, scores, ks=(0.005, 0.01, 0.02, 0.05)) -> list[dict]:
    """Top-k operating points, each against the ceiling a perfect ranker could
    reach with the same budget (budget < positives makes 100% recall impossible)."""
    y = np.asarray(y_true, dtype=int)
    n, pos = len(y), int(y.sum())
    out = []
    for k in ks:
        n_take = max(1, int(round(n * k)))
        prec = precision_at_k(y, scores, k)
        rec = recall_at_k(y, scores, k)
        out.append({"k": k, "n_selected": n_take,
                    "precision": prec, "recall": rec,
                    "lift": prec / (pos / n) if pos else float("nan"),
                    "ceiling_precision": min(1.0, pos / n_take),
                    "ceiling_recall": min(1.0, n_take / pos) if pos else float("nan")})
    return out


def net_benefit(y_true, proba, thresholds=None) -> list[dict]:
    """Decision-curve analysis. Net benefit = TP/n - FP/n * t/(1-t), where the
    threshold t encodes the cost ratio: acting on a fire that would not have
    escalated costs t/(1-t) as much as missing one that did."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    prev = y.mean()
    if thresholds is None:
        # log-spaced low tail down to 0.001: for wildfire the plausible cost
        # ratio puts the operating point at t ~ 0.001-0.005, which the original
        # 0.01-0.50 grid missed entirely (Round-4 red team)
        thresholds = np.unique(np.round(np.concatenate([
            np.geomspace(0.001, 0.01, 8), np.arange(0.01, 0.51, 0.01)]), 5))
    rows = []
    for t in thresholds:
        flag = p >= t
        tp = int(np.sum(flag & (y == 1)))
        fp = int(np.sum(flag & (y == 0)))
        w = t / (1 - t)
        rows.append({"threshold": float(t),
                     "net_benefit_model": float(tp / n - (fp / n) * w),
                     "net_benefit_treat_all": float(prev - (1 - prev) * w),
                     "n_flagged": int(flag.sum())})
    return rows


def nadeau_bengio_t(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test for CV fold scores. The naive paired t-test is
    anti-conservative because CV folds share training data; this inflates the
    variance by (1/k + n_test/n_train)."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    d = a - b
    k = len(d)
    if k < 2:
        return {"t": float("nan"), "p": float("nan"), "df": 0, "mean_diff": float(d.mean())}
    var = d.var(ddof=1)
    corrected_se = np.sqrt(var * (1 / k + n_test / max(n_train, 1)))
    t = float(d.mean() / corrected_se) if corrected_se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=k - 1)) if np.isfinite(t) else float("nan")
    return {"t": t, "p": p, "df": k - 1, "mean_diff": float(d.mean())}


class TestGate:
    """Pre-registration enforcement: the test era may be scored once per model
    family. Every touch is logged; the log is published in the article so the
    reader can audit multiplicity instead of taking our word for it."""

    def __init__(self):
        self.log: list[dict] = []

    def touch(self, family: str, note: str = "") -> None:
        from datetime import datetime, timezone
        if any(e["family"] == family for e in self.log):
            raise RuntimeError(f"TestGate: '{family}' already scored on the test era. "
                               f"Re-scoring would invalidate the pre-registered protocol.")
        self.log.append({"family": family, "note": note,
                         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def to_list(self) -> list[dict]:
        return list(self.log)
