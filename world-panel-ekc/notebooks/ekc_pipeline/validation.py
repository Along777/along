"""
validation.py
=============
Out-of-sample validation, done honestly. Two tasks on a temporal holdout
(train year <= SPLIT_YEAR, test after):

  1. LEVELS (persistence check). Predicting the *level* of CO2 per capita is
     trivial: the series is a near-random-walk (within-country autocorrelation
     ~0.996), so a naive "carry last training value forward" baseline already
     scores R^2 ~ 0.98. We report that baseline as the reference row -- any
     model that only ties it has added nothing. Levels are NOT where a model
     earns its keep.

  2. CHANGES (the real test). Predicting the within-country annual *change*
     in CO2 per capita from contemporaneous first-differenced drivers. The
     naive "predict zero change" baseline scores ~0 here, so this is a genuine
     signal-extraction problem -- and it is where the parsimonious structural
     (differenced-OLS) model and the flexible gradient-boosting benchmark
     actually diverge.

Fairness of the ML benchmark:
  - the GBM is given RAW features only (not the engineered square) and is free
    to find curvature itself;
  - its hyperparameters are chosen on an inner temporal fold (no leakage);
  - importances are permutation importances on the held-out set, not the biased
    impurity importances.

Every model in a task is scored on the *identical* set of held-out rows.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from . import config as C
from . import data as D


# structural (econometric) model keeps the quadratic; the ML benchmark does not
STRUCT_FEATURES = [C.GDP, C.GDP_SQ, C.MODERATOR] + C.CONTROLS
GBM_FEATURES = [C.GDP, C.MODERATOR] + C.CONTROLS      # raw; trees find curvature
# drivers first-differenced for the change task
CHANGE_DRIVERS = [C.GDP, "renew_share", "fossil_fuel_pct", "urban_pct",
                  "trade_pct_gdp", "dependency_ratio", C.MODERATOR]


def _rmse_r2(y, yhat):
    return (float(np.sqrt(mean_squared_error(y, yhat))), float(r2_score(y, yhat)))


def _prep(df: pd.DataFrame):
    full = D.build_sample(df, "full")
    keep = ["iso3", "year", "income_group", C.TARGET] + STRUCT_FEATURES
    return full.dropna(subset=[c for c in keep if c in full.columns])[keep].copy()


# --------------------------------------------------------------------------- #
# GBM hyperparameter selection on an inner temporal fold                        #
# --------------------------------------------------------------------------- #

def _tune_gbm(Xtr, ytr, years):
    """Pick GBM params on an inner temporal split (fit <= TUNE_YEAR, score the
    gap up to SPLIT_YEAR). Falls back to defaults if the inner fold is empty."""
    years = np.asarray(years)
    inner_tr = years <= C.TUNE_YEAR
    inner_va = years > C.TUNE_YEAR
    if inner_tr.sum() < 50 or inner_va.sum() < 20:
        return dict(C.GBM_PARAMS)
    Xtr = Xtr.reset_index(drop=True)
    ytr = np.asarray(ytr)
    best, best_rmse = None, np.inf
    keys = list(C.GBM_GRID)
    for combo in itertools.product(*(C.GBM_GRID[k] for k in keys)):
        params = dict(C.GBM_PARAMS)
        params.update(dict(zip(keys, combo)))
        m = GradientBoostingRegressor(**params)
        m.fit(Xtr[inner_tr], ytr[inner_tr])
        pred = m.predict(Xtr[inner_va])
        rmse = np.sqrt(mean_squared_error(ytr[inner_va], pred))
        if rmse < best_rmse:
            best, best_rmse = params, rmse
    return best


# --------------------------------------------------------------------------- #
# task 1: levels                                                                #
# --------------------------------------------------------------------------- #

def _level_task(data: pd.DataFrame):
    train, test = D.temporal_split(data)
    # every model scores on the SAME rows: test countries seen in training
    te = test[test["iso3"].isin(train["iso3"].unique())].copy()
    y_te = te[C.TARGET].values

    # naive persistence: carry each country's last training-year value forward
    last_tr = train.sort_values("year").groupby("iso3")[C.TARGET].last()
    pred_naive = te["iso3"].map(last_tr).values

    # structural: country-FE OLS in log1p space (quadratic + controls + dummies)
    Xtr_d = pd.get_dummies(train[STRUCT_FEATURES + ["iso3"]], columns=["iso3"],
                           drop_first=True, dtype=float)
    Xte_d = pd.get_dummies(te[STRUCT_FEATURES + ["iso3"]], columns=["iso3"],
                           drop_first=True, dtype=float).reindex(
                               columns=Xtr_d.columns, fill_value=0.0)
    Xtr = sm.add_constant(Xtr_d, has_constant="add")
    Xte = sm.add_constant(Xte_d, has_constant="add")
    ols = sm.OLS(np.log1p(train[C.TARGET].values), Xtr).fit()
    pred_struct = np.expm1(np.asarray(ols.predict(Xte)))

    # GBM benchmark: raw features + country/income dummies, tuned, log1p target
    cat = pd.concat([train, te])[["iso3", "income_group"]]
    dum = pd.get_dummies(cat, columns=["iso3", "income_group"], dtype=float)
    Xtr_g = pd.concat([train[GBM_FEATURES].reset_index(drop=True),
                       dum.loc[train.index].reset_index(drop=True)], axis=1)
    Xte_g = pd.concat([te[GBM_FEATURES].reset_index(drop=True),
                       dum.loc[te.index].reset_index(drop=True)], axis=1)
    ytr_g = np.log1p(train[C.TARGET].values)
    best = _tune_gbm(Xtr_g, ytr_g, train["year"].values)
    gbm = GradientBoostingRegressor(**best).fit(Xtr_g, ytr_g)
    pred_gbm = np.expm1(gbm.predict(Xte_g))

    rows = []
    for name, pred in [("naive_persistence", pred_naive),
                       ("structural_country_FE_OLS", pred_struct),
                       ("gradient_boosting", pred_gbm)]:
        rmse, r2 = _rmse_r2(y_te, pred)
        rows.append({"task": "levels", "model": name,
                     "test_rmse": round(rmse, 3), "test_r2": round(r2, 3),
                     "n_test": len(te)})
    metrics = pd.DataFrame(rows)

    imp = _perm_importance(gbm, Xte_g, np.log1p(y_te), collapse=True)

    preds = te[["iso3", "year", "income_group", C.TARGET]].copy()
    preds["pred_struct"] = pred_struct
    preds["pred_gbm"] = pred_gbm
    preds["pred_naive"] = pred_naive
    return metrics, imp, preds, best


# --------------------------------------------------------------------------- #
# task 2: changes (the real test)                                               #
# --------------------------------------------------------------------------- #

def _change_task(data: pd.DataFrame):
    d = data.sort_values(["iso3", "year"]).copy()
    d["d_co2"] = d.groupby("iso3")[C.TARGET].diff()
    dcols = []
    for c in CHANGE_DRIVERS:
        dc = f"d_{c}"
        d[dc] = d.groupby("iso3")[c].diff()
        dcols.append(dc)
    d = d.dropna(subset=["d_co2"] + dcols)
    train, test = D.temporal_split(d)
    if len(test) < 20:
        return None, None, None

    y_te = test["d_co2"].values
    # naive: predict zero change
    pred_naive = np.zeros(len(test))
    # structural: OLS on first differences
    ols = sm.OLS(train["d_co2"].values,
                 sm.add_constant(train[dcols])).fit()
    pred_ols = np.asarray(ols.predict(sm.add_constant(test[dcols])))
    # GBM on first differences (tuned on inner temporal fold)
    best = _tune_gbm(train[dcols].reset_index(drop=True),
                     train["d_co2"].values, train["year"].values)
    gbm = GradientBoostingRegressor(**best).fit(train[dcols], train["d_co2"].values)
    pred_gbm = gbm.predict(test[dcols])

    rows = []
    for name, pred in [("naive_zero_change", pred_naive),
                       ("structural_diff_OLS", pred_ols),
                       ("gradient_boosting_diff", pred_gbm)]:
        rmse, r2 = _rmse_r2(y_te, pred)
        rows.append({"task": "changes", "model": name,
                     "test_rmse": round(rmse, 4), "test_r2": round(r2, 4),
                     "n_test": len(test)})
    metrics = pd.DataFrame(rows)

    imp = _perm_importance(gbm, test[dcols], y_te, collapse=False)

    preds = test[["iso3", "year", "income_group", "d_co2"]].copy()
    preds["pred_ols"] = pred_ols
    preds["pred_gbm"] = pred_gbm
    return metrics, imp, preds


# --------------------------------------------------------------------------- #
# permutation importance                                                        #
# --------------------------------------------------------------------------- #

def _perm_importance(model, X, y, collapse):
    r = permutation_importance(model, X, y, n_repeats=C.PERM_IMPORTANCE_REPS,
                               random_state=C.BOOTSTRAP_SEED,
                               scoring="neg_root_mean_squared_error")
    imp = pd.Series(r.importances_mean, index=X.columns)
    if collapse:
        iso = imp[[i for i in imp.index if i.startswith("iso3_")]].sum()
        inc = imp[[i for i in imp.index if i.startswith("income_group_")]].sum()
        keep = imp[[i for i in imp.index
                    if not i.startswith("iso3_") and not i.startswith("income_group_")]]
        imp = pd.concat([keep, pd.Series({"country_fixed_effects": iso,
                                          "income_group": inc})])
    imp = imp.sort_values(ascending=False)
    return imp.rename("perm_importance").reset_index().rename(columns={"index": "feature"})


# --------------------------------------------------------------------------- #
# orchestrator                                                                  #
# --------------------------------------------------------------------------- #

def run(df: pd.DataFrame):
    data = _prep(df)

    lvl_metrics, lvl_imp, lvl_preds, gbm_best = _level_task(data)
    lvl_metrics.to_csv(C.OUT_DIR / "validation_metrics.csv", index=False)
    lvl_imp.to_csv(C.OUT_DIR / "permutation_importance.csv", index=False)
    pd.DataFrame([{"param": k, "value": v} for k, v in gbm_best.items()]).to_csv(
        C.OUT_DIR / "gbm_tuned_params.csv", index=False)

    chg_metrics, chg_imp, chg_preds = _change_task(data)
    if chg_metrics is not None:
        chg_metrics.to_csv(C.OUT_DIR / "validation_change_metrics.csv", index=False)
        chg_imp.to_csv(C.OUT_DIR / "change_importance.csv", index=False)

    return {
        "level_metrics": lvl_metrics, "level_imp": lvl_imp, "level_preds": lvl_preds,
        "gbm_best": gbm_best,
        "change_metrics": chg_metrics, "change_imp": chg_imp, "change_preds": chg_preds,
    }
