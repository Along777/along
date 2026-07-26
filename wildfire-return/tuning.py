from __future__ import annotations

"""The three-track hyperparameter bake-off.

The question this answers is not "what are the best hyperparameters" but
"does the modern tuning stack actually beat a well-designed plain one, at equal
budget?" -- so all three tracks search the SAME space, under the SAME
spatial-block CV, scored by the SAME metric, with the SAME number of trials:

    A. sklearn      random search (the honest baseline everyone skips)
    B. optuna       TPE / Bayesian search over an identical space
    C. lightgbm     a different gradient-boosting library, tuned by optuna

Protocol guards baked in:
  * CV folds are SPATIAL BLOCKS inside the TRAIN era only. The test era is never
    touched here -- not once, not for early stopping, not for model selection.
  * early_stopping is OFF during search. With class_weight='balanced' it carves
    an unstratified internal split that makes trials both slower and noisier.
  * Everything runs single-threaded per fit on a 7 GB box; float32 matrices.

Usage:
    python tuning.py --probe              # report which stacks are importable
    python tuning.py --trials 30          # run the bake-off, write JSON
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

import wildfire as wf

SEARCH_FOLDS = 3          # 3 folds during search, 5 for the final re-score
DEFAULT_TRIALS = 30
WALL_CLOCK_GUARD_S = 1500  # per track


# ---------------------------------------------------------------------------
def probe_stacks() -> dict:
    """Import each optional stack and record what actually loaded. An install
    failure degrades the bake-off to fewer tracks and is reported as a finding."""
    out = {}
    for name in ("sklearn", "optuna", "lightgbm", "shap"):
        try:
            mod = __import__(name)
            out[name] = {"available": True, "version": getattr(mod, "__version__", "?")}
        except Exception as e:  # noqa: BLE001
            out[name] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    return out


# ---------------------------------------------------------------------------
# One shared search space, expressed once, sampled three ways.
# ---------------------------------------------------------------------------
def sample_hgb(rng: np.random.Generator) -> dict:
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        "max_leaf_nodes": int(rng.integers(15, 128)),
        "min_samples_leaf": int(rng.integers(5, 200)),
        "l2_regularization": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
        "max_features": float(rng.uniform(0.4, 1.0)),
        "max_iter": int(rng.integers(100, 400)),
    }


def optuna_hgb(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "max_iter": trial.suggest_int("max_iter", 100, 400),
    }


def optuna_lgbm(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
    }


def make_hgb(params: dict, balanced: bool = False):
    """class_weight defaults to NONE, unlike Round 1.

    Measured on the validation era at a fixed configuration: balancing changed
    average precision by -0.001 (noise), made ROC-AUC slightly worse
    (0.785 vs 0.802), took 60% longer to fit, and drove the Brier skill score
    from +0.02 to -2.56 by inflating mean predicted risk from 0.016 to 0.174 on
    a problem whose true rate is 0.021. For a rank-then-calibrate pipeline it is
    all cost and no benefit -- and it is the direct cause of the calibration
    claim Round 1 got wrong.
    """
    kw = dict(params)
    kw.update(categorical_features="from_dtype", early_stopping=False,
              random_state=wf_seed(), class_weight="balanced" if balanced else None)
    return HistGradientBoostingClassifier(**kw)


def make_lgbm(params: dict, balanced: bool = False):
    import lightgbm as lgb
    kw = dict(params)
    kw.update(objective="binary", n_jobs=1, verbose=-1, random_state=wf_seed(),
              class_weight="balanced" if balanced else None)
    return lgb.LGBMClassifier(**kw)


def wf_seed() -> int:
    return 42


# ---------------------------------------------------------------------------
def spatial_folds(df: pd.DataFrame, n_splits: int = SEARCH_FOLDS):
    """GroupKFold over 1-degree blocks: nearby fires cannot straddle folds, so a
    model cannot score well by memorising coordinates."""
    import models
    return GroupKFold(n_splits=n_splits), models.spatial_blocks(df)


def cv_average_precision(factory, params, X, y, groups, n_splits=SEARCH_FOLDS) -> tuple[float, list]:
    """Mean average-precision across spatial folds. AP is the right metric for a
    2%-prevalence ranking problem -- accuracy and ROC-AUC both flatter it."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, groups=groups):
        clf = factory(params)
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        scores.append(average_precision_score(y[te], p))
    return float(np.mean(scores)), [float(s) for s in scores]


# ---------------------------------------------------------------------------
def track_sklearn(X, y, groups, n_trials: int, seed: int = 42) -> dict:
    """Track A: plain random search. The control arm."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    trials, best = [], None
    for i in range(n_trials):
        if time.time() - t0 > WALL_CLOCK_GUARD_S:
            break
        params = sample_hgb(rng)
        score, folds = cv_average_precision(make_hgb, params, X, y, groups)
        trials.append({"trial": i, "params": params, "score": score,
                       "seconds": round(time.time() - t0, 1)})
        if best is None or score > best["score"]:
            best = {"params": params, "score": score, "folds": folds, "trial": i}
        print(f"    [A/sklearn] trial {i + 1}/{n_trials} AP={score:.4f} "
              f"(best {best['score']:.4f})", flush=True)
    return {"track": "sklearn_random", "trials": trials, "best": best,
            "n_trials": len(trials), "wall_clock_s": round(time.time() - t0, 1)}


def track_optuna(X, y, groups, n_trials: int, seed: int = 42) -> dict:
    """Track B: TPE over the identical space and budget."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    trials = []

    def objective(trial):
        if time.time() - t0 > WALL_CLOCK_GUARD_S:
            raise optuna.TrialPruned()
        params = optuna_hgb(trial)
        score, _ = cv_average_precision(make_hgb, params, X, y, groups)
        trials.append({"trial": trial.number, "params": params, "score": score,
                       "seconds": round(time.time() - t0, 1)})
        print(f"    [B/optuna]  trial {trial.number + 1}/{n_trials} AP={score:.4f}", flush=True)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    best_params = optuna_hgb_from_dict(study.best_params) if study.best_trial else None
    _, folds = (cv_average_precision(make_hgb, best_params, X, y, groups)
                if best_params else (None, []))
    return {"track": "optuna_tpe", "trials": trials,
            "best": {"params": best_params, "score": float(study.best_value),
                     "folds": folds, "trial": study.best_trial.number},
            "n_trials": len(trials), "wall_clock_s": round(time.time() - t0, 1)}


def optuna_hgb_from_dict(d: dict) -> dict:
    return {k: d[k] for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf",
                              "l2_regularization", "max_features", "max_iter") if k in d}


def track_lightgbm(X, y, groups, n_trials: int, seed: int = 42) -> dict:
    """Track C: a different GBM library, tuned the same way."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    trials = []

    def objective(trial):
        if time.time() - t0 > WALL_CLOCK_GUARD_S:
            raise optuna.TrialPruned()
        params = optuna_lgbm(trial)
        score, _ = cv_average_precision(make_lgbm, params, X, y, groups)
        trials.append({"trial": trial.number, "params": params, "score": score,
                       "seconds": round(time.time() - t0, 1)})
        print(f"    [C/lightgbm] trial {trial.number + 1}/{n_trials} AP={score:.4f}", flush=True)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    best = dict(study.best_params)
    best.setdefault("subsample_freq", 1)
    _, folds = cv_average_precision(make_lgbm, best, X, y, groups)
    return {"track": "lightgbm_optuna", "trials": trials,
            "best": {"params": best, "score": float(study.best_value), "folds": folds,
                     "trial": study.best_trial.number},
            "n_trials": len(trials), "wall_clock_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
def run_bakeoff(X, y, groups, n_trials: int = DEFAULT_TRIALS) -> dict:
    stacks = probe_stacks()
    print(f"[bakeoff] stacks: " + ", ".join(
        f"{k}={'ok' if v['available'] else 'MISSING'}" for k, v in stacks.items()))
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "stacks": stacks, "n_trials_requested": n_trials,
           "cv": {"folds": SEARCH_FOLDS, "grouping": "1-degree spatial blocks",
                  "scoring": "average_precision", "era": "train only (<=2014)"},
           "tracks": {}}

    print("  [A] sklearn random search")
    out["tracks"]["A_sklearn_random"] = track_sklearn(X, y, groups, n_trials)
    if stacks["optuna"]["available"]:
        print("  [B] optuna TPE")
        out["tracks"]["B_optuna_tpe"] = track_optuna(X, y, groups, n_trials)
    if stacks["optuna"]["available"] and stacks["lightgbm"]["available"]:
        print("  [C] lightgbm + optuna")
        out["tracks"]["C_lightgbm_optuna"] = track_lightgbm(X, y, groups, n_trials)

    # Reference configs, same CV. Two of them, so the effect of TUNING and the
    # effect of CLASS WEIGHTING can be read separately instead of confounded.
    print("  [ref] Round-1 default config")
    default = {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31,
               "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0}
    ref_score, ref_folds = cv_average_precision(
        lambda p: make_hgb(p, balanced=True), default, X, y, groups)
    out["default_reference"] = {"params": default, "score": ref_score, "folds": ref_folds,
                                "note": "exactly what Round 1 shipped, incl. class_weight=balanced"}
    unw_score, unw_folds = cv_average_precision(make_hgb, default, X, y, groups)
    out["default_unweighted"] = {"params": default, "score": unw_score, "folds": unw_folds,
                                 "note": "same config, class weighting removed"}
    print(f"       default (balanced) AP={ref_score:.4f}; unweighted AP={unw_score:.4f}")

    # How much did the choice of search STRATEGY matter? If the spread across
    # tracks is smaller than the gap to the default config, the honest headline
    # is "tuning mattered, the tuner did not."
    bests = [t["best"]["score"] for t in out["tracks"].values() if t.get("best")]
    if bests:
        out["spread"] = float(max(bests) - min(bests))
        out["gain_over_default"] = float(max(bests) - ref_score)
        print(f"       track spread={out['spread']:.5f}  "
              f"gain over default={out['gain_over_default']:+.5f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="three-track hyperparameter bake-off")
    ap.add_argument("--probe", action="store_true", help="report stack availability and exit")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--state", default="CA")
    ap.add_argument("--threshold", type=float, default=100.0)
    args = ap.parse_args()

    if args.probe:
        for k, v in probe_stacks().items():
            print(f"{k:10s} {'OK  ' + v['version'] if v['available'] else 'MISSING: ' + v['error']}")
        return

    import features as fx
    import models
    df = wf.load_state_fires(args.state)
    df = fx.add_features(df)
    tr = df[models.temporal_split(df)["train"]]
    X = fx.build_matrix(tr, "R6_history")
    y = (tr["fire_size"] >= args.threshold).to_numpy(dtype=int)
    groups = models.spatial_blocks(tr)
    print(f"[bakeoff] {args.state} train era: {len(X):,} rows x {X.shape[1]} features, "
          f"{y.sum():,} positives ({y.mean():.2%})")
    res = run_bakeoff(X, y, groups, args.trials)
    (wf.DATA / "tuning_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[done] data/tuning_results.json")


if __name__ == "__main__":
    main()
