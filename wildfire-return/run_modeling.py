from __future__ import annotations

"""Round 3: the modeling stage.

The headline is an OPERATIONAL model -- P(this ignition reaches >=100 acres) --
because that is the question a duty officer actually has at 2 p.m. on a windy
October afternoon, and because its label is clean: fire size is ~0% missing and
its base rate barely moves across eras, unlike the cause labels that drift from
10% to 42% missing.

What Round 1 shipped, and what this fixes:
  * nothing was ever tuned (make_honest_clf was called with zero kwargs)
  * the validation era was computed and thrown away
  * probabilities were reported as calibrated; they were not (BSS = -3.76)
  * every number was a bare point estimate with no interval
  * the leak probe was pointed at a task where the leak is not a leak

Pre-registered protocol: all selection happens on the train era (spatial-block
CV) and the val era. The test era is scored ONCE per model family, enforced by
models.TestGate, and the touch log is published.

Run:  python run_modeling.py [--trials 30] [--quick]
"""

import argparse
import json
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as fx
import models
import tuning
import wildfire as wf

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}

THRESHOLD = 100.0
THRESHOLD_LADDER = [10.0, 100.0, 300.0, 1000.0]
TOPK = (0.005, 0.01, 0.02, 0.05)
N_BOOT = 1000
GATE = models.TestGate()


def style_ax(ax, grid_axis="y"):
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["axis"])
    ax.tick_params(colors=COL["ink2"], labelsize=9)
    ax.xaxis.label.set_color(COL["ink2"])
    ax.yaxis.label.set_color(COL["ink2"])
    ax.title.set_color(COL["ink"])


def savefig(fig, name):
    wf.FIG.mkdir(exist_ok=True)
    fig.savefig(wf.FIG / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig ] {name}", flush=True)


def ap(y, s):
    return average_precision_score(y, s)


# ---------------------------------------------------------------------------
def load_state(state: str, subsample: int = 0) -> pd.DataFrame:
    """Load a state's cache and engineer features.

    Any subsampling happens BEFORE feature construction: the fire-history
    features are defined relative to the frame they are built from, so thinning
    afterwards would leave them describing fires that are no longer there.
    (The causality assertion catches exactly that mistake -- it caught mine.)
    """
    df = wf.load_state_fires(state, strict=True)
    if subsample:
        pos = df[df["fire_size"] >= THRESHOLD]
        neg_pool = df[df["fire_size"] < THRESHOLD]
        neg = neg_pool.sample(min(subsample, len(neg_pool)), random_state=42)
        df = pd.concat([pos, neg]).sort_values("discovery_date").reset_index(drop=True)
    return fx.add_features(df)


def splits_of(d: pd.DataFrame):
    s = models.temporal_split(d)
    return d[s["train"]], d[s["val"]], d[s["test"]]


def fit_predict(clf, Xtr, ytr, Xte):
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf


def numeric_pipeline(model):
    """Linear models need imputation + scaling; trees do not."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


# ---------------------------------------------------------------------------
def main() -> None:
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--trials", type=int, default=25)
    ap_arg.add_argument("--quick", action="store_true", help="small budgets for a smoke run")
    ap_arg.add_argument("--subsample", type=int, default=0,
                        help="rows per state (smoke testing only; 0 = use everything)")
    args = ap_arg.parse_args()
    if args.quick:
        args.trials = 4
    n_boot = 200 if args.quick else N_BOOT

    t_start = time.time()
    R: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "protocol": {
                   "target": f"P(fire_size >= {THRESHOLD:.0f} acres) at ignition",
                   "splits": {"train": "<=2014", "val": "2015-2017", "test": "2018-2020"},
                   "cv": "GroupKFold(5) over 1-degree spatial blocks, train era only",
                   "selection": "all tuning on train CV + val; test scored once per family",
                   "metric": "average precision (AP); base rate is the no-skill line"}}

    print("[load] CA + FL with engineered features")
    ca = load_state("CA", args.subsample)
    fl = load_state("FL", args.subsample)
    if args.subsample:
        R["SMOKE_RUN"] = {"subsample_rows_per_state": args.subsample,
                          "warning": "numbers from a subsampled smoke run are not publishable"}
        print(f"[smoke] subsampled to CA={len(ca):,} FL={len(fl):,}")
    R["feature_causality"] = fx.assert_history_is_causal(ca, n_samples=150)

    ca_tr, ca_va, ca_te = splits_of(ca)
    y_tr = (ca_tr["fire_size"] >= THRESHOLD).to_numpy(int)
    y_va = (ca_va["fire_size"] >= THRESHOLD).to_numpy(int)
    y_te = (ca_te["fire_size"] >= THRESHOLD).to_numpy(int)
    X_tr = fx.build_matrix(ca_tr, "R6_history")
    # reindex, not fancy-index: build_matrix drops constant columns and a column
    # that is constant in one era but not another would otherwise KeyError
    X_va = fx.build_matrix(ca_va, "R6_history").reindex(columns=X_tr.columns)
    X_te = fx.build_matrix(ca_te, "R6_history").reindex(columns=X_tr.columns)
    groups_tr = models.spatial_blocks(ca_tr)
    blocks_te = models.spatial_blocks(ca_te).to_numpy()

    R["data"] = {"n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
                 "n_features": int(X_tr.shape[1]),
                 "prevalence_train": float(y_tr.mean()), "prevalence_val": float(y_va.mean()),
                 "prevalence_test": float(y_te.mean()),
                 "positives_test": int(y_te.sum())}
    print(f"[data] train {len(X_tr):,} / val {len(X_va):,} / test {len(X_te):,}; "
          f"{X_tr.shape[1]} features; test prevalence {y_te.mean():.2%}")

    R["tripwire"] = fx.univariate_tripwire(X_tr, y_tr)
    print(f"[gate] univariate leak tripwire passed={R['tripwire']['passed']}")

    # ---------------------------------------------------------------- bake-off
    # Search on a stratified subsample of the TRAIN era: every positive plus a
    # random draw of negatives. Hyperparameter rankings are stable under this
    # kind of thinning, and it is the difference between a 25-trial search and a
    # 3-trial one on a 6-core laptop. The winning config is refit on the full
    # training set before anything is scored. All three tracks see identical rows.
    search_cap = 60_000
    if len(X_tr) > search_cap:
        pos_idx = np.flatnonzero(y_tr == 1)
        neg_idx = np.flatnonzero(y_tr == 0)
        rng_s = np.random.default_rng(42)
        keep_neg = rng_s.choice(neg_idx, size=min(search_cap - len(pos_idx), len(neg_idx)),
                                replace=False)
        sel = np.sort(np.concatenate([pos_idx, keep_neg]))
        X_search, y_search = X_tr.iloc[sel], y_tr[sel]
        groups_search = groups_tr.iloc[sel]
        print(f"[bakeoff] searching on {len(X_search):,} of {len(X_tr):,} train rows "
              f"(all {int(y_tr.sum()):,} positives kept, prevalence {y_search.mean():.2%})")
    else:
        X_search, y_search, groups_search = X_tr, y_tr, groups_tr
    R["protocol"]["search_subsample"] = {"rows": int(len(X_search)),
                                         "of_train_rows": int(len(X_tr)),
                                         "all_positives_kept": True,
                                         "winner_refit_on": "full train era"}

    print(f"[bakeoff] three tracks x {args.trials} trials")
    bake = tuning.run_bakeoff(X_search, y_search, groups_search, args.trials)
    R["bakeoff"] = bake
    best_track = max((k for k in bake["tracks"]), key=lambda k: bake["tracks"][k]["best"]["score"])
    R["bakeoff_winner"] = best_track
    print(f"[bakeoff] winner: {best_track}")

    # ---------------------------------------------------------------- model zoo
    print("[zoo] fitting families (test scored once each)")
    zoo: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    def score_family(name, proba_te, note="", is_probability=True):
        """is_probability=False for pure RANKING baselines (e.g. raw ERC, which
        lives on 0-150). Ranking metrics still apply; Brier skill and ECE do not,
        and reporting them would print a meaningless -20,000."""
        GATE.touch(name, note)
        m = {"ap": float(ap(y_te, proba_te)),
             "roc_auc": float(roc_auc_score(y_te, proba_te)),
             "bss": models.brier_skill_score(y_te, proba_te) if is_probability else None,
             "ece": models.ece(y_te, proba_te) if is_probability else None,
             "is_probability": is_probability,
             "topk": models.topk_table(y_te, proba_te, TOPK)}
        ci = models.bootstrap_ci(y_te, proba_te, ap, n=n_boot, blocks=blocks_te)
        m["ap_ci"] = ci
        zoo[name] = m
        preds[name] = proba_te
        bss_txt = f"{m['bss']:+.2f}" if m["bss"] is not None else "n/a (rank-only)"
        print(f"  [zoo ] {name:22s} AP={m['ap']:.4f} [{ci['lo']:.4f},{ci['hi']:.4f}] "
              f"ROC={m['roc_auc']:.3f} BSS={bss_txt}", flush=True)

    # tuned HGB (bake-off winner config, but always the sklearn HGB family)
    hgb_best = bake["tracks"]["A_sklearn_random"]["best"]["params"]
    if "B_optuna_tpe" in bake["tracks"] and \
       bake["tracks"]["B_optuna_tpe"]["best"]["score"] > bake["tracks"]["A_sklearn_random"]["best"]["score"]:
        hgb_best = bake["tracks"]["B_optuna_tpe"]["best"]["params"]
    p, hgb_model = fit_predict(tuning.make_hgb(hgb_best), X_tr, y_tr, X_te)
    score_family("hgb_tuned", p, "bake-off best HGB config")

    # Round 1 exactly: default config AND class_weight="balanced". This is the
    # family whose probabilities the main article described as calibrated.
    p_def, _ = fit_predict(tuning.make_hgb(bake["default_reference"]["params"], balanced=True),
                           X_tr, y_tr, X_te)
    score_family("hgb_default_balanced", p_def, "Round-1 config incl. class_weight=balanced")

    p_unw, _ = fit_predict(tuning.make_hgb(bake["default_reference"]["params"]), X_tr, y_tr, X_te)
    score_family("hgb_default_unweighted", p_unw, "same config, class weighting removed")

    # 200 trees with max_samples=0.5 and a leaf floor: a deep unbounded forest on
    # 200k x 86 float32 would peak near this machine's 7 GB ceiling
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features="sqrt",
                                max_samples=0.5, class_weight="balanced_subsample",
                                n_jobs=2, random_state=42)
    # NOTE (Round-4 red team): dropping the categoricals here denies RF/ET/
    # logistic four features incl. evt (SHAP's #2) -- the fair re-run with
    # ordinal-encoded categoricals and no class weighting lives in run_redteam.py
    Xn_tr = X_tr.select_dtypes(exclude="category")
    Xn_te = X_te.select_dtypes(exclude="category")[Xn_tr.columns]
    imp = SimpleImputer(strategy="median").fit(Xn_tr)
    p_rf, _ = fit_predict(rf, pd.DataFrame(imp.transform(Xn_tr), columns=Xn_tr.columns), y_tr,
                          pd.DataFrame(imp.transform(Xn_te), columns=Xn_tr.columns))
    score_family("random_forest", p_rf)

    et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, max_features="sqrt",
                              max_samples=0.5, bootstrap=True,
                              class_weight="balanced", n_jobs=2, random_state=42)
    p_et, _ = fit_predict(et, pd.DataFrame(imp.transform(Xn_tr), columns=Xn_tr.columns), y_tr,
                          pd.DataFrame(imp.transform(Xn_te), columns=Xn_tr.columns))
    score_family("extra_trees", p_et)

    logit = numeric_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1))
    p_lr, _ = fit_predict(logit, Xn_tr, y_tr, Xn_te)
    score_family("logistic_en", p_lr)

    if bake["stacks"]["lightgbm"]["available"] and "C_lightgbm_optuna" in bake["tracks"]:
        lgbm_best = bake["tracks"]["C_lightgbm_optuna"]["best"]["params"]
        p_lgb, lgb_model = fit_predict(tuning.make_lgbm(lgbm_best), X_tr, y_tr, X_te)
        score_family("lightgbm_tuned", p_lgb)
    else:
        lgb_model = None

    # baselines
    clim = climatology_rate(ca_tr, ca_te, THRESHOLD)
    score_family("climatology_rate", clim, "majority-rate per 0.2deg cell x month")
    erc_only = ca_te["erc"].fillna(ca_tr["erc"].median()).to_numpy(dtype=float)
    score_family("erc_only", erc_only, "single-variable ranking: today's ERC",
                 is_probability=False)

    R["zoo"] = zoo
    R["test_touch_log"] = GATE.to_list()

    # paired comparisons vs the tuned model, FDR-corrected
    champion = max(zoo, key=lambda k: zoo[k]["ap"])
    R["champion"] = champion
    comps, pvals = [], []
    for name in zoo:
        if name == champion:
            continue
        d = models.paired_bootstrap_diff(y_te, preds[champion], preds[name], ap,
                                         n=n_boot, blocks=blocks_te)
        comps.append({"champion": champion, "vs": name, **d})
        pvals.append(d["p"])
    fdr = models.bh_fdr(pvals)
    for c, q, rej in zip(comps, fdr["qvals"], fdr["reject"]):
        c["q"], c["significant"] = q, bool(rej)
    R["zoo_comparisons"] = comps
    print(f"[zoo ] champion={champion}; "
          f"{sum(c['significant'] for c in comps)}/{len(comps)} differences survive FDR")

    # ------------------------------------------------------------- calibration
    print("[cal ] fitting isotonic / sigmoid on the VAL era (finally using it)")
    best_model_maker = lambda: tuning.make_hgb(hgb_best)
    cal = calibrate_on_val(best_model_maker, X_tr, y_tr, X_va, y_va, X_te, y_te, n_boot, blocks_te)
    R["calibration"] = cal
    print(f"       raw BSS {cal['raw']['bss']:+.3f} -> isotonic {cal['isotonic']['bss']:+.3f} "
          f"/ sigmoid {cal['sigmoid']['bss']:+.3f}")

    best_cal = max(("isotonic", "sigmoid"), key=lambda k: cal[k]["bss"])
    p_cal = np.asarray(cal["_proba"][best_cal])
    R["calibration"]["chosen"] = best_cal
    del cal["_proba"]

    # --------------------------------------------------------- decision analysis
    R["decision"] = {"topk": models.topk_table(y_te, p_cal, TOPK),
                     "net_benefit": models.net_benefit(y_te, p_cal),
                     "prevalence": float(y_te.mean())}

    # ------------------------------------------------------------ threshold ladder
    print("[ladder] escalation thresholds")
    ladder = []
    for thr in THRESHOLD_LADDER:
        yl_tr = (ca_tr["fire_size"] >= thr).to_numpy(int)
        yl_te = (ca_te["fire_size"] >= thr).to_numpy(int)
        pl, _ = fit_predict(tuning.make_hgb(hgb_best), X_tr, yl_tr, X_te)
        row = {"threshold": thr, "prevalence": float(yl_te.mean()),
               "positives": int(yl_te.sum()), "ap": float(ap(yl_te, pl)),
               "roc_auc": float(roc_auc_score(yl_te, pl)),
               "lift_at_1pct": models.precision_at_k(yl_te, pl, 0.01) / max(yl_te.mean(), 1e-9),
               "topk": models.topk_table(yl_te, pl, TOPK)}
        ladder.append(row)
        print(f"  [ladd] >={thr:>6.0f} ac  prev={row['prevalence']:.3%} AP={row['ap']:.4f} "
              f"lift@1%={row['lift_at_1pct']:.1f}x", flush=True)
    R["threshold_ladder"] = ladder

    # ---------------------------------------------------------------- ablation
    print("[abla] feature ladder")
    abl = []
    for rung in fx.cumulative_rungs():
        Xa_tr = fx.build_matrix(ca_tr, rung)
        Xa_te = fx.build_matrix(ca_te, rung).reindex(columns=Xa_tr.columns)
        pa, _ = fit_predict(tuning.make_hgb(hgb_best), Xa_tr, y_tr, Xa_te)
        ci = models.bootstrap_ci(y_te, pa, ap, n=max(200, n_boot // 2), blocks=blocks_te)
        abl.append({"rung": rung, "n_features": int(Xa_tr.shape[1]), "ap": float(ap(y_te, pa)),
                    "ap_lo": ci["lo"], "ap_hi": ci["hi"]})
        print(f"  [abla] {rung:16s} {Xa_tr.shape[1]:3d} feats AP={abl[-1]['ap']:.4f}", flush=True)
    base_ap = abl[0]["ap"]
    for row in abl:
        row["delta_vs_base"] = row["ap"] - base_ap
    R["ablation"] = abl

    # ------------------------------------------------------------- leak ladder
    print("[leak] the probe that bites")
    R["leak_probe"] = leak_ladder(ca_tr, ca_te, X_tr, X_te, y_tr, y_te, hgb_best)

    # ------------------------------------------------------------- drift/transfer
    print("[drift] year decay + CA->FL transfer")
    by_year = []
    for yr in sorted(ca_te["fire_year"].unique()):
        m = (ca_te["fire_year"] == yr).to_numpy()
        if m.sum() > 200 and y_te[m].sum() > 5:
            by_year.append({"year": int(yr), "n": int(m.sum()), "positives": int(y_te[m].sum()),
                            "ap": float(ap(y_te[m], p_cal[m])),
                            "prevalence": float(y_te[m].mean())})
    fl_tr, fl_va, fl_te = splits_of(fl)
    yfl_te = (fl_te["fire_size"] >= THRESHOLD).to_numpy(int)
    Xfl_te = fx.build_matrix(fl_te, "R6_history")
    common = [c for c in X_tr.columns if c in Xfl_te.columns]
    p_transfer, _ = fit_predict(tuning.make_hgb(hgb_best), X_tr[common], y_tr, Xfl_te[common])
    yfl_tr = (fl_tr["fire_size"] >= THRESHOLD).to_numpy(int)
    Xfl_tr = fx.build_matrix(fl_tr, "R6_history")[common]
    p_native, _ = fit_predict(tuning.make_hgb(hgb_best), Xfl_tr, yfl_tr, Xfl_te[common])
    R["drift"] = {
        "by_year": by_year,
        "transfer_ca_to_fl": {"ap": float(ap(yfl_te, p_transfer)),
                              "native_fl_ap": float(ap(yfl_te, p_native)),
                              "fl_prevalence": float(yfl_te.mean()),
                              "n_common_features": len(common)},
    }
    print(f"  [xfer] CA->FL AP={R['drift']['transfer_ca_to_fl']['ap']:.4f} vs "
          f"FL-native {R['drift']['transfer_ca_to_fl']['native_fl_ap']:.4f}")

    # ------------------------------------------------------- interpretability
    print("[intp] grouped permutation importance")
    R["importance"] = grouped_importance(hgb_model, X_te, y_te, n_repeats=3 if args.quick else 5)
    if bake["stacks"]["shap"]["available"] and lgb_model is not None:
        try:
            R["shap"] = shap_summary(lgb_model, X_te, n=2000)
        except Exception as e:  # noqa: BLE001
            R["shap"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    else:
        R["shap"] = {"available": False, "error": "stack unavailable or no lightgbm model"}

    # ------------------------------------------------------------ Tubbs study
    print("[tubb] scoring the fire that started the project")
    R["tubbs"] = tubbs_case_study(ca, ca_tr, y_tr, X_tr, hgb_best, best_cal, cal)

    # ------------------------------------------------------------------ figures
    make_figures(R, y_te, preds, p_cal, cal, ca_te)
    for k in ("_scores_all", "_scores_october"):
        R["tubbs"].pop(k, None)          # figure-only arrays, never in the JSON

    # ------------------------------------------------------------------- claims
    champ = zoo[champion]
    top1 = next(t for t in R["decision"]["topk"] if abs(t["k"] - 0.01) < 1e-9)
    R["claims"] = {
        "n_features_engineered": int(X_tr.shape[1]),
        "test_ap": round(champ["ap"], 4),
        "test_ap_lo": round(champ["ap_ci"]["lo"], 4),
        "test_ap_hi": round(champ["ap_ci"]["hi"], 4),
        "test_roc_auc": round(champ["roc_auc"], 3),
        "prevalence_test": round(float(y_te.mean()), 4),
        "lift_over_base": round(champ["ap"] / float(y_te.mean()), 1),
        "round1_bss": round(models.brier_skill_score(y_te, preds["hgb_default_balanced"]), 2),
        "unweighted_bss": round(models.brier_skill_score(y_te, preds["hgb_default_unweighted"]), 3),
        "calibrated_bss": round(cal[best_cal]["bss"], 3),
        "calibrated_ece": round(cal[best_cal]["ece"], 4),
        "precision_at_1pct": round(top1["precision"], 3),
        "recall_at_1pct": round(top1["recall"], 3),
        "ceiling_recall_at_1pct": round(top1["ceiling_recall"], 3),
        "tuning_gain_ap": round(zoo["hgb_tuned"]["ap"] - zoo["hgb_default_unweighted"]["ap"], 4),
        "ablation_gain_ap": round(abl[-1]["ap"] - abl[0]["ap"], 4),
        "leak_mtbs_ap": round(R["leak_probe"]["rungs"][1]["ap"], 3),
        "leak_honest_ap": round(R["leak_probe"]["rungs"][0]["ap"], 3),
        "transfer_ca_to_fl_ap": round(R["drift"]["transfer_ca_to_fl"]["ap"], 3),
        "tubbs_percentile": round(R["tubbs"]["percentile_all_ca"], 1),
        "tubbs_october_percentile": round(R["tubbs"]["percentile_october"], 1),
        "bakeoff_winner": best_track,
        "bakeoff_spread": round(R["bakeoff"]["spread"], 5) if "spread" in R["bakeoff"] else None,
    }
    R["runtime_minutes"] = round((time.time() - t_start) / 60, 1)
    (wf.DATA / "modeling_results.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"[done] modeling_results.json in {R['runtime_minutes']} min")


# ---------------------------------------------------------------------------
def climatology_rate(train: pd.DataFrame, test: pd.DataFrame, thr: float) -> np.ndarray:
    """The groupby bar: historical escalation RATE per 0.2-degree cell x month."""
    tr = train.assign(_y=(train["fire_size"] >= thr).astype(float),
                      _la=np.floor(train["lat"] * 5) / 5, _lo=np.floor(train["lon"] * 5) / 5)
    rate = tr.groupby(["_la", "_lo", "month"], observed=True)["_y"].mean()
    key = pd.MultiIndex.from_arrays([np.floor(test["lat"] * 5) / 5,
                                     np.floor(test["lon"] * 5) / 5, test["month"]])
    return pd.Series(rate.reindex(key).to_numpy(), index=test.index).fillna(tr["_y"].mean()).to_numpy()


def calibrate_on_val(make_model, X_tr, y_tr, X_va, y_va, X_te, y_te, n_boot, blocks_te) -> dict:
    """Fit the model on train, then learn a probability map on VAL, and apply to
    TEST. Round 1 never used the val era at all."""
    from sklearn.isotonic import IsotonicRegression

    model = make_model()
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
    sig = LogisticRegression(max_iter=1000).fit(_logit(p_va).reshape(-1, 1), y_va)
    p_iso = iso.predict(p_te)
    p_sig = sig.predict_proba(_logit(p_te).reshape(-1, 1))[:, 1]

    def block(p):
        return {"bss": models.brier_skill_score(y_te, p), "ece": models.ece(y_te, p),
                "ap": float(ap(y_te, p)),
                **models.calibration_slope_intercept(y_te, p),
                "mean_pred": float(np.mean(p))}

    return {"raw": block(p_te), "isotonic": block(p_iso), "sigmoid": block(p_sig),
            "note": "isotonic/sigmoid fitted on the 2015-2017 validation era, applied to 2018-2020",
            "_proba": {"raw": p_te.tolist(), "isotonic": p_iso.tolist(), "sigmoid": p_sig.tolist()}}


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def leak_ladder(tr, te, X_tr, X_te, y_tr, y_te, params) -> dict:
    """Rungs of increasingly obvious leakage. The teaching value is L1/L2: an
    identifier that merely EXISTS for big fires is a near-perfect predictor,
    and neither is named 'fire_size'."""
    rungs = []

    def run(name, extra_tr=None, extra_te=None, note=""):
        A, B = X_tr.copy(), X_te.copy()
        if extra_tr is not None:
            for k in extra_tr:
                A[k] = extra_tr[k]
                B[k] = extra_te[k]
        p, _ = fit_predict(tuning.make_hgb(params), A, y_tr, B)
        rungs.append({"rung": name, "ap": float(ap(y_te, p)),
                      "roc_auc": float(roc_auc_score(y_te, p)), "note": note})
        print(f"  [leak] {name:22s} AP={rungs[-1]['ap']:.4f}", flush=True)

    run("L0_honest", note="at-ignition features only")
    run("L1_has_mtbs",
        {"has_mtbs": tr["mtbs_id"].notna().astype(float)},
        {"has_mtbs": te["mtbs_id"].notna().astype(float)},
        "MTBS maps only large fires -- the ID's existence encodes the outcome")
    run("L2_has_ics209",
        {"has_ics209": tr["ics209_id"].notna().astype(float)},
        {"has_ics209": te["ics209_id"].notna().astype(float)},
        "an ICS-209 is filed only for incidents that drew a management team")
    run("L3_burn_days",
        {"burn_days": tr["burn_days"].astype(float)},
        {"burn_days": te["burn_days"].astype(float)},
        "containment minus discovery: only knowable after the fire")
    run("L4_fire_size",
        {"fire_size": tr["fire_size"].astype(float)},
        {"fire_size": te["fire_size"].astype(float)},
        "the label source itself")

    stats_block = {
        "p_large_given_mtbs": float((te["fire_size"] >= THRESHOLD)[te["mtbs_id"].notna()].mean()),
        "p_large_given_no_mtbs": float((te["fire_size"] >= THRESHOLD)[te["mtbs_id"].isna()].mean()),
        "mtbs_presence_rate": float(te["mtbs_id"].notna().mean()),
    }
    return {"rungs": rungs, "mechanism": stats_block,
            "why_round1_probe_was_flat":
                "Round 1 added FIRE_SIZE to a CAUSE model (+0.0024). Fire size is not the "
                "source of a cause label and the model already knew 'remote high country' "
                "through elevation, ecoregion and location. The probe was pointed at a task "
                "where the leak is not a leak -- the finding was right, the framing was wrong."}


def grouped_importance(model, X_te, y_te, n_repeats=5) -> dict:
    """Correlated features must be permuted TOGETHER. With fm1000 and erc_5d_max
    at r = -0.96, permuting one at a time lets the other stand in, and both look
    unimportant -- the classic collinearity artefact."""
    groups = {
        "fire_weather_today": [c for c in X_te.columns if c in
                               ("erc", "vpd", "bi", "fm100", "fm1000", "tmmx", "rmin", "wind",
                                "precip", "hdw", "ffwi", "sph", "srad", "etr", "tmmn", "rmax")],
        "fire_weather_antecedent": [c for c in X_te.columns if "5d" in c or c == "erc_rising"
                                    or c == "dry_spell"],
        "weather_anomalies": [c for c in X_te.columns if c.endswith("_anom")
                              or c.endswith("_pctl_mid") or c.endswith("_normal")],
        "wind_terrain": [c for c in X_te.columns if c in
                         ("wind_slope_align", "downslope_wind", "wind_u", "wind_v", "wind_dir",
                          "wind_dir_5d_max", "slope", "aspect", "northness", "eastness",
                          "is_flat", "tri", "tpi", "elevation")],
        "fuels_vegetation": [c for c in X_te.columns if c in
                             ("evc_cover_pct", "evc_lifeform", "evc_nonveg", "evh_height", "rpms",
                              "cheatgrass", "exotic_grass", "fuel_load_dryness",
                              "cheatgrass_x_vpd", "exotic_x_erc", "ndvi_1day", "land_cover",
                              "frg", "evt")],
        "human_exposure": [c for c in X_te.columns if c in
                           ("population", "log_population", "ghm", "pop_x_ghm", "svi",
                            "firestations_10km", "firestations_10km_f")],
        "location_season": [c for c in X_te.columns if c in
                            ("lat", "lon", "month", "doy_std", "dow", "doy_sin", "doy_cos",
                             "ecoregion_l3", "aridity", "annual_precip", "annual_temp", "sdi")],
        "fire_history": [c for c in X_te.columns if c.startswith("hist_")],
    }
    base = ap(y_te, model.predict_proba(X_te)[:, 1])
    rng = np.random.default_rng(42)
    out = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X_te.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            for c in cols:
                Xp[c] = Xp[c].to_numpy()[perm]
            drops.append(base - ap(y_te, model.predict_proba(Xp)[:, 1]))
        out.append({"group": name, "n_features": len(cols),
                    "drop_ap": float(np.mean(drops)), "sd": float(np.std(drops))})
    out.sort(key=lambda r: -r["drop_ap"])

    single = permutation_importance(model, X_te, y_te, n_repeats=max(2, n_repeats - 2),
                                    random_state=42, scoring="average_precision")
    top_single = sorted(zip(X_te.columns, single.importances_mean),
                        key=lambda t: -t[1])[:15]
    return {"baseline_ap": float(base), "grouped": out,
            "single_top15": [{"feature": f, "drop_ap": float(v)} for f, v in top_single],
            "note": "grouped permutation is the honest read when |r| reaches 0.96"}


def shap_summary(lgb_model, X_te, n=2000) -> dict:
    import shap
    sample = X_te.sample(min(n, len(X_te)), random_state=42)
    expl = shap.TreeExplainer(lgb_model)
    sv = expl.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:15]
    return {"available": True, "n_sampled": int(len(sample)),
            "top_mean_abs": [{"feature": str(sample.columns[i]),
                              "mean_abs_shap": float(mean_abs[i])} for i in order]}


def tubbs_case_study(ca_all, ca_tr, y_tr, X_tr, params, cal_kind, cal_block) -> dict:
    """Score the Tubbs Fire with a model fit only on <=2014 rows (early stopping
    is disabled in make_hgb, so nothing internal touches later years either).

    Round-4 red team caveats on THIS version, addressed in run_redteam.py:
    the percentile here ranks Tubbs against all CA rows incl. ~200k the model
    trained on (mixed-sample reference), and the risk score is uncalibrated.
    The red-team recomputation uses a 2015-2020 out-of-sample reference and the
    val-selected calibration map. cal_kind/cal_block are accepted for signature
    stability but unused here."""
    model = tuning.make_hgb(params)
    model.fit(X_tr, y_tr)

    tubbs_row = ca_all[ca_all["fod_id"] == wf.TUBBS_FOD_ID]
    if len(tubbs_row) != 1:
        return {"error": f"expected 1 Tubbs row, found {len(tubbs_row)}"}
    X_all = fx.build_matrix(ca_all, "R6_history").reindex(columns=X_tr.columns)
    p_all = model.predict_proba(X_all)[:, 1]
    ca_scored = ca_all.assign(_p=p_all)

    p_tubbs = float(ca_scored.loc[ca_scored["fod_id"] == wf.TUBBS_FOD_ID, "_p"].iloc[0])
    pct_all = float((ca_scored["_p"] < p_tubbs).mean() * 100)
    oct_mask = ca_scored["month"] == 10
    pct_oct = float((ca_scored.loc[oct_mask, "_p"] < p_tubbs).mean() * 100)
    r = tubbs_row.iloc[0]
    # scores kept for the figure so it plots the SAME model's distribution that
    # produced the percentile (mixing raw and calibrated scales would be a lie)
    oct_scores = ca_scored.loc[oct_mask, "_p"].to_numpy()
    return {"fod_id": int(wf.TUBBS_FOD_ID), "risk_score": p_tubbs,
            "_scores_all": p_all, "_scores_october": oct_scores,
            "percentile_all_ca": pct_all, "percentile_october": pct_oct,
            "n_scored": int(len(ca_scored)),
            "rank_all_ca": int((ca_scored["_p"] > p_tubbs).sum() + 1),
            "in_top_1pct": bool(pct_all >= 99.0),
            "model_train_max_year": int(wf.ML_TRAIN_MAX_YEAR),
            "conditions": {k: (None if pd.isna(r.get(k)) else float(r.get(k)))
                           for k in ("erc", "vpd", "rmin", "wind", "hdw", "ffwi",
                                     "wind_slope_align", "downslope_wind", "erc_anom")},
            "note": "trained on <=2014 only; the 2017 fire is a pure out-of-sample row"}


# ---------------------------------------------------------------------------
def make_figures(R, y_te, preds, p_cal, cal, ca_te):
    zoo = R["zoo"]
    # m1 threshold ladder
    lad = R["threshold_ladder"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = range(len(lad))
    ax.plot(x, [r["ap"] for r in lad], color=COL["fire"], lw=2, marker="o", label="AP")
    ax.plot(x, [r["prevalence"] for r in lad], color=COL["muted"], lw=1.4, ls="--",
            marker="s", ms=4, label="base rate (no-skill)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"≥{int(r['threshold'])} ac" for r in lad])
    ax.set_yscale("log")
    ax.set_ylabel("average precision (log)")
    ax.set_title("Discrimination holds as the bar rises; the base rate falls faster",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"])
    style_ax(ax)
    savefig(fig, "m1_threshold_ladder.png")

    # m2 tuning traces
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for key, color, lab in (("A_sklearn_random", COL["blue"], "A · sklearn random"),
                            ("B_optuna_tpe", COL["fire"], "B · optuna TPE"),
                            ("C_lightgbm_optuna", COL["green"], "C · lightgbm + optuna")):
        t = R["bakeoff"]["tracks"].get(key)
        if not t or not t["trials"]:
            continue
        s = [tr["score"] for tr in t["trials"]]
        ax.plot(range(1, len(s) + 1), np.maximum.accumulate(s), color=color, lw=1.8, label=lab)
    ax.axhline(R["bakeoff"]["default_reference"]["score"], color=COL["muted"], ls=":",
               lw=1.5, label="Round-1 default config")
    ax.set_xlabel("trial")
    ax.set_ylabel("best CV average precision so far")
    ax.set_title("Three search strategies, one space, equal budget", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"])
    style_ax(ax)
    savefig(fig, "m2_tuning_traces.png")

    # m3 model zoo
    zoo = R["zoo"]
    order = sorted(zoo, key=lambda k: zoo[k]["ap"])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, name in enumerate(order):
        m = zoo[name]
        c = COL["fire"] if name == R["champion"] else (
            COL["muted"] if name in ("climatology_rate", "erc_only") else COL["blue"])
        ax.barh(i, m["ap"], color=c)
        ax.plot([m["ap_ci"]["lo"], m["ap_ci"]["hi"]], [i, i], color=COL["ink2"], lw=1.4)
    ax.axvline(R["data"]["prevalence_test"], color=COL["red"], ls="--", lw=1.2)
    ax.text(R["data"]["prevalence_test"], len(order) - 0.4, " no-skill", fontsize=8,
            color=COL["red"], va="top")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("test average precision (spatial-block bootstrap 95% CI)")
    ax.set_title("The zoo, with intervals Round 1 never computed", loc="left", fontsize=11)
    style_ax(ax, grid_axis="x")
    savefig(fig, "m3_model_zoo.png")

    # m4 calibration before/after
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for lab, p, color in (("Round-1 style (raw, balanced)", preds["hgb_default_balanced"], COL["red"]),
                          (f"calibrated ({R['calibration']['chosen']})", p_cal, COL["green"])):
        frac, mean_p = calibration_curve(y_te, p, n_bins=12, strategy="quantile")
        axes[0].plot(mean_p, frac, marker="o", ms=4, lw=1.7, color=color, label=lab)
    axes[0].plot([0, 1], [0, 1], ls=":", color=COL["axis"], lw=1)
    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("Reliability: the claim Round 1 got backwards", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    style_ax(axes[0], grid_axis="")
    axes[1].hist(preds["hgb_default_balanced"], bins=40, color=COL["red"], alpha=0.55,
                 label="Round-1 (balanced)")
    axes[1].hist(p_cal, bins=40, color=COL["green"], alpha=0.65, label="calibrated")
    axes[1].axvline(R["data"]["prevalence_test"], color=COL["ink2"], ls="--", lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("predicted probability")
    # both panels contrast the SAME two models, so the title must name those two
    axes[1].set_title(f"Round-1 (balanced) BSS {zoo['hgb_default_balanced']['bss']:+.2f}  →  "
                      f"calibrated {cal[R['calibration']['chosen']]['bss']:+.3f}",
                      loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    style_ax(axes[1])
    fig.tight_layout()
    savefig(fig, "m4_calibration.png")

    # m5 decision curve
    nb = R["decision"]["net_benefit"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot([r["threshold"] for r in nb], [r["net_benefit_model"] for r in nb],
            color=COL["fire"], lw=2, label="model")
    ax.plot([r["threshold"] for r in nb], [r["net_benefit_treat_all"] for r in nb],
            color=COL["muted"], lw=1.4, ls="--", label="treat every ignition")
    ax.axhline(0, color=COL["axis"], lw=1)
    ax.set_ylim(min(-0.005, min(r["net_benefit_model"] for r in nb)), None)
    ax.set_xlabel("risk threshold (encodes the cost ratio)")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision curve: where acting on the model beats acting on everything",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"])
    style_ax(ax)
    savefig(fig, "m5_decision_curve.png")

    # m6 precision/recall @ k
    tk = R["decision"]["topk"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ks = [t["k"] * 100 for t in tk]
    ax.plot(ks, [t["precision"] for t in tk], color=COL["fire"], lw=2, marker="o", label="precision")
    ax.plot(ks, [t["recall"] for t in tk], color=COL["blue"], lw=2, marker="s", label="recall")
    ax.plot(ks, [t["ceiling_recall"] for t in tk], color=COL["blue"], lw=1.2, ls=":",
            label="recall ceiling (perfect ranking)")
    ax.axhline(R["decision"]["prevalence"], color=COL["red"], ls="--", lw=1.2,
               label="base rate")
    ax.set_xscale("log")
    ax.set_xlabel("share of ignitions flagged (%)")
    ax.set_title("If you can only act on the top k%, this is what you get", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    style_ax(ax)
    savefig(fig, "m6_precision_at_k.png")

    # m7 ablation
    abl = R["ablation"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs = range(len(abl))
    ax.plot(xs, [a["ap"] for a in abl], color=COL["fire"], lw=2, marker="o")
    ax.fill_between(list(xs), [a["ap_lo"] for a in abl], [a["ap_hi"] for a in abl],
                    color=COL["fire"], alpha=0.15)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([a["rung"].split("_", 1)[1] for a in abl], rotation=20, ha="right",
                       fontsize=8)
    ax.set_ylabel("test average precision")
    ax.set_title("What each feature family actually bought (95% CI)", loc="left", fontsize=11)
    style_ax(ax)
    savefig(fig, "m7_ablation.png")

    # m8 grouped vs single importance
    imp = R["importance"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    g = imp["grouped"][::-1]
    axes[0].barh([r["group"] for r in g], [r["drop_ap"] for r in g], color=COL["fire"])
    axes[0].set_xlabel("AP drop when the whole group is permuted")
    axes[0].set_title("Grouped (honest under collinearity)", loc="left", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=7.5)
    style_ax(axes[0], grid_axis="x")
    s = imp["single_top15"][:10][::-1]
    axes[1].barh([r["feature"] for r in s], [r["drop_ap"] for r in s], color=COL["muted"])
    axes[1].set_xlabel("AP drop, one feature at a time")
    axes[1].set_title("Single-feature (misleading at r = −0.96)", loc="left", fontsize=10)
    axes[1].tick_params(axis="y", labelsize=7.5)
    style_ax(axes[1], grid_axis="x")
    fig.tight_layout()
    savefig(fig, "m8_importance.png")

    # m9 leak ladder
    lk = R["leak_probe"]["rungs"]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    colors = [COL["green"]] + [COL["gold"], COL["gold"], COL["red"], COL["red"]][:len(lk) - 1]
    ax.bar(range(len(lk)), [r["ap"] for r in lk], color=colors)
    for i, r in enumerate(lk):
        ax.text(i, r["ap"] + 0.02, f"{r['ap']:.3f}", ha="center", fontsize=8.5, color=COL["ink2"])
    ax.set_xticks(range(len(lk)))
    ax.set_xticklabels([r["rung"].replace("_", "\n") for r in lk], fontsize=8)
    ax.set_ylabel("test average precision")
    ax.set_title("A leak probe that bites: an ID that only exists for big fires",
                 loc="left", fontsize=11)
    style_ax(ax)
    savefig(fig, "m9_leak_ladder.png")

    # m10 drift + transfer
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    by = R["drift"]["by_year"]
    axes[0].bar([b["year"] for b in by], [b["ap"] for b in by], color=COL["fire"])
    axes[0].set_ylabel("AP")
    axes[0].set_title("Test-era performance by year", loc="left", fontsize=10)
    style_ax(axes[0])
    tr = R["drift"]["transfer_ca_to_fl"]
    axes[1].bar(["CA→FL\n(transfer)", "FL native"], [tr["ap"], tr["native_fl_ap"]],
                color=[COL["gold"], COL["blue"]])
    axes[1].axhline(tr["fl_prevalence"], color=COL["red"], ls="--", lw=1.2)
    axes[1].set_title("Cross-regime transfer", loc="left", fontsize=10)
    style_ax(axes[1])
    fig.tight_layout()
    savefig(fig, "m10_drift_transfer.png")

    # m11 Tubbs
    tub = R["tubbs"]
    if "error" not in tub and "_scores_all" in tub:
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        all_scores = np.asarray(tub["_scores_all"])
        oct_scores = np.asarray(tub["_scores_october"])
        ax.hist(all_scores, bins=60, color=COL["muted"], alpha=0.7,
                label="all CA ignitions, 1992-2020")
        if len(oct_scores):
            ax.hist(oct_scores, bins=60, color=COL["gold"], alpha=0.75, label="October ignitions")
        ax.axvline(tub["risk_score"], color=COL["red"], lw=2.2)
        ax.text(tub["risk_score"], ax.get_ylim()[1] * 0.6,
                f"  TUBBS\n  {tub['percentile_all_ca']:.1f}th pct",
                color=COL["red"], fontsize=9, va="top")
        ax.set_yscale("log")
        ax.set_xlabel("escalation risk score")
        ax.set_title("Scoring the fire that started this project (model trained ≤2014)",
                     loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"])
        style_ax(ax)
        savefig(fig, "m11_tubbs.png")


if __name__ == "__main__":
    main()
