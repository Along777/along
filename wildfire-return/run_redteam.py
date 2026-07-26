from __future__ import annotations

"""Round 4: the hostile pre-stakeholder review, executed.

A fresh-context adversarial reviewer audited the Round-3 code and returned 35
findings. This runner executes every experiment those findings demanded and
writes the numbers that either fix a hole or defend a choice:

  seed stability        are the champion margin / tuner spread / feature gain
                        bigger than seed noise? (they had never been measured)
  clean calibration     maps selected on VAL (Round 3 selected on test),
                        for BOTH systems, labeled
  geography ablation    how much of the skill is memorized place?
  yesterday model       drop all same-day weather: the operational feasibility number
  same-day load         the resource-competition feature the review found missing
  embargoed history     years-since-large using only fires KNOWN large at ignition
  fair zoo              RF/ET/logistic with encoded categoricals, no class weighting
  dup sensitivity       the check Round 2 promised and Round 3 never ran
  2020 concentration    leave-2020-out AP; pooled vs by-year macro
  earned nulls          Nadeau-Bengio t on the bake-off fold vectors
  tripwire v2 demo      must fire on all four leak-ladder rungs, pass the honest set
  Tubbs redo            out-of-sample reference, calibrated risk
  remoteness terciles   is it a fire model or a remoteness lookup?
  decision-curve redo   log-spaced thresholds, truncated where n_flagged < 50

The named SYSTEM OF RECORD is the tuned HistGradientBoosting model: it is the
one that was calibrated, ablated, leak-probed, transferred and Tubbs-scored in
Round 3, and it carries no extra dependencies. LightGBM's higher single test
score is reported, not selected (the difference is not FDR-significant).

Run:  python run_redteam.py     (~18 full-train fits, 45-70 min)
"""

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import features as fx
import models
import tuning
import wildfire as wf

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
THRESHOLD = 100.0
SEEDS = [42, 7, 101, 2020, 1234]

# Same-day weather: everything not knowable before the day ends (tmmx is the
# day's MAX temp; rmin the day's MIN humidity). The yesterday-knowledge model
# drops all of it and keeps antecedents, normals, statics, and history.
SAME_DAY_COLS = ["tmmx", "rmin", "wind", "precip", "vpd", "erc", "bi", "fm100", "fm1000",
                 "hdw", "ffwi", "diurnal_range", "rh_range", "erc_rising",
                 "wind_u", "wind_v", "wind_slope_align", "downslope_wind", "wind_dir",
                 "erc_anom", "vpd_anom", "fm100_anom", "fm1000_anom", "bi_anom",
                 "tmmx_anom", "rmin_anom", "precip_anom",
                 "erc_pctl_mid", "vpd_pctl_mid", "fm100_pctl_mid", "bi_pctl_mid",
                 "wind_pctl_mid"]
GEO_COLS_PREFIX = ("hist_",)
GEO_COLS = ["lat", "lon"]
GEO_CATS = ["ecoregion_l3"]


def style_ax(ax, grid_axis="y"):
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COL["axis"])
    ax.tick_params(colors=COL["ink2"], labelsize=9)
    ax.title.set_color(COL["ink"])


def savefig(fig, name):
    wf.FIG.mkdir(exist_ok=True)
    fig.savefig(wf.FIG / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig ] {name}", flush=True)


def ap(y, s):
    return average_precision_score(y, s)


def sigmoid_map(p_fit, y_fit):
    x = np.log(np.clip(p_fit, 1e-6, 1 - 1e-6) / (1 - np.clip(p_fit, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(x.reshape(-1, 1), y_fit)

    def apply(p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return lr.predict_proba(z.reshape(-1, 1))[:, 1]

    return apply


def cal_metrics(y, p):
    return {"bss": models.brier_skill_score(y, p), "ece": models.ece(y, p),
            "ap": float(ap(y, p)), **models.calibration_slope_intercept(y, p),
            "mean_pred": float(np.mean(p))}


def main() -> None:
    t0 = time.time()
    R: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "system_of_record": "hgb_tuned",
               "why": "the model that was calibrated, ablated, leak-probed, transferred and "
                      "Tubbs-scored in Round 3; zero extra dependencies; LightGBM's higher "
                      "single test AP is reported, not selected (q=0.085, not separable)"}

    M = json.loads((wf.DATA / "modeling_results.json").read_text())
    bake = M["bakeoff"]
    hgb_best = bake["tracks"]["A_sklearn_random"]["best"]["params"]
    if bake["tracks"]["B_optuna_tpe"]["best"]["score"] > \
       bake["tracks"]["A_sklearn_random"]["best"]["score"]:
        hgb_best = bake["tracks"]["B_optuna_tpe"]["best"]["params"]
    lgb_best = bake["tracks"]["C_lightgbm_optuna"]["best"]["params"]
    R["configs"] = {"hgb": hgb_best, "lgbm": lgb_best}

    # coincidence check (F1): the search-CV winner and the max-on-test winner agree
    R["champion_procedure"] = {
        "search_cv_winner": max(bake["tracks"], key=lambda k: bake["tracks"][k]["best"]["score"]),
        "max_on_test_winner": M["champion"],
        "coincide": max(bake["tracks"], key=lambda k: bake["tracks"][k]["best"]["score"])
                    == "C_lightgbm_optuna" and M["champion"] == "lightgbm_tuned",
        "note": "Round 3's champion function took a max over 9 test scores (winner's curse); "
                "it happened to coincide with the pre-test search winner. Round 4 re-keys the "
                "headline to the pre-committed, fully-analyzed system instead."}

    print("[load] CA with engineered features")
    ca = fx.add_features(wf.load_state_fires("CA", strict=True))
    R["history_causality"] = fx.assert_history_is_causal(ca, n_samples=120)
    s = models.temporal_split(ca)
    tr, va, te = ca[s["train"]], ca[s["val"]], ca[s["test"]]
    y_tr = (tr["fire_size"] >= THRESHOLD).to_numpy(int)
    y_va = (va["fire_size"] >= THRESHOLD).to_numpy(int)
    y_te = (te["fire_size"] >= THRESHOLD).to_numpy(int)
    X_tr = fx.build_matrix(tr, "R6_history")
    X_va = fx.build_matrix(va, "R6_history").reindex(columns=X_tr.columns)
    X_te = fx.build_matrix(te, "R6_history").reindex(columns=X_tr.columns)
    blocks_te = models.spatial_blocks(te).to_numpy()
    base_te = float(y_te.mean())

    # ------------------------------------------------------------- tripwire v2 demo
    print("[trip] tripwire v2: honest must pass, all four leaks must fire")
    honest = fx.leak_tripwire_v2(X_tr, y_tr)
    cand = X_tr.copy()
    cand["has_mtbs"] = tr["mtbs_id"].notna().astype(float)
    cand["has_ics209"] = tr["ics209_id"].notna().astype(float)
    cand["burn_days"] = tr["burn_days"].astype(float)
    cand["fire_size"] = tr["fire_size"].astype(float)
    leaky = fx.leak_tripwire_v2(cand, y_tr)
    caught = sorted(f["feature"] for f in leaky["flagged"])
    assert honest["passed"], f"tripwire v2 fired on honest features: {honest['flagged']}"
    assert set(caught) >= {"has_mtbs", "has_ics209", "burn_days", "fire_size"}, \
        f"tripwire v2 failed to catch all four leaks; caught only {caught}"
    R["tripwire_v2"] = {"honest_passed": True, "honest_top": honest["top"][:4],
                        "leaks_caught": caught,
                        "thresholds": {"ap_lift": 5.0, "presence_lift": 20.0, "auc": 0.80},
                        "calibration_note": "thresholds set from measured margins: honest "
                                            "ceiling 1.72x AP lift / 0.68 AUC; weakest leak "
                                            "7.1x / 0.80"}
    print(f"       honest passed; caught {caught}")

    # ------------------------------------------------------------- seed stability
    print(f"[seed] {len(SEEDS)} seeds x 2 systems on full train")
    seed_runs: dict[str, list] = {"hgb": [], "lgbm": []}
    keep_models: dict = {}
    for seed in SEEDS:
        h = tuning.make_hgb(hgb_best)
        h.set_params(random_state=seed)
        h.fit(X_tr, y_tr)
        pa = h.predict_proba(X_te)[:, 1]
        seed_runs["hgb"].append(float(ap(y_te, pa)))
        g = tuning.make_lgbm(lgb_best)
        g.set_params(random_state=seed)
        g.fit(X_tr, y_tr)
        pb = g.predict_proba(X_te)[:, 1]
        seed_runs["lgbm"].append(float(ap(y_te, pb)))
        if seed == SEEDS[0]:
            keep_models = {"hgb": h, "lgbm": g,
                           "hgb_te": pa, "lgbm_te": pb,
                           "hgb_va": h.predict_proba(X_va)[:, 1],
                           "lgbm_va": g.predict_proba(X_va)[:, 1]}
        print(f"       seed {seed}: hgb {seed_runs['hgb'][-1]:.4f}  "
              f"lgbm {seed_runs['lgbm'][-1]:.4f}", flush=True)
    sd_h = float(np.std(seed_runs["hgb"], ddof=1))
    sd_g = float(np.std(seed_runs["lgbm"], ddof=1))
    R["seed_stability"] = {
        "hgb": {"ap_by_seed": seed_runs["hgb"], "mean": float(np.mean(seed_runs["hgb"])),
                "sd": sd_h},
        "lgbm": {"ap_by_seed": seed_runs["lgbm"], "mean": float(np.mean(seed_runs["lgbm"])),
                 "sd": sd_g},
        "verdicts": {
            "champion_margin_0p0070": "within noise" if 0.0070 < 2 * max(sd_h, sd_g)
            else "exceeds 2 seed-sd",
            "tuner_spread_0p0053": "within noise" if 0.0053 < 2 * max(sd_h, sd_g)
            else "exceeds 2 seed-sd",
            "feature_gain_0p0031": "within noise" if 0.0031 < 2 * max(sd_h, sd_g)
            else "exceeds 2 seed-sd"},
    }
    print(f"       hgb {R['seed_stability']['hgb']['mean']:.4f} sd {sd_h:.4f} | "
          f"lgbm {R['seed_stability']['lgbm']['mean']:.4f} sd {sd_g:.4f}")

    # subsample-transfer premise (F15): full-data val ordering vs search ordering
    d_def = tuning.make_hgb(bake["default_reference"]["params"])
    d_def.fit(X_tr, y_tr)
    val_order = {
        "hgb_tuned": float(ap(y_va, keep_models["hgb_va"])),
        "lgbm_tuned": float(ap(y_va, keep_models["lgbm_va"])),
        "hgb_default": float(ap(y_va, d_def.predict_proba(X_va)[:, 1]))}
    R["subsample_transfer_check"] = {
        "val_ap": val_order,
        "tuned_beats_default_on_val": bool(val_order["hgb_tuned"] > val_order["hgb_default"]),
        "lgbm_vs_hgb_order_held": bool(val_order["lgbm_tuned"] >= val_order["hgb_tuned"]),
        "note": "the search (7.3%-prevalence subsample) ranked lgbm >= hgb_tuned >> default; "
                "this is the same ordering question asked on full-data val-era AP"}
    print(f"       val ordering: {val_order}")

    # ------------------------------------------------------------- clean calibration
    print("[cal ] val-selected calibration for both systems")
    cal_out = {}
    p_cal_sor = None
    for name in ("hgb", "lgbm"):
        p_va, p_te = keep_models[f"{name}_va"], keep_models[f"{name}_te"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
        sig = sigmoid_map(p_va, y_va)
        val_bss = {"isotonic": models.brier_skill_score(y_va, iso.predict(p_va)),
                   "sigmoid": models.brier_skill_score(y_va, sig(p_va))}
        chosen = max(val_bss, key=val_bss.get)
        p_cal = iso.predict(p_te) if chosen == "isotonic" else sig(p_te)
        m = cal_metrics(y_te, p_cal)
        tk = models.topk_table(y_te, p_cal, (0.005, 0.01, 0.02, 0.05))
        cal_out[name] = {"selected_on_val": chosen, "val_bss": val_bss, "test": m,
                         "precision_at_1pct": tk[1]["precision"],
                         "recall_at_1pct": tk[1]["recall"], "topk": tk}
        if name == "hgb":
            p_cal_sor = p_cal
        print(f"       {name}: chose {chosen} on val; test BSS {m['bss']:+.4f} "
              f"ECE {m['ece']:.4f} p@1% {tk[1]['precision']:.3f}", flush=True)
    R["calibration_v2"] = cal_out

    # ECE reference values (F23): what does 0.004 even mean at a 2% base rate?
    rng = np.random.default_rng(0)
    p_const = np.full(len(y_te), base_te)
    p_noise = np.clip(rng.normal(base_te, base_te / 2, len(y_te)), 1e-4, 0.5)
    R["ece_references"] = {
        "constant_base_rate": models.ece(y_te, p_const),
        "pure_noise_near_base": models.ece(y_te, p_noise),
        "system_of_record": cal_out["hgb"]["test"]["ece"],
        "note": "equal-count-bin ECE is mechanically small when predictions concentrate "
                "near a 2% base rate; slope/intercept are the discriminating metrics"}
    # tail reliability: the top 5% of predictions, where decisions actually happen
    k = max(1, int(0.05 * len(p_cal_sor)))
    top_idx = np.argsort(-p_cal_sor)[:k]
    R["tail_reliability"] = {
        "top_share": 0.05, "n": int(k),
        "mean_predicted": float(np.mean(p_cal_sor[top_idx])),
        "observed_rate": float(y_te[top_idx].mean()),
        "ece_top5pct": models.ece(y_te[top_idx], p_cal_sor[top_idx], bins=8)}
    print(f"       tail: predicted {R['tail_reliability']['mean_predicted']:.3f} vs "
          f"observed {R['tail_reliability']['observed_rate']:.3f}")

    # ------------------------------------------------------------- ablation refits
    def refit_ap(drop_cols=None, add_col=None, label=""):
        Xtr2, Xte2 = X_tr, X_te
        if drop_cols:
            keep = [c for c in X_tr.columns if c not in drop_cols]
            Xtr2, Xte2 = X_tr[keep], X_te[keep]
        if add_col is not None:
            Xtr2 = Xtr2.copy()
            Xte2 = Xte2.copy()
            Xtr2[add_col] = tr[add_col].astype("float32").to_numpy()
            Xte2[add_col] = te[add_col].astype("float32").to_numpy()
        m = tuning.make_hgb(hgb_best)
        m.fit(Xtr2, y_tr)
        p = m.predict_proba(Xte2)[:, 1]
        a = float(ap(y_te, p))
        print(f"  [expt] {label:34s} AP={a:.4f} ({Xtr2.shape[1]} cols)", flush=True)
        return a, p

    baseline_ap = seed_runs["hgb"][0]
    print("[expt] targeted refits (system-of-record config)")
    geo_drop = GEO_COLS + GEO_CATS + [c for c in X_tr.columns if c.startswith("hist_")]
    ap_nogeo, _ = refit_ap(drop_cols=set(geo_drop), label="no location/history (geography)")
    ap_yday, _ = refit_ap(drop_cols=set(SAME_DAY_COLS), label="yesterday-knowledge (no same-day wx)")
    ap_sameday, p_sameday = refit_ap(add_col="same_day_cell_ignitions",
                                     label="+ same-day cell ignition load")

    # embargoed history (F17)
    print("  [expt] embargoed years-since-large ...", flush=True)
    ca_emb = fx.add_history(wf.load_state_fires("CA", strict=True), embargo_large=True)
    ca_emb = fx.add_features(ca_emb, with_history=False)
    tr_e = ca_emb[models.temporal_split(ca_emb)["train"]]
    te_e = ca_emb[models.temporal_split(ca_emb)["test"]]
    Xtr_e = fx.build_matrix(tr_e, "R6_history").reindex(columns=X_tr.columns)
    Xte_e = fx.build_matrix(te_e, "R6_history").reindex(columns=X_tr.columns)
    m_e = tuning.make_hgb(hgb_best)
    m_e.fit(Xtr_e, y_tr)
    ap_embargo = float(ap(y_te, m_e.predict_proba(Xte_e)[:, 1]))
    print(f"  [expt] {'embargoed history':34s} AP={ap_embargo:.4f}", flush=True)

    d_same = models.paired_bootstrap_diff(y_te, p_sameday, keep_models["hgb_te"], ap,
                                          n=1000, blocks=blocks_te)
    R["experiments"] = {
        "baseline_seed42": baseline_ap,
        "no_geography": {"ap": ap_nogeo, "delta": ap_nogeo - baseline_ap,
                         "dropped": sorted(set(geo_drop))},
        "yesterday_knowledge": {"ap": ap_yday, "delta": ap_yday - baseline_ap,
                                "n_dropped": len(set(SAME_DAY_COLS) & set(X_tr.columns))},
        "same_day_load": {"ap": ap_sameday, "delta": ap_sameday - baseline_ap,
                          "paired": d_same,
                          "note": "post-hoc feature suggested by the red team; date-resolution "
                                  "simultaneity caveat applies"},
        "embargoed_history": {"ap": ap_embargo, "delta": ap_embargo - baseline_ap},
    }

    # ------------------------------------------------------------- fair zoo redux
    print("[zoo2] fair re-runs: encoded categoricals, no class weighting")
    cats = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    nums = [c for c in X_tr.columns if c not in cats]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-2)
    imp = SimpleImputer(strategy="median")

    def enc_frame(X):
        a = imp.transform(X[nums]) if hasattr(imp, "statistics_") else imp.fit_transform(X[nums])
        b = enc.transform(X[cats]) if hasattr(enc, "categories_") else enc.fit_transform(X[cats])
        return np.hstack([a, b])

    Ztr = enc_frame(X_tr)
    Zte = enc_frame(X_te)
    fair = {}
    for name, mdl in (
        ("random_forest", RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                                 max_features="sqrt", max_samples=0.5,
                                                 n_jobs=2, random_state=42)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10,
                                             max_features="sqrt", max_samples=0.5,
                                             bootstrap=True, n_jobs=2, random_state=42)),
        ("ridge_logistic", make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=3000, C=0.1))),
    ):
        mdl.fit(Ztr, y_tr)
        p = mdl.predict_proba(Zte)[:, 1]
        fair[name] = {"ap": float(ap(y_te, p)), "bss": models.brier_skill_score(y_te, p),
                      "round3_ap": M["zoo"].get(
                          name if name != "ridge_logistic" else "logistic_en", {}).get("ap")}
        print(f"       {name:16s} AP={fair[name]['ap']:.4f} "
              f"(round-3: {fair[name]['round3_ap']:.4f}) BSS={fair[name]['bss']:+.2f}",
              flush=True)
    R["fair_zoo"] = {"note": "same features as the GBMs (ordinal-encoded categoricals), "
                             "no class weighting -- Round 3 denied these models 4 features "
                             "incl. evt and handed them the broken balanced weighting",
                     "families": fair}

    # ------------------------------------------------------------- dup / 2020 / lift CI
    print("[misc] dup sensitivity, 2020 concentration, lift CI, NB-t")
    dup = pd.read_parquet(wf.DATA / "dup_flags.parquet")
    flagged = set(dup.loc[dup["tier1"], "fod_id"])
    mask_keep = ~te["fod_id"].isin(flagged).to_numpy()
    R["dup_sensitivity"] = {
        "n_flagged_in_test": int((~mask_keep).sum()),
        "ap_all": baseline_ap,
        "ap_without_flagged": float(ap(y_te[mask_keep], keep_models["hgb_te"][mask_keep])),
    }
    R["dup_sensitivity"]["delta"] = (R["dup_sensitivity"]["ap_without_flagged"]
                                     - R["dup_sensitivity"]["ap_all"])

    years = te["fire_year"].to_numpy()
    by_year = {}
    for yr in (2018, 2019, 2020):
        m = years == yr
        by_year[str(yr)] = {"n": int(m.sum()), "positives": int(y_te[m].sum()),
                            "ap": float(ap(y_te[m], keep_models["hgb_te"][m]))}
    no2020 = years != 2020
    R["concentration_2020"] = {
        "by_year": by_year,
        "macro_avg_ap": float(np.mean([v["ap"] for v in by_year.values()])),
        "pooled_ap": baseline_ap,
        "leave_2020_out_ap": float(ap(y_te[no2020], keep_models["hgb_te"][no2020])),
        "share_of_positives_2020": float(y_te[years == 2020].sum() / y_te.sum())}
    print(f"       leave-2020-out AP: {R['concentration_2020']['leave_2020_out_ap']:.4f}")

    # lift CI: AP/prevalence per block resample cancels the base-rate component
    members = models._block_members(blocks_te)
    rng = np.random.default_rng(42)
    lifts, raw_aps = [], []
    for _ in range(1000):
        idx = models._resample_idx(len(y_te), members, rng)
        if len(np.unique(y_te[idx])) < 2:
            continue
        a = ap(y_te[idx], keep_models["hgb_te"][idx])
        raw_aps.append(a)
        lifts.append(a / y_te[idx].mean())
    R["lift_ci"] = {
        "n_blocks": int(len(members)), "n_singleton_blocks": int(sum(len(m) == 1 for m in members)),
        "ap_ci": [float(np.quantile(raw_aps, 0.025)), float(np.quantile(raw_aps, 0.975))],
        "lift_point": baseline_ap / base_te,
        "lift_ci": [float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975))],
        "note": "the raw AP interval mixes ranking uncertainty with base-rate resampling "
                "(prevalence swings ~±12% across 63-block resamples); the lift interval "
                "cancels the base-rate component"}

    # earned null: Nadeau-Bengio on the bake-off fold vectors
    fa = bake["tracks"]["A_sklearn_random"]["best"]["folds"]
    fb = bake["tracks"]["B_optuna_tpe"]["best"]["folds"]
    fc = bake["tracks"]["C_lightgbm_optuna"]["best"]["folds"]
    n_search = bake.get("n_trials_requested", 25)
    R["earned_nulls"] = {
        "note": "3 spatial folds on the 60k search subsample; approximation stated",
        "optuna_vs_random": models.nadeau_bengio_t(fb, fa, n_train=40000, n_test=20000),
        "lightgbm_vs_random": models.nadeau_bengio_t(fc, fa, n_train=40000, n_test=20000)}
    print(f"       NB-t optuna-vs-random p={R['earned_nulls']['optuna_vs_random']['p']:.3f}; "
          f"lgbm-vs-random p={R['earned_nulls']['lightgbm_vs_random']['p']:.3f}")

    # ------------------------------------------------------------- remoteness terciles
    print("[terc] AP within population terciles + direction checks")
    pop = tr["population"].fillna(0)
    q1, q2 = pop.quantile([1 / 3, 2 / 3])
    pop_te = te["population"].fillna(0).to_numpy()
    terc = {}
    for name, m in (("low_pop", pop_te <= q1), ("mid_pop", (pop_te > q1) & (pop_te <= q2)),
                    ("high_pop", pop_te > q2)):
        terc[name] = {"n": int(m.sum()), "positives": int(y_te[m].sum()),
                      "prevalence": float(y_te[m].mean()),
                      "ap": float(ap(y_te[m], keep_models["hgb_te"][m])),
                      "lift": float(ap(y_te[m], keep_models["hgb_te"][m]) / max(y_te[m].mean(), 1e-9))}
    # direction: correlation of predicted risk with remoteness proxies
    ghm_te = te["ghm"].to_numpy(dtype="float64")
    ok = np.isfinite(ghm_te)
    R["remoteness"] = {
        "terciles": terc,
        "corr_risk_ghm": float(np.corrcoef(keep_models["hgb_te"][ok], ghm_te[ok])[0, 1]),
        "corr_risk_logpop": float(np.corrcoef(keep_models["hgb_te"],
                                              np.log1p(np.clip(pop_te, 0, None)))[0, 1]),
        "note": "negative correlations = higher predicted risk in remoter, emptier places; "
                "the model still ranks well WITHIN every tercile (lift column), so it is "
                "not merely a remoteness lookup -- but remoteness is a real component"}
    for k2, v in terc.items():
        print(f"       {k2:9s} AP={v['ap']:.4f} lift={v['lift']:.1f}x "
              f"(prev {v['prevalence']:.3%})", flush=True)

    # ------------------------------------------------------------- Tubbs redo
    print("[tubb] out-of-sample reference, calibrated risk")
    oos = ca[ca["fire_year"] >= 2015]
    X_oos = fx.build_matrix(oos, "R6_history").reindex(columns=X_tr.columns)
    p_oos = keep_models["hgb"].predict_proba(X_oos)[:, 1]
    iso_or_sig = cal_out["hgb"]["selected_on_val"]
    sig_apply = sigmoid_map(keep_models["hgb_va"], y_va)
    p_oos_cal = sig_apply(p_oos) if iso_or_sig == "sigmoid" else \
        IsotonicRegression(out_of_bounds="clip").fit(keep_models["hgb_va"], y_va).predict(p_oos)
    tub_mask = (oos["fod_id"] == wf.TUBBS_FOD_ID).to_numpy()
    if tub_mask.sum() == 1:
        p_t = float(p_oos[tub_mask][0])
        oct_mask = (oos["month"] == 10).to_numpy()
        R["tubbs_v2"] = {
            "reference": "CA ignitions 2015-2020 only (out-of-sample for the <=2014 model)",
            "n_reference": int(len(oos)),
            "percentile_oos": float((p_oos < p_t).mean() * 100),
            "percentile_october_oos": float((p_oos[oct_mask] < p_t).mean() * 100),
            "calibrated_risk": float(p_oos_cal[tub_mask][0]),
            "round3_percentile_mixed_sample": M["tubbs"]["percentile_all_ca"]}
        print(f"       OOS percentile {R['tubbs_v2']['percentile_oos']:.1f} "
              f"(Oct {R['tubbs_v2']['percentile_october_oos']:.1f}); "
              f"calibrated risk {R['tubbs_v2']['calibrated_risk']:.3f}")

    # ------------------------------------------------------------- decision-curve redo
    print("[dca ] decision curve with log-spaced low thresholds")
    nb = models.net_benefit(y_te, p_cal_sor)
    usable = [r for r in nb if r["n_flagged"] >= 50]
    wins = [r for r in usable if r["net_benefit_model"] > max(r["net_benefit_treat_all"], 0)]
    R["decision_v2"] = {
        "rows": nb, "truncated_at_n_flagged": 50,
        "useful_range": [wins[0]["threshold"], wins[-1]["threshold"]] if wins else None,
        "note": "Round 3's grid started at 0.01 and claimed usefulness to 0.49 on rows with "
                "<50 flagged fires; this version reaches down to 0.001 (the plausible wildfire "
                "cost-ratio region) and truncates where counts are noise"}
    if wins:
        print(f"       model beats both references for t in "
              f"[{wins[0]['threshold']:.3f}, {wins[-1]['threshold']:.3f}]")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs = [r["threshold"] for r in usable]
    ax.plot(xs, [r["net_benefit_model"] for r in usable], color=COL["fire"], lw=2,
            label="system of record (calibrated)")
    ax.plot(xs, [r["net_benefit_treat_all"] for r in usable], color=COL["muted"], lw=1.4,
            ls="--", label="treat every ignition")
    ax.axhline(0, color=COL["axis"], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("risk threshold (log; encodes the cost ratio)")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision curve, redone: log-spaced thresholds, truncated where n<50",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"])
    style_ax(ax)
    savefig(fig, "m5_decision_curve.png")

    # ------------------------------------------------------------- touch accounting
    R["test_touch_accounting"] = {
        "gated_selection_touches": 9,
        "diagnostic_evaluations_enumerated": {
            "calibration_variants": 3, "threshold_ladder": 4, "ablation": 7,
            "leak_ladder": 5, "by_year": 3, "transfer": 2,
            "grouped_permutation": 40, "single_permutation": 258, "shap": 1, "tubbs": 1},
        "defense": "diagnostics are reported in full and never selected on; the Round-3 "
                   "protocol text ('scored once per family, enforced') described only the "
                   "selection path and has been amended",
        "round4_note": "this red-team runner adds further enumerated diagnostic evaluations "
                       "of the test era (seed refits, ablation refits, terciles); same rule: "
                       "reported in full, nothing selected on them"}

    # ------------------------------------------------------------- figure r1
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    names = ["full model", "no location\n/history", "yesterday\nknowledge",
             "+same-day\nload", "embargoed\nhistory"]
    vals = [baseline_ap, ap_nogeo, ap_yday, ap_sameday, ap_embargo]
    colors = [COL["fire"], COL["gold"], COL["blue"], COL["green"], COL["muted"]]
    axes[0].bar(names, vals, color=colors)
    axes[0].axhline(base_te, color=COL["red"], ls="--", lw=1.2)
    axes[0].text(3.4, base_te * 1.3, "no-skill", fontsize=8, color=COL["red"])
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8.5, color=COL["ink2"])
    axes[0].set_ylabel("test average precision")
    axes[0].set_title(f"Red-team refits (seed sd = {sd_h:.4f})", loc="left", fontsize=10)
    axes[0].tick_params(axis="x", labelsize=8)
    style_ax(axes[0])
    tnames = list(terc)
    axes[1].bar(tnames, [terc[t]["lift"] for t in tnames], color=COL["blue"])
    for i, t in enumerate(tnames):
        axes[1].text(i, terc[t]["lift"] + 0.1, f"{terc[t]['lift']:.1f}x", ha="center",
                     fontsize=8.5, color=COL["ink2"])
    axes[1].axhline(1.0, color=COL["red"], ls="--", lw=1.2)
    axes[1].set_ylabel("AP / prevalence (lift) within tercile")
    axes[1].set_title("Skill survives inside every population tercile", loc="left", fontsize=10)
    style_ax(axes[1])
    fig.tight_layout()
    savefig(fig, "r1_redteam.png")

    # ------------------------------------------------------------- claims
    sor = cal_out["hgb"]
    R["claims"] = {
        "sor_test_ap": round(R["seed_stability"]["hgb"]["mean"], 4),
        "sor_seed_sd": round(sd_h, 4),
        "lgbm_test_ap": round(R["seed_stability"]["lgbm"]["mean"], 4),
        "lgbm_seed_sd": round(sd_g, 4),
        "sor_bss": round(sor["test"]["bss"], 3),
        "sor_slope": round(sor["test"]["slope"], 2),
        "sor_precision_at_1pct": round(sor["precision_at_1pct"], 3),
        "lgbm_precision_at_1pct": round(cal_out["lgbm"]["precision_at_1pct"], 3),
        "no_geography_ap": round(ap_nogeo, 4),
        "yesterday_ap": round(ap_yday, 4),
        "same_day_delta": round(ap_sameday - baseline_ap, 4),
        "embargo_delta": round(ap_embargo - baseline_ap, 4),
        "dup_delta": round(R["dup_sensitivity"]["delta"], 4),
        "leave_2020_out_ap": round(R["concentration_2020"]["leave_2020_out_ap"], 4),
        "share_positives_2020": round(R["concentration_2020"]["share_of_positives_2020"], 3),
        "lift_point": round(R["lift_ci"]["lift_point"], 1),
        "lift_lo": round(R["lift_ci"]["lift_ci"][0], 1),
        "lift_hi": round(R["lift_ci"]["lift_ci"][1], 1),
        "n_blocks": R["lift_ci"]["n_blocks"],
        "nb_t_optuna_p": round(R["earned_nulls"]["optuna_vs_random"]["p"], 3),
        "low_pop_lift": round(terc["low_pop"]["lift"], 1),
        "high_pop_lift": round(terc["high_pop"]["lift"], 1),
        "tubbs_oos_percentile": round(R["tubbs_v2"]["percentile_oos"], 1),
        "tubbs_calibrated_risk": round(R["tubbs_v2"]["calibrated_risk"], 3),
        "published_round1_bss": -3.78,
        "tail_predicted": round(R["tail_reliability"]["mean_predicted"], 3),
        "tail_observed": round(R["tail_reliability"]["observed_rate"], 3),
        "findings_total": 35,
    }
    R["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    (wf.DATA / "redteam_results.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"[done] redteam_results.json in {R['runtime_minutes']} min")

    # Round 5: the Round-3 record must stop silently asserting retracted claims.
    # Originals are preserved verbatim; this only appends the amendment ledger,
    # so a future run_modeling.py re-run (which drops it) is healed by re-running
    # this script. Idempotent.
    amend_round3_record(M)


def amend_round3_record(M: dict) -> None:
    M["protocol_amendments"] = {
        "amended_by": "run_redteam.py (Round 4/5)",
        "authoritative_record": "redteam_results.json",
        "amendments": [
            {"field": "protocol.cv",
             "original": M["protocol"]["cv"],
             "correction": "the search ran GroupKFold(3), not (5); see modeling.html s01"},
            {"field": "protocol.selection",
             "original": M["protocol"]["selection"],
             "correction": "the gate covered the 9 selection scorings only; ~330 diagnostic "
                           "evaluations of the test era ran ungated, reported in full and "
                           "never selected on; see test_touch_accounting"},
            {"field": "champion",
             "original": M["champion"],
             "correction": "selected as a max over 9 test scores (winner's curse). Round 4 "
                           "re-keys the headline to the pre-committed system of record "
                           "hgb_tuned (5-seed test AP 0.1264 +- 0.0023); lightgbm_tuned's "
                           "higher single test AP is reported, not selected (q=0.085)"},
            {"field": "calibration.chosen",
             "original": M["calibration"]["chosen"],
             "correction": "chosen on TEST Brier skill (circular). Re-selected on the "
                           "validation era: val chooses isotonic; see calibration_v2"},
            {"field": "claims.test_roc_auc",
             "original": M["claims"]["test_roc_auc"],
             "correction": "this is lightgbm_tuned's 0.8196, inherited from the "
                           "max-on-test champion. The system of record's test ROC-AUC "
                           "is 0.8168 (zoo.hgb_tuned.roc_auc). Both round to 0.82, so "
                           "the published digit stands, but the provenance was wrong. "
                           "The train/val/test table is in generalization.json"},
        ],
    }
    (wf.DATA / "modeling_results.json").write_text(json.dumps(M, indent=1, default=float))
    print("[done] modeling_results.json: protocol_amendments ledger appended")


if __name__ == "__main__":
    main()
