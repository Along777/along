from __future__ import annotations

"""Part III redeemed: the 2020 random forest, rebuilt as a museum piece, next to
models evaluated the way 2026 requires.

The museum replica reconstructs the 2020 notebook faithfully -- the same
RandomForest configs, the same random 70/30 split, the dead MONTH/DAY_OF_WEEK
constants (its Julian-date bug made every fire January 1970), and the leaked
FIRE_SIZE features -- on 1992-2015 large fires with the approximate
NWCG -> old-13-class crosswalk. Its scores are then re-measured under a
temporal split, which is where memorized geography goes to die.

The honest models: HistGradientBoosting on at-ignition features only,
train <= 2014 / val 2015-17 / test 2018-20 (the model never sees the Tubbs
era), spatial-block CV inside the train era, and baselines any model must
beat: majority class, a groupby climatology (majority cause per 0.2-degree
cell x month), and a linear model.

Tasks: T1 Natural-vs-Human (CA + FL) - T2 7-class cause (CA) - T3 P(>=100 ac)
(CA). Falsifications: leakage probe, split-protocol gap, label shuffle.

Outputs: data/ml_lab_results.json + g1-g5 and f4-f6 figures.
Run (offline):  python run_ml_lab.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

import models
import wildfire as wf

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}

# The feature contract lives in wildfire.py (the data stage's formal handoff);
# this lab consumes it. At-ignition only; leaks banned by name.
NUM_FEATURES = wf.ML_FEATURES_NUM
CAT_FEATURES = wf.ML_FEATURES_CAT
FEATURES = NUM_FEATURES + CAT_FEATURES
EXCLUDED_LEAKS = wf.ML_EXCLUDED_LEAKS


def style_ax(ax, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COL["axis"])
    ax.tick_params(colors=COL["ink2"], labelsize=9)
    ax.xaxis.label.set_color(COL["ink2"])
    ax.yaxis.label.set_color(COL["ink2"])
    ax.title.set_color(COL["ink"])


def savefig(fig, name: str) -> None:
    wf.FIG.mkdir(exist_ok=True)
    fig.savefig(wf.FIG / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig ] {name}")


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    return X


def climatology_grid(train: pd.DataFrame, test: pd.DataFrame, label: pd.Series,
                     label_test_index: pd.Index) -> pd.Series:
    """Majority label per 0.2-degree cell x month -- the groupby bar."""
    key_tr = pd.DataFrame({
        "la": np.floor(train["lat"] * 5) / 5, "lo": np.floor(train["lon"] * 5) / 5,
        "m": train["month"], "y": label.loc[train.index]})
    lookup = key_tr.groupby(["la", "lo", "m"])["y"].agg(lambda s: s.mode().iloc[0])
    global_maj = label.loc[train.index].mode().iloc[0]
    key_te = pd.MultiIndex.from_arrays([np.floor(test["lat"] * 5) / 5,
                                        np.floor(test["lon"] * 5) / 5, test["month"]])
    pred = pd.Series(lookup.reindex(key_te).values, index=label_test_index)
    return pred.fillna(global_maj)


def logistic_baseline(Xtr, ytr, Xte) -> np.ndarray:
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(Xtr[NUM_FEATURES], ytr)
    return pipe.predict(Xte[NUM_FEATURES])


def run_task(df: pd.DataFrame, label_col: str, task_name: str, R: dict,
             binary_pos: str | None = None) -> tuple:
    """Temporal-protocol fit + baselines + spatial CV for one task. Returns model+splits."""
    d = df[df[label_col].notna()].copy()
    excluded_share = 1 - len(d) / len(df)
    splits = models.temporal_split(d)
    tr, va, te = d[splits["train"]], d[splits["val"]], d[splits["test"]]
    Xtr, Xva, Xte = feature_frame(tr), feature_frame(va), feature_frame(te)
    ytr, yva, yte = tr[label_col], va[label_col], te[label_col]
    labels = sorted(ytr.unique())

    clf = models.make_honest_clf()
    clf.fit(Xtr, ytr)
    pred_te = clf.predict(Xte)
    proba_te = clf.predict_proba(Xte)[:, list(clf.classes_).index(binary_pos)] if binary_pos else None

    maj = pd.Series(ytr.mode().iloc[0], index=yte.index)
    clim = climatology_grid(tr, te, d[label_col], yte.index)
    logi = logistic_baseline(Xtr, ytr, Xte)

    # spatial-block CV inside the training era (protocol stability, no test leakage)
    blocks = models.spatial_blocks(tr)
    cv_scores = []
    gkf = GroupKFold(n_splits=5)
    for tr_i, te_i in gkf.split(Xtr, ytr, groups=blocks):
        m = models.make_honest_clf()
        m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        cv_scores.append(models.metrics_table(ytr.iloc[te_i], m.predict(Xtr.iloc[te_i]))["macro_f1"])

    R[task_name] = {
        "n_rows_known_label": int(len(d)),
        "label_excluded_share": round(float(excluded_share), 4),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "labels": [str(x) for x in labels],
        "test": models.metrics_table(yte, pred_te, proba_te, labels=labels),
        "baseline_majority": models.metrics_table(yte, maj, labels=labels),
        "baseline_climatology": models.metrics_table(yte, clim, labels=labels),
        "baseline_logistic": models.metrics_table(yte, logi, labels=labels),
        "spatial_cv_macro_f1_mean": round(float(np.mean(cv_scores)), 4),
        "spatial_cv_macro_f1_sd": round(float(np.std(cv_scores)), 4),
    }
    print(f"  [task] {task_name}: test macro-F1 {R[task_name]['test']['macro_f1']:.3f} "
          f"(majority {R[task_name]['baseline_majority']['macro_f1']:.3f}, "
          f"climatology {R[task_name]['baseline_climatology']['macro_f1']:.3f}, "
          f"logistic {R[task_name]['baseline_logistic']['macro_f1']:.3f})")
    return clf, d, splits, Xte, yte, pred_te, proba_te


def main() -> None:
    for f in ("ca_fires.parquet", "fl_fires.parquet"):
        wf.verify_manifest(f, strict=True)
    ca = wf.load_state_fires("CA")
    fl = wf.load_state_fires("FL")
    R: dict = {"features": FEATURES, "excluded_as_leaks": EXCLUDED_LEAKS}

    # ------------------------------------------------------------------ museum piece
    old = ca[(ca["fire_year"] <= 2015) & ca["size_class"].isin(wf.LARGE_CLASSES)].copy()
    old = old[old["old13"].notna()]
    # the 2020 feature set, bugs preserved: dead date features + leaked size
    le_size = LabelEncoder().fit(old["size_class"].astype(str))
    X_mus = pd.DataFrame({
        "LATITUDE": old["lat"], "LONGITUDE": old["lon"], "FIRE_YEAR": old["fire_year"],
        "MONTH": 1, "DAY_OF_WEEK": 3,                      # constants: the Julian bug
        "FIRE_SIZE": old["fire_size"],                     # leaked outcome
        "FIRE_SIZE_CLASS": le_size.transform(old["size_class"].astype(str)),
    })
    y13 = old["old13"]
    Xa, Xb, ya, yb = train_test_split(X_mus, y13, test_size=0.3, random_state=42)
    rf13 = models.museum_replica(50).fit(Xa, ya)
    acc13_random = float(rf13.score(Xb, yb))
    # the same museum model under a temporal split
    tr_m = old["fire_year"] <= 2010
    rf13_t = models.museum_replica(50).fit(X_mus[tr_m], y13[tr_m])
    m13_t = models.metrics_table(y13[~tr_m], rf13_t.predict(X_mus[~tr_m]))
    # the campfire binary, as celebrated
    y_camp = (old["old13"] == "Campfire")
    Xa, Xb, ya, yb = train_test_split(X_mus, y_camp, test_size=0.3, random_state=42)
    rf_camp = models.museum_replica(200).fit(Xa, ya)
    acc_camp_random = float(rf_camp.score(Xb, yb))
    camp_majority = float((~yb).mean())
    R["museum"] = {
        "n_large_fires_1992_2015": int(len(old)),
        "crosswalk_note": "old 13 classes reconstructed via approximate NWCG crosswalk",
        "acc13_random_split": round(acc13_random, 4),
        "acc13_2020_notebook": 0.6133,
        "m13_temporal": m13_t,
        "acc_campfire_random_split": round(acc_camp_random, 4),
        "acc_campfire_2020_notebook": 0.9478,
        "campfire_majority_share_testset": round(camp_majority, 4),
    }
    print(f"  [muse] 13-class random-split acc {acc13_random:.3f} (2020 notebook: 0.613); "
          f"temporal macro-F1 {m13_t['macro_f1']:.3f}")
    print(f"  [muse] campfire random-split acc {acc_camp_random:.3f} (2020: 0.948) vs "
          f"majority {camp_majority:.3f}")

    # ------------------------------------------------------------------ honest tasks
    ca["binary_cause"] = ca["cause_class"].where(ca["cause_class"].isin(["Human", "Natural"]))
    fl["binary_cause"] = fl["cause_class"].where(fl["cause_class"].isin(["Human", "Natural"]))
    clf1, d1, sp1, Xte1, yte1, pred1, proba1 = run_task(ca, "binary_cause", "T1_ca_natural_vs_human",
                                                        R, binary_pos="Natural")
    run_task(fl, "binary_cause", "T1_fl_natural_vs_human", R, binary_pos="Natural")
    clf2, d2, sp2, Xte2, yte2, pred2, _ = run_task(ca, "cause_group", "T2_ca_7class", R)
    ca["is_100ac"] = np.where(ca["fire_size"].notna(), (ca["fire_size"] >= 100).astype(str), None)
    clf3, d3, sp3, Xte3, yte3, pred3, proba3 = run_task(ca, "is_100ac", "T3_ca_ge100ac",
                                                        R, binary_pos="True")

    # ------------------------------------------------------------------ falsifications
    # f4: put the leaked feature back -> watch the score inflate (the 2020 sin, live)
    d = d1
    tr, te = d[sp1["train"]], d[sp1["test"]]
    X_leak_tr = feature_frame(tr).assign(fire_size=tr["fire_size"])
    X_leak_te = feature_frame(te).assign(fire_size=te["fire_size"])
    leak_clf = models.make_honest_clf().fit(X_leak_tr, tr["binary_cause"])
    leak_f1 = models.metrics_table(te["binary_cause"], leak_clf.predict(X_leak_te))["macro_f1"]
    R["falsification_leak_probe"] = {
        "honest_macro_f1": R["T1_ca_natural_vs_human"]["test"]["macro_f1"],
        "with_leaked_fire_size_macro_f1": round(float(leak_f1), 4),
    }

    # f5: protocol gap on the SAME honest model/features: random CV vs spatial CV vs temporal test
    tr_all = d[sp1["train"] | sp1["val"]]
    X_all, y_all = feature_frame(tr_all), tr_all["binary_cause"]
    rand_scores, spat_scores = [], []
    blocks = models.spatial_blocks(tr_all)
    for tr_i, te_i in GroupKFold(5).split(X_all, y_all, groups=blocks):
        m = models.make_honest_clf().fit(X_all.iloc[tr_i], y_all.iloc[tr_i])
        spat_scores.append(models.metrics_table(y_all.iloc[te_i], m.predict(X_all.iloc[te_i]))["macro_f1"])
    rng = np.random.default_rng(models.RNG)
    idx = rng.permutation(len(X_all))
    for k in range(5):
        te_i = idx[k::5]
        tr_i = np.setdiff1d(idx, te_i)
        m = models.make_honest_clf().fit(X_all.iloc[tr_i], y_all.iloc[tr_i])
        rand_scores.append(models.metrics_table(y_all.iloc[te_i], m.predict(X_all.iloc[te_i]))["macro_f1"])
    R["falsification_protocol_gap"] = {
        "random_cv_macro_f1": round(float(np.mean(rand_scores)), 4),
        "spatial_cv_macro_f1": round(float(np.mean(spat_scores)), 4),
        "temporal_test_macro_f1": R["T1_ca_natural_vs_human"]["test"]["macro_f1"],
    }

    # f6: label shuffle -> the null every pipeline must fail against
    y_shuf = y_all.sample(frac=1.0, random_state=models.RNG).reset_index(drop=True)
    m = models.make_honest_clf().fit(X_all.reset_index(drop=True), y_shuf)
    te = d[sp1["test"]]
    shuf_f1 = models.metrics_table(te["binary_cause"], m.predict(feature_frame(te)))["macro_f1"]
    R["falsification_label_shuffle"] = {"macro_f1": round(float(shuf_f1), 4)}

    # ------------------------------------------------------------------ figures
    # g1: 7-class confusion (row-normalized)
    labels2 = sorted(yte2.unique())
    cm = models.confusion(yte2, pred2, labels2)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(cm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels2)))
    ax.set_xticklabels(labels2, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels2)))
    ax.set_yticklabels(labels2, fontsize=8)
    for i in range(len(labels2)):
        for j in range(len(labels2)):
            v = cm[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.5 else COL["ink2"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("7-class cause, temporal test 2018-2020 (rows sum to 1)", loc="left", fontsize=11)
    style_ax(ax, grid_axis="")
    savefig(fig, "g1_confusion.png")
    R["T2_confusion"] = {"labels": [str(x) for x in labels2], "matrix": cm}

    # g2: the museum vs honesty bar chart
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars = [
        ("2020 campfire model\nrandom split (their 94.7%)", acc_camp_random, COL["muted"]),
        ("...majority baseline\n(guess 'not campfire')", camp_majority, COL["axis"]),
        ("2020 13-class model\nrandom split (their 61.3%)", acc13_random, COL["muted"]),
        ("same model,\ntemporal split (accuracy)", m13_t["accuracy"], COL["gold"]),
        ("2026 honest binary,\ntemporal test (macro-F1)", R["T1_ca_natural_vs_human"]["test"]["macro_f1"],
         COL["fire"]),
    ]
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=COL["ink2"])
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.6)
    ax.set_ylim(0, 1.05)
    ax.set_title("What the 94.7% was actually made of", loc="left", fontsize=11)
    style_ax(ax)
    savefig(fig, "g2_museum_vs_honest.png")

    # g3: permutation importance (T1 CA, temporal test)
    print("  [perm] computing permutation importance (T1 CA test)...")
    pi = permutation_importance(clf1, Xte1, yte1, n_repeats=5, random_state=models.RNG,
                                scoring="f1_macro")
    order = np.argsort(pi.importances_mean)[-15:]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh([FEATURES[i] for i in order], pi.importances_mean[order],
            xerr=pi.importances_std[order], color=COL["fire"], error_kw={"ecolor": COL["axis"]})
    ax.set_xlabel("drop in macro-F1 when shuffled (temporal test)")
    ax.set_title("What the honest model actually uses (correlated weather features share credit)",
                 loc="left", fontsize=11)
    style_ax(ax, grid_axis="x")
    savefig(fig, "g3_permutation_importance.png")
    R["T1_permutation_importance_top"] = {FEATURES[i]: round(float(pi.importances_mean[i]), 4)
                                          for i in order[::-1]}

    # g4: PDP for top numeric features
    top_num = [FEATURES[i] for i in order[::-1] if FEATURES[i] in NUM_FEATURES][:4]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    PartialDependenceDisplay.from_estimator(clf1, Xte1, top_num, ax=ax, n_cols=4,
                                            line_kw={"color": COL["fire"], "lw": 1.8})
    for a in fig.axes:
        style_ax(a, grid_axis="")
        a.set_ylabel(a.get_ylabel(), fontsize=7)
    fig.suptitle("P(Natural) along the top features, everything else held as-is",
                 x=0.02, ha="left", fontsize=11, color=COL["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, "g4_pdp_top.png")

    # g5: calibration (T3 large-fire risk)
    frac_pos, mean_pred = calibration_curve((yte3 == "True").astype(int), proba3, n_bins=12,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], color=COL["axis"], lw=1, ls=":")
    ax.plot(mean_pred, frac_pos, color=COL["fire"], lw=1.8, marker="o", ms=4)
    ax.set_xlabel("predicted P(fire reaches 100+ acres)")
    ax.set_ylabel("observed share")
    ax.set_title(f"Growth-risk probabilities are honest\n(Brier {R['T3_ca_ge100ac']['test']['brier']:.4f}, "
                 f"PR-AUC {R['T3_ca_ge100ac']['test']['pr_auc']:.3f})", loc="left", fontsize=10)
    style_ax(ax, grid_axis="")
    savefig(fig, "g5_calibration.png")

    # f4/f5/f6 falsification chart
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    fal = [
        ("label shuffle\n(null)", R["falsification_label_shuffle"]["macro_f1"], COL["axis"]),
        ("temporal test\n(honest)", R["T1_ca_natural_vs_human"]["test"]["macro_f1"], COL["fire"]),
        ("spatial CV", R["falsification_protocol_gap"]["spatial_cv_macro_f1"], COL["gold"]),
        ("random CV\n(2020's protocol)", R["falsification_protocol_gap"]["random_cv_macro_f1"], COL["muted"]),
        ("+ leaked FIRE_SIZE\n(2020's feature)", R["falsification_leak_probe"]["with_leaked_fire_size_macro_f1"],
         COL["red"]),
    ]
    ax.bar(range(len(fal)), [x[1] for x in fal], color=[x[2] for x in fal], width=0.6)
    for i, (_, v, _) in enumerate(fal):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=COL["ink2"])
    ax.set_xticks(range(len(fal)))
    ax.set_xticklabels([x[0] for x in fal], fontsize=8)
    ax.set_ylabel("macro-F1 (Natural vs Human, CA)")
    ax.set_title("Same data, five protocols: how evaluation choices manufacture scores",
                 loc="left", fontsize=11)
    style_ax(ax)
    savefig(fig, "f4_protocols_and_leaks.png")

    # ------------------------------------------------------------------ claims
    t1 = R["T1_ca_natural_vs_human"]
    R["claims"] = {
        "museum_campfire_acc": round(acc_camp_random, 3),
        "museum_campfire_majority": round(camp_majority, 3),
        "t1_ca_test_macro_f1": t1["test"]["macro_f1"],
        "t1_ca_test_pr_auc": t1["test"]["pr_auc"],
        "t1_ca_climatology_macro_f1": t1["baseline_climatology"]["macro_f1"],
        "t2_ca_test_macro_f1": R["T2_ca_7class"]["test"]["macro_f1"],
        "t2_ca_majority_macro_f1": R["T2_ca_7class"]["baseline_majority"]["macro_f1"],
        "t3_pr_auc": R["T3_ca_ge100ac"]["test"]["pr_auc"],
        "t3_brier": R["T3_ca_ge100ac"]["test"]["brier"],
        "protocol_random_cv": R["falsification_protocol_gap"]["random_cv_macro_f1"],
        "protocol_spatial_cv": R["falsification_protocol_gap"]["spatial_cv_macro_f1"],
        "protocol_temporal": R["falsification_protocol_gap"]["temporal_test_macro_f1"],
        "leak_probe_f1": R["falsification_leak_probe"]["with_leaked_fire_size_macro_f1"],
        "label_shuffle_f1": R["falsification_label_shuffle"]["macro_f1"],
        "t1_label_excluded_share": t1["label_excluded_share"],
    }
    (wf.DATA / "ml_lab_results.json").write_text(json.dumps(R, indent=2, default=float))
    print("[done] ml_lab_results.json + 6 figures")


if __name__ == "__main__":
    main()
