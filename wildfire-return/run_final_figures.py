"""Round 5: re-render the two calibration-era figures under the val-selected map.

Round 4 re-selected the calibration map on the validation era (isotonic, not the
test-selected sigmoid) and re-keyed sections 07-08 of modeling.html to it, but the
figures m4_calibration.png and m6_precision_at_k.png still plotted the sigmoid-era
curves. This script redraws both from the same state and configs run_redteam.py
used, and hard-asserts that the recomputed metrics match data/redteam_results.json
before writing a single pixel -- a figure must never disagree with the JSON the
verifier enforces.

m6 needs no refit (its four budget points are stored in redteam_results.json);
m4 needs two fits: the seed-42 system of record and the Round-1-style balanced
default it is contrasted against.
"""
from __future__ import annotations

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

import features as fx
import models
import tuning
import wildfire as wf

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
THRESHOLD = 100.0
SEED = 42  # run_redteam.py's first seed; its calibration_v2 block was computed on this fit


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


def main() -> None:
    t0 = time.time()
    M = json.loads((wf.DATA / "modeling_results.json").read_text())
    RT = json.loads((wf.DATA / "redteam_results.json").read_text())
    cal_ref = RT["calibration_v2"]["hgb"]
    assert cal_ref["selected_on_val"] == "isotonic"

    bake = M["bakeoff"]
    hgb_best = bake["tracks"]["A_sklearn_random"]["best"]["params"]
    if bake["tracks"]["B_optuna_tpe"]["best"]["score"] > \
       bake["tracks"]["A_sklearn_random"]["best"]["score"]:
        hgb_best = bake["tracks"]["B_optuna_tpe"]["best"]["params"]

    print("[load] CA with engineered features")
    ca = fx.add_features(wf.load_state_fires("CA", strict=True))
    s = models.temporal_split(ca)
    tr, va, te = ca[s["train"]], ca[s["val"]], ca[s["test"]]
    y_tr = (tr["fire_size"] >= THRESHOLD).to_numpy(int)
    y_va = (va["fire_size"] >= THRESHOLD).to_numpy(int)
    y_te = (te["fire_size"] >= THRESHOLD).to_numpy(int)
    X_tr = fx.build_matrix(tr, "R6_history")
    X_va = fx.build_matrix(va, "R6_history").reindex(columns=X_tr.columns)
    X_te = fx.build_matrix(te, "R6_history").reindex(columns=X_tr.columns)
    prevalence = float(y_te.mean())

    print("[fit ] seed-42 system of record + Round-1-style balanced default")
    h = tuning.make_hgb(hgb_best)
    h.set_params(random_state=SEED)
    h.fit(X_tr, y_tr)
    p_va = h.predict_proba(X_va)[:, 1]
    p_te = h.predict_proba(X_te)[:, 1]
    replica = models.make_honest_clf()  # class_weight="balanced": the Round-1 configuration
    replica.fit(X_tr, y_tr)
    p_bal = replica.predict_proba(X_te)[:, 1]

    print("[cal ] isotonic fitted on the validation era")
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
    p_cal = iso.predict(p_te)

    # The figure must agree with the JSON the verifier enforces, to the digit.
    got = {"bss": models.brier_skill_score(y_te, p_cal), "ece": models.ece(y_te, p_cal),
           "ap": float(average_precision_score(y_te, p_cal))}
    for k in got:
        ref = cal_ref["test"][k]
        if abs(got[k] - ref) > 5e-4:
            print(f"[FAIL] recomputed {k}={got[k]:.6f} != redteam_results {ref:.6f}")
            sys.exit(1)
    print(f"       matches redteam_results.json: BSS {got['bss']:+.4f} "
          f"ECE {got['ece']:.4f} AP {got['ap']:.4f}")

    # m4: reliability + distributions, now on the val-selected isotonic path
    bal_bss = M["zoo"]["hgb_default_balanced"]["bss"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for lab, p, color in (("Round-1 style (raw, balanced)", p_bal, COL["red"]),
                          ("calibrated (val-selected isotonic)", p_cal, COL["green"])):
        frac, mean_p = calibration_curve(y_te, p, n_bins=12, strategy="quantile")
        axes[0].plot(mean_p, frac, marker="o", ms=4, lw=1.7, color=color, label=lab)
    axes[0].plot([0, 1], [0, 1], ls=":", color=COL["axis"], lw=1)
    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("Reliability: the claim Round 1 got backwards", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    style_ax(axes[0], grid_axis="")
    axes[1].hist(p_bal, bins=40, color=COL["red"], alpha=0.55, label="Round-1 (balanced)")
    axes[1].hist(p_cal, bins=40, color=COL["green"], alpha=0.65, label="calibrated (isotonic)")
    axes[1].axvline(prevalence, color=COL["ink2"], ls="--", lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("predicted probability")
    axes[1].set_title(f"Round-1 (balanced) BSS {bal_bss:+.2f}  →  "
                      f"val-selected isotonic {got['bss']:+.3f}", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    style_ax(axes[1])
    fig.tight_layout()
    savefig(fig, "m4_calibration.png")

    # m6: precision/recall @ k from the stored isotonic top-k table -- no refit
    tk = cal_ref["topk"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ks = [t["k"] * 100 for t in tk]
    ax.plot(ks, [t["precision"] for t in tk], color=COL["fire"], lw=2, marker="o",
            label="precision")
    ax.plot(ks, [t["recall"] for t in tk], color=COL["blue"], lw=2, marker="s", label="recall")
    ax.plot(ks, [t["ceiling_recall"] for t in tk], color=COL["blue"], lw=1.2, ls=":",
            label="recall ceiling (perfect ranking)")
    ax.axhline(prevalence, color=COL["red"], ls="--", lw=1.2, label="base rate")
    ax.set_xscale("log")
    ax.set_xlabel("share of ignitions flagged (%)")
    ax.set_title("If you can only act on the top k% (val-selected isotonic), this is what you get",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    style_ax(ax)
    savefig(fig, "m6_precision_at_k.png")

    print(f"[done] both figures re-rendered in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
