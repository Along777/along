"""Round 6: measure generalization directly, and bound the ignition question.

Two gaps this closes.

(1) Nothing in this project has ever scored the model on its own training era.
Every published anti-overfitting argument is indirect: calibration slope, seed
stability, ablation flatness, cross-state transfer. The direct measurement is a
train/val/test table, and it costs one fit. This script produces it, for the
system of record only, at seed 42, and hard-asserts the three cells that already
exist in the stored JSONs before writing anything.

(2) A label-shuffle null. run_ml_lab.py shuffles labels for the CAUSE task, so
that pipeline has been shown it can fail. The escalation model never has.

It also counts the space-time facts that decide whether "predict fires before
they happen" is answerable from these caches. It is not, and the numbers here
say why in a form the verifier can enforce: every cache row is a fire that
happened, so a model of ignition OCCURRENCE has no negatives to learn from.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import features as fx
import models
import tuning
import wildfire as wf

THRESHOLD = 100.0
SEED = 42          # the seed every stored single-fit number was computed at
HISTORY_CELL_DEG = 0.25   # matches features.HISTORY_CELL_DEG


def scores(y, p) -> dict:
    return {"ap": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "base_rate": float(np.mean(y)),
            "n": int(len(y)),
            "positives": int(y.sum())}


def main() -> None:
    t0 = time.time()
    M = json.loads((wf.DATA / "modeling_results.json").read_text())
    RT = json.loads((wf.DATA / "redteam_results.json").read_text())

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

    print("[fit ] system of record, seed 42")
    h = tuning.make_hgb(hgb_best)
    h.set_params(random_state=SEED)
    h.fit(X_tr, y_tr)

    gen = {"train": scores(y_tr, h.predict_proba(X_tr)[:, 1]),
           "val": scores(y_va, h.predict_proba(X_va)[:, 1]),
           "test": scores(y_te, h.predict_proba(X_te)[:, 1])}
    for era in gen:
        gen[era]["lift"] = gen[era]["ap"] / gen[era]["base_rate"]

    # Nothing is written unless the three cells that already exist agree with
    # what the verifier enforces elsewhere.
    anchors = [("test ap", gen["test"]["ap"], M["zoo"]["hgb_tuned"]["ap"]),
               ("test roc_auc", gen["test"]["roc_auc"], M["zoo"]["hgb_tuned"]["roc_auc"]),
               ("val ap", gen["val"]["ap"],
                RT["subsample_transfer_check"]["val_ap"]["hgb_tuned"])]
    for name, got, ref in anchors:
        if abs(got - ref) > 5e-4:
            print(f"[FAIL] recomputed {name}={got:.6f} != stored {ref:.6f}")
            sys.exit(1)
    print(f"       anchors match: test AP {gen['test']['ap']:.4f} "
          f"ROC-AUC {gen['test']['roc_auc']:.4f} | val AP {gen['val']['ap']:.4f}")
    print(f"       train AP {gen['train']['ap']:.4f} "
          f"ROC-AUC {gen['train']['roc_auc']:.4f} (new)")

    print("[null] refit on shuffled training labels")
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y_tr)
    hs = tuning.make_hgb(hgb_best)
    hs.set_params(random_state=SEED)
    hs.fit(X_tr, y_shuf)
    p_null = hs.predict_proba(X_te)[:, 1]
    null = {"test_ap": float(average_precision_score(y_te, p_null)),
            "test_roc_auc": float(roc_auc_score(y_te, p_null)),
            "test_base_rate": float(np.mean(y_te))}
    null["lift"] = null["test_ap"] / null["test_base_rate"]
    print(f"       shuffled-label AP {null['test_ap']:.4f} "
          f"(base rate {null['test_base_rate']:.4f}, lift {null['lift']:.2f}x) "
          f"ROC-AUC {null['test_roc_auc']:.3f}")

    # ---- what an ignition-occurrence model would be up against
    print("[grid] space-time accounting for the ignition question")
    dates = ca["discovery_date"]
    span_days = int((dates.max() - dates.min()).days) + 1
    days_with_fire = int(dates.dt.normalize().nunique())
    cells = models.spatial_blocks(ca, deg=HISTORY_CELL_DEG).nunique()
    cell_days = int(cells) * span_days
    ignition = {
        "n_fires": int(len(ca)),
        "span_days": span_days,
        "days_with_at_least_one_fire": days_with_fire,
        "share_of_days_with_fire": days_with_fire / span_days,
        "cells_deg": HISTORY_CELL_DEG,
        "cells_with_fire_history": int(cells),
        "cell_day_universe": cell_days,
        "ignition_rate_per_cell_day": len(ca) / cell_days,
        "escalation_base_rate_test": float(np.mean(y_te)),
        "missing_negative_cell_days": cell_days - int(len(ca)),
        "note": "cells are 0.25deg squares that recorded at least one fire in "
                "29 years. Rarity is NOT the obstacle: an ignition is about as "
                "common per cell-day as an escalation is per reported fire. The "
                "obstacle is that the negative rows do not exist in any cache, "
                "and there is no gridded daily weather with which to describe "
                "them, so they cannot be constructed either.",
    }
    print(f"       {days_with_fire:,} of {span_days:,} days had >=1 CA ignition "
          f"({100 * ignition['share_of_days_with_fire']:.1f}%)")
    print(f"       {cells:,} cells x {span_days:,} days = {cell_days:,} cell-days; "
          f"ignition rate {ignition['ignition_rate_per_cell_day']:.4f} per cell-day "
          f"vs escalation {ignition['escalation_base_rate_test']:.4f} per reported fire")
    print(f"       {ignition['missing_negative_cell_days']:,} negative cell-days "
          f"do not exist in any cache")

    R = {
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "system_of_record": "hgb_tuned",
        "seed": SEED,
        "generalization": gen,
        "shuffled_label_null": null,
        "ignition_question": ignition,
        "claims": {
            "train_ap": round(gen["train"]["ap"], 3),
            "train_auc": round(gen["train"]["roc_auc"], 3),
            "val_ap": round(gen["val"]["ap"], 3),
            "val_auc": round(gen["val"]["roc_auc"], 3),
            "test_ap": round(gen["test"]["ap"], 3),
            "test_auc": round(gen["test"]["roc_auc"], 3),
            "train_test_ap_gap": round(gen["train"]["ap"] - gen["test"]["ap"], 3),
            "null_ap": round(null["test_ap"], 4),
            "null_auc": round(null["test_roc_auc"], 3),
            "days_with_fire_pct": round(100 * ignition["share_of_days_with_fire"], 1),
            "cell_days_millions": round(cell_days / 1e6, 1),
            "missing_negatives_millions": round(
                ignition["missing_negative_cell_days"] / 1e6, 1),
            "ignition_rate_per_cell_day": round(
                ignition["ignition_rate_per_cell_day"], 3),
            "train_lift": round(gen["train"]["lift"], 1),
            "test_lift": round(gen["test"]["lift"], 1),
            "val_lift": round(gen["val"]["lift"], 1),
        },
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (wf.DATA / "generalization.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"[done] generalization.json in {R['runtime_minutes']} min")


if __name__ == "__main__":
    main()
