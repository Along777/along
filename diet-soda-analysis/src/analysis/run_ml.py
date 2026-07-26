"""ML + hyperparameter search for myth M8 and predictive ΔR² tests."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.config import get_paths, load_config

warnings.filterwarnings("ignore")
RNG = 42


def run_ml(cfg=None) -> None:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    df = pd.read_parquet(paths["processed"] / "analysis_ready.parquet")

    # --- Task A: classify ASB-only vs Neither from health profile (reverse story) ---
    cls = df[df["bev_group"].isin(["ASB-only", "Neither"])].copy()
    cls["y"] = (cls["bev_group"] == "ASB-only").astype(int)
    feat_cls = ["age", "female", "pir", "education", "bmi", "waist", "hba1c", "sbp_mean", "hdl", "tg", "diabetes_sr", "smoking_status", "total_kcal_d1"]
    feat_cls = [c for c in feat_cls if c in cls.columns]
    Xc = cls[feat_cls]
    yc = cls["y"]
    wc = cls["w_mec"].fillna(1.0).values

    pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    pipe_lr = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RNG)),
        ]
    )
    # HistGBM handles NaN natively
    hgb = HistGradientBoostingClassifier(random_state=RNG)
    param = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_iter": [100, 200],
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RNG)
    search = RandomizedSearchCV(
        hgb,
        param_distributions=param,
        n_iter=12,
        scoring="roc_auc",
        cv=cv,
        random_state=RNG,
        n_jobs=-1,
    )
    # Fit weighted HGB via sample_weight
    Xc_np = Xc.to_numpy()
    search.fit(Xc_np, yc, sample_weight=wc)
    best = search.best_estimator_
    # CV AUC weighted model
    proba = cross_val_predict(best, Xc_np, yc, cv=cv, method="predict_proba")[:, 1]
    auc_w = roc_auc_score(yc, proba, sample_weight=wc)
    # Unweighted fit comparison
    search_u = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=RNG),
        param_distributions=param,
        n_iter=8,
        scoring="roc_auc",
        cv=cv,
        random_state=RNG,
        n_jobs=-1,
    )
    search_u.fit(Xc_np, yc)
    proba_u = cross_val_predict(search_u.best_estimator_, Xc_np, yc, cv=cv, method="predict_proba")[:, 1]
    auc_u = roc_auc_score(yc, proba_u)

    best.fit(Xc_np, yc, sample_weight=wc)
    imp = permutation_importance(best, Xc_np, yc, n_repeats=8, random_state=RNG, scoring="roc_auc")
    imp_df = pd.DataFrame({"feature": feat_cls, "importance_mean": imp.importances_mean}).sort_values(
        "importance_mean", ascending=False
    )
    imp_df.to_csv(paths["tables"] / "feature_importance_asb_classifier.csv", index=False)

    # --- Task B: predict BMI with/without ASB features ---
    reg = df.dropna(subset=["bmi"]).copy()
    base_feats = ["age", "female", "pir", "education", "smoking_status", "total_kcal_d1", "diabetes_sr"]
    base_feats = [c for c in base_feats if c in reg.columns]
    asb_feats = base_feats + ["asb_any_d1", "asb_serv_d1", "ssb_any_d1", "ssb_serv_d1"]
    yb = reg["bmi"].values
    wb = reg["w_mec"].fillna(1.0).values

    def _r2(feats):
        X = reg[feats].to_numpy()
        model = HistGradientBoostingRegressor(random_state=RNG, max_depth=5, learning_rate=0.1)
        # simple holdout by cycle
        mask_te = reg["cycle"] == "2017-2018"
        if mask_te.sum() < 100:
            mask_te = reg["cycle"] == reg["cycle"].iloc[-1]
        model.fit(X[~mask_te], yb[~mask_te], sample_weight=wb[~mask_te])
        pred = model.predict(X[mask_te])
        return float(r2_score(yb[mask_te], pred, sample_weight=wb[mask_te]))

    r2_base = _r2(base_feats)
    r2_asb = _r2(asb_feats)

    # plot importance
    try:
        import matplotlib.pyplot as plt
        from src.analysis.plot_style import apply_style

        apply_style()
        top = imp_df.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["feature"], top["importance_mean"], color="#1f77b4")
        ax.set_title("M8: What predicts diet-soda use? (permutation importance)")
        ax.set_xlabel("Δ ROC-AUC")
        fig.savefig(paths["figures"] / "myth_m8_asb_feature_importance.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print("plot fail", exc)

    summary = {
        "asb_classifier_auc_weighted": auc_w,
        "asb_classifier_auc_unweighted": auc_u,
        "best_params": search.best_params_,
        "top_features": imp_df.head(8).to_dict(orient="records"),
        "bmi_r2_without_asb": r2_base,
        "bmi_r2_with_asb": r2_asb,
        "bmi_delta_r2": r2_asb - r2_base,
        "n_classifier": int(len(cls)),
        "n_asb_pos": int(yc.sum()),
    }
    with open(paths["tables"] / "ml_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    run_ml()


if __name__ == "__main__":
    main()
