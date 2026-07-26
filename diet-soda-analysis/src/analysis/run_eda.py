"""EDA myth dashboard — generate core figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.plot_style import BEV_COLORS, BEV_ORDER, apply_style
from src.data.config import get_paths, load_config


def _load() -> pd.DataFrame:
    paths = get_paths()
    return pd.read_parquet(paths["processed"] / "analysis_ready.parquet")


def run_eda(cfg=None) -> None:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    fig_dir = paths["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    df = _load()

    # 1 Who drinks what by sex
    fig, ax = plt.subplots(figsize=(9, 5))
    ct = pd.crosstab(df["female"].map({0: "Male", 1: "Female"}), df["bev_group"], normalize="index") * 100
    ct = ct.reindex(columns=[c for c in BEV_ORDER if c in ct.columns])
    ct.plot(kind="bar", stacked=True, ax=ax, color=[BEV_COLORS[c] for c in ct.columns])
    ax.set_ylabel("% within sex")
    ax.set_xlabel("")
    ax.set_title("Myth M5 prelude: beverage group by sex (analytic sample, unweighted %)")
    ax.legend(title="Beverage", bbox_to_anchor=(1.02, 1))
    fig.savefig(fig_dir / "myth_m5_bev_by_sex.png")
    plt.close(fig)

    # 2 Exposure distribution among ASB consumers
    cons = df.loc[df["asb_any_d1"] == 1, "asb_serv_d1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(cons, bins=30, ax=ax, color=BEV_COLORS["ASB-only"])
    ax.set_xlabel("Diet soft drink servings (Day-1, 355g units)")
    ax.set_title("M6: Dose among diet soft drink consumers (zero-excluded)")
    fig.savefig(fig_dir / "myth_m6_asb_dose_hist.png")
    plt.close(fig)

    # 3 Violins for BMI, HbA1c, SBP
    for outcome, title, fname in [
        ("bmi", "M1: BMI by beverage group (crude)", "myth_m1_bmi_violin.png"),
        ("hba1c", "M2: HbA1c by beverage group (crude)", "myth_m2_hba1c_violin.png"),
        ("sbp_mean", "M3: Mean SBP by beverage group (crude)", "myth_m3_sbp_violin.png"),
    ]:
        plot_df = df.dropna(subset=[outcome, "bev_group"])
        plot_df = plot_df[plot_df["bev_group"].isin(BEV_ORDER)]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(
            data=plot_df,
            x="bev_group",
            y=outcome,
            order=[c for c in BEV_ORDER if c in plot_df["bev_group"].unique()],
            palette=BEV_COLORS,
            cut=0,
            ax=ax,
        )
        ax.set_title(title + " — unweighted")
        ax.set_xlabel("")
        fig.savefig(fig_dir / fname)
        plt.close(fig)

    # 4 Love plot-ish SMD ASB-only vs Neither
    a = df[df["bev_group"] == "ASB-only"]
    n = df[df["bev_group"] == "Neither"]
    covs = {
        "age": "Age",
        "female": "Female",
        "pir": "PIR",
        "bmi": "BMI",
        "diabetes_sr": "Diabetes (SR)",
        "total_kcal_d1": "Energy (kcal)",
        "smoking_status": "Smoking code",
    }
    smds = []
    for col, lab in covs.items():
        if col not in df.columns:
            continue
        xa, xn = a[col].dropna(), n[col].dropna()
        if len(xa) < 30 or len(xn) < 30:
            continue
        pooled = np.sqrt((xa.var(ddof=1) + xn.var(ddof=1)) / 2)
        smd = (xa.mean() - xn.mean()) / pooled if pooled > 0 else 0
        smds.append({"covariate": lab, "smd": smd})
    smd_df = pd.DataFrame(smds).sort_values("smd")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(0, color="black", lw=1)
    ax.scatter(smd_df["smd"], smd_df["covariate"], s=80, color=BEV_COLORS["ASB-only"])
    ax.set_xlabel("Standardized mean difference (ASB-only − Neither)")
    ax.set_title("M5: Confounder imbalance — who drinks diet soda?")
    fig.savefig(fig_dir / "myth_m5_smd_loveplot.png")
    plt.close(fig)
    smd_df.to_csv(paths["tables"] / "smd_asb_vs_neither.csv", index=False)

    # 5 Reverse causation scatter
    sub = df.dropna(subset=["asb_serv_d1", "bmi"]).copy()
    sub = sub[sub["asb_serv_d1"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for val, lab, col in [(0, "No diabetes", "#2ca02c"), (1, "Diabetes", "#d62728")]:
        part = sub[sub["diabetes_sr"] == val]
        if len(part) == 0:
            continue
        ax.scatter(part["asb_serv_d1"], part["bmi"], alpha=0.25, s=12, label=lab, c=col)
    ax.set_xlabel("Diet soft servings (Day-1)")
    ax.set_ylabel("BMI")
    ax.set_title("M1/M2 reverse-causation visual: ASB dose vs BMI by diabetes")
    ax.legend()
    fig.savefig(fig_dir / "myth_m1_reverse_causation_scatter.png")
    plt.close(fig)

    # 6 Cycle stability crude mean BMI
    tab = (
        df.groupby(["cycle", "bev_group"], observed=True)["bmi"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for g in BEV_ORDER:
        part = tab[tab["bev_group"] == g]
        if len(part):
            ax.plot(part["cycle"], part["bmi"], marker="o", label=g, color=BEV_COLORS[g])
    ax.set_title("Cycle stability: crude mean BMI by beverage group")
    ax.set_ylabel("Mean BMI")
    ax.legend()
    fig.savefig(fig_dir / "myth_m7_cycle_stability_bmi.png")
    plt.close(fig)

    # 7 Cancer crude rates by group
    ctab = df.groupby("bev_group", observed=True).agg(
        n=("cancer_ever", "size"),
        cancer_rate=("cancer_ever", "mean"),
        mean_age=("age", "mean"),
    ).reindex(BEV_ORDER)
    ctab.to_csv(paths["tables"] / "cancer_crude_by_group.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = ctab["cancer_rate"] * 100
    ax.bar(rates.index.astype(str), rates.values, color=[BEV_COLORS[i] for i in rates.index])
    ax.set_ylabel("Ever-cancer % (crude, unweighted)")
    ax.set_title("M4: Crude self-reported cancer by beverage group (age confounding likely)")
    fig.savefig(fig_dir / "myth_m4_cancer_crude.png")
    plt.close(fig)

    # 8 Correlation heatmap
    cols = [
        c
        for c in [
            "asb_serv_d1",
            "ssb_serv_d1",
            "age",
            "bmi",
            "waist",
            "hba1c",
            "sbp_mean",
            "hdl",
            "tg",
            "total_kcal_d1",
            "pir",
        ]
        if c in df.columns
    ]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Spearman correlations (analytic sample)")
    fig.savefig(fig_dir / "myth_corr_heatmap.png")
    plt.close(fig)

    # 9 Missingness
    miss = df[["bmi", "hba1c", "sbp_mean", "glucose", "tg", "cancer_ever", "insulin"]].isna().mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    miss.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_xlabel("% missing")
    ax.set_title("Missingness of key outcomes (analytic sample)")
    fig.savefig(fig_dir / "myth_missingness.png")
    plt.close(fig)

    print(f"EDA figures written to {fig_dir}")


def main():
    run_eda()


if __name__ == "__main__":
    main()
