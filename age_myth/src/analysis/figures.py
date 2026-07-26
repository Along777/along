"""Publication-style figures for the age myth analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.panels import ANALYSIS_DIR, LABELS, PRIMARY_ALLOWLIST, save_panels
from src.paths import OUTPUTS, ensure_dirs

FIGS = OUTPUTS / "figures"
sns.set_theme(style="whitegrid", context="talk")


def _load_panel() -> pd.DataFrame:
    path = ANALYSIS_DIR / "hmd_summary_wide_both.parquet"
    if not path.exists():
        save_panels()
    return pd.read_parquet(path)


def f1_sweden(panel: pd.DataFrame) -> Path:
    s = panel[panel["region_id"] == "SWE"].sort_values("year")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax = axes[0, 0]
    ax.plot(s["year"], s["e0"], color="#1f77b4", lw=2, label="e0 (at birth)")
    ax.axhline(30, color="crimson", ls="--", lw=1, label="Myth line (30)")
    ax.axhline(35, color="salmon", ls=":", lw=1, label="Myth line (35)")
    ax.set_ylabel("Years")
    ax.set_title("Life expectancy at birth")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(s["year"], s["exp_death_65"], color="#2ca02c", lw=2)
    ax.axhline(30, color="crimson", ls="--", lw=1)
    ax.axhline(35, color="salmon", ls=":", lw=1)
    ax.set_ylabel("Years")
    ax.set_title("Expected age at death if alive at 65 (65 + e65)")

    ax = axes[1, 0]
    ax.plot(s["year"], s["imr"], color="#d62728", lw=2)
    ax.set_ylabel("Deaths per 1,000 births")
    ax.set_xlabel("Year")
    ax.set_title("Infant mortality rate")

    ax = axes[1, 1]
    ax.plot(s["year"], s["s_to_65"] * 100, color="#9467bd", lw=2)
    ax.set_ylabel("Percent")
    ax.set_xlabel("Year")
    ax.set_title("Survival from birth to age 65")

    fig.suptitle(
        "Sweden — the “died at 30” myth vs period life tables (HMD public summary)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    path = FIGS / "F1_sweden_myth_killer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f1b_multicountry(panel: pd.DataFrame) -> Path:
    primary = panel[panel["region_id"].isin(PRIMARY_ALLOWLIST)].copy()
    # pick countries with data
    have = [c for c in PRIMARY_ALLOWLIST if c in set(primary["region_id"])]
    n = len(have)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for i, rid in enumerate(have):
        ax = axes[i]
        g = primary[primary["region_id"] == rid].sort_values("year")
        ax.plot(g["year"], g["e0"], label="e₀", color="#1f77b4", lw=1.5)
        ax.plot(g["year"], g["exp_death_65"], label="age|65", color="#2ca02c", lw=1.5)
        ax.axhline(35, color="salmon", ls=":", lw=0.8)
        ax.set_title(LABELS.get(rid, rid), fontsize=11)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("e₀ vs expected age at death if alive at 65 (staggered historical starts)", y=1.01)
    fig.tight_layout()
    path = FIGS / "F1b_multicountry_e0_vs_exp65.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f2_scatter_e0_exp65(panel: pd.DataFrame) -> Path:
    d = panel.copy()
    fig, ax = plt.subplots(figsize=(9, 8))
    eras = ["pre_1850", "1850_1899", "1900_1949", "1950_1999", "2000_plus"]
    palette = sns.color_palette("viridis", n_colors=len(eras))
    for era, color in zip(eras, palette):
        g = d[d["era"] == era]
        ax.scatter(g["e0"], g["exp_death_65"], s=12, alpha=0.45, label=era, c=[color])
    lims = [20, 95]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.axvline(35, color="salmon", ls=":", lw=1)
    ax.set_xlabel("Life expectancy at birth (e₀)")
    ax.set_ylabel("Expected age at death if alive at 65")
    ax.set_title("When e₀ is low, adult conditional longevity is still high")
    ax.legend(fontsize=8, title="Era")
    ax.set_xlim(20, 90)
    ax.set_ylim(60, 95)
    path = FIGS / "F2_scatter_e0_vs_exp_death_65.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f3_e0_imr(panel: pd.DataFrame) -> Path:
    d = panel.dropna(subset=["e0", "imr"])
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(d["imr"], d["e0"], c=d["year"].astype(float), s=10, alpha=0.5, cmap="coolwarm")
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_xlabel("Infant mortality (per 1,000 live births)")
    ax.set_ylabel("Life expectancy at birth (e₀)")
    ax.set_title("e₀ collapses when infant mortality is high")
    path = FIGS / "F3_e0_vs_imr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f4_low_e0_distribution(panel: pd.DataFrame) -> Path:
    low = panel[panel["e0"] < 40]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].hist(low["e0"], bins=20, color="#1f77b4", edgecolor="white")
    axes[0].set_title("e₀ | e₀ < 40")
    axes[0].axvline(30, color="crimson", ls="--")
    axes[0].axvline(35, color="salmon", ls=":")
    axes[1].hist(low["exp_death_65"], bins=20, color="#2ca02c", edgecolor="white")
    axes[1].set_title("Expected age | 65  (same years)")
    axes[1].axvline(30, color="crimson", ls="--")
    axes[1].axvline(70, color="black", ls="--", label="70")
    axes[1].legend(fontsize=8)
    axes[2].hist(low["s_to_65"] * 100, bins=20, color="#9467bd", edgecolor="white")
    axes[2].set_title("S(0→65) %")
    fig.suptitle(f"Myth A test sample: n={len(low)} country-years with e₀ < 40", y=1.03)
    fig.tight_layout()
    path = FIGS / "F4_low_e0_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f5_eurostat_age_profile() -> Path:
    path_p = ANALYSIS_DIR / "eurostat_long.parquet"
    if not path_p.exists():
        save_panels()
    eu = pd.read_parquet(path_p)
    # latest common year-ish
    countries = ["SE", "FR", "DE", "IT", "ES", "PL"]
    sub = eu[(eu["sex"] == "both") & (eu["region_id"].isin(countries))].copy()
    # pick year with most coverage
    year_counts = sub.groupby("year").size()
    year = int(year_counts.idxmax())
    sub = sub[sub["year"] == year]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        g = sub[sub["region_id"] == c].sort_values("age")
        if g.empty:
            continue
        ax.plot(g["age"], g["life_expectancy"], marker="o", label=c)
    ax.set_xlabel("Age x")
    ax.set_ylabel("Remaining life expectancy e(x)")
    ax.set_title(f"Eurostat remaining LE by age (both sexes, {year})")
    ax.legend()
    path = FIGS / "F5_eurostat_age_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f6_concordance() -> Path:
    from src.analysis.myth_tests import concordance

    panel = _load_panel()
    # rebuild merge for plot
    from src.analysis.panels import build_owid_e0

    alias = {
        "SWE": "SWE",
        "USA": "USA",
        "NLD": "NLD",
        "DNK": "DNK",
        "NOR": "NOR",
        "BEL": "BEL",
        "ITA": "ITA",
        "CHE": "CHE",
        "FIN": "FIN",
        "ISL": "ISL",
        "AUS": "AUS",
        "CAN": "CAN",
        "JPN": "JPN",
        "ESP": "ESP",
        "FRANCE:_TOTAL_POPULATION": "FRA",
    }
    h = panel.copy()
    h["iso_try"] = h["region_id"].map(lambda x: alias.get(x))
    h = h.dropna(subset=["iso_try"])
    o = build_owid_e0()
    o = o[o["source_id"] == "owid_le_hmd_unwpp"]
    o["iso_try"] = o["region_id"].astype(str)
    m = h.merge(
        o[["iso_try", "year", "life_expectancy"]].rename(columns={"life_expectancy": "e0_owid"}),
        on=["iso_try", "year"],
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(m["e0_owid"], m["e0"], s=10, alpha=0.4)
    lim = [m[["e0", "e0_owid"]].min().min(), m[["e0", "e0_owid"]].max().max()]
    ax.plot(lim, lim, "k--", lw=1)
    stats = concordance(panel)
    ax.set_xlabel("OWID e₀ (HMD–UN series)")
    ax.set_ylabel("HMD public summary e₀")
    ax.set_title(
        f"Concordance (n={stats.get('n', 0):,}, r={stats.get('corr', float('nan')):.3f}, "
        f"RMSE={stats.get('rmse', float('nan')):.2f})"
    )
    path = FIGS / "F6_concordance_hmd_owid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def f_myth_b_dumbbell(panel: pd.DataFrame) -> Path:
    pre = panel[panel["year"] < 1900].groupby(["region_id", "label"])["e65"].mean()
    post = panel[panel["year"] >= 2000].groupby(["region_id", "label"])["e65"].mean()
    both = pre.to_frame("pre").join(post.to_frame("post"), how="inner").dropna().reset_index()
    both["delta"] = both["post"] - both["pre"]
    both = both.sort_values("delta")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(both))))
    y = np.arange(len(both))
    ax.hlines(y, both["pre"], both["post"], color="gray", lw=2)
    ax.scatter(both["pre"], y, color="#1f77b4", label="pre-1900 mean e₆₅", zorder=3)
    ax.scatter(both["post"], y, color="#2ca02c", label="post-2000 mean e₆₅", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(both["label"], fontsize=9)
    ax.set_xlabel("Remaining life expectancy at age 65 (years)")
    ax.set_title("Myth B: adult remaining LE rose substantially")
    ax.legend(fontsize=9)
    path = FIGS / "F_mythB_dumbbell_e65.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_all() -> list[Path]:
    ensure_dirs()
    FIGS.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    paths = [
        f1_sweden(panel),
        f1b_multicountry(panel),
        f2_scatter_e0_exp65(panel),
        f3_e0_imr(panel),
        f4_low_e0_distribution(panel),
        f5_eurostat_age_profile(),
        f6_concordance(),
        f_myth_b_dumbbell(panel),
    ]
    for p in paths:
        print(f"Wrote {p}")
    return paths


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
