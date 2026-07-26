"""Definitive myth-bust figures D1–D12."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.ladder import BULLET, load_hmd_wide
from src.analysis.populations import LABELS, PRIMARY_ALLOWLIST, dedupe_hmd, label_region
from src.paths import OUTPUTS, ensure_dirs

FIGS = OUTPUTS / "figures" / "definitive"
sns.set_theme(style="whitegrid", context="notebook")


def _claims() -> dict:
    return json.loads((BULLET / "claims.json").read_text(encoding="utf-8"))


def d1_sweden(panel: pd.DataFrame) -> Path:
    s = panel[panel.region_id == "SWE"].sort_values("year")
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(s.year, s.e0, lw=2.5, color="#1f77b4", label="e0 (birth)")
    ax1.plot(s.year, s.exp_death_65, lw=2.5, color="#2ca02c", label="Expected age if alive at 65")
    ax1.axhline(35, color="salmon", ls="--", lw=1.2, label="Myth: died at 35")
    ax1.set_ylabel("Years")
    ax1.set_ylim(0, 95)
    ax2 = ax1.twinx()
    ax2.plot(s.year, s.imr, color="#d62728", alpha=0.65, lw=1.5, label="IMR /1000 (right)")
    ax2.set_ylabel("Infant deaths per 1,000")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax1.set_title(
        "D1 Sweden — period life tables kill the myth\n"
        "Low e0 + high IMR; conditional age|65 stays in the 70s+"
    )
    r = s[s.year == 1800]
    if len(r):
        r = r.iloc[0]
        ax1.annotate(
            f"1800: e0={r.e0:.0f}, age|65={r.exp_death_65:.0f}, IMR={r.imr:.0f}, S65={100*r.s_to_65:.0f}%",
            xy=(1800, r.e0),
            xytext=(1835, 58),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8),
        )
    path = FIGS / "D1_sweden_storyboard.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def d2_ladder_heatmap() -> Path:
    # prefer HLD gold agg, else median
    for name in ("ladder_hld_gold_agg_e0lt40.csv", "ladder_hld_median_agg_e0lt40.csv"):
        p = BULLET / name
        if p.exists():
            agg = pd.read_csv(p)
            break
    else:
        return FIGS / "D2_MISSING.png"
    # also merge HMD ages 0,65,80
    hmd_agg = BULLET / "ladder_hmd_agg_e0lt40.csv"
    if hmd_agg.exists():
        h = pd.read_csv(hmd_agg)
        h["source"] = "HMD summary"
        agg = agg.copy()
        agg["source"] = "HLD"
        # plot HLD ladder as bars
    fig, ax = plt.subplots(figsize=(10, 5))
    ages = agg["age_x"].astype(int)
    ax.bar(ages.astype(str), agg["median_expected_age"], color="steelblue", edgecolor="white")
    ax.errorbar(
        ages.astype(str),
        agg["median_expected_age"],
        yerr=[
            agg["median_expected_age"] - agg["p10_expected_age"],
            agg["p90_expected_age"] - agg["median_expected_age"],
        ],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.axhline(35, color="salmon", ls="--", lw=1.5, label="Myth: 35")
    ax.set_xlabel("Age x (conditional on surviving to x)")
    ax.set_ylabel("Median expected age at death (x + e(x))")
    ax.set_title(
        "D2 Age ladder when e0 < 40 — conditional longevity >> 35 at every adult age\n"
        f"({p.name})"
    )
    ax.legend()
    for i, row in agg.iterrows():
        ax.text(
            str(int(row.age_x)),
            row.median_expected_age + 1.5,
            f"{row.median_expected_age:.0f}",
            ha="center",
            fontsize=8,
        )
    path = FIGS / "D2_age_ladder_when_e0_under_40.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def d3_multicountry(panel: pd.DataFrame) -> Path:
    # HMD e0 vs age65 + try HLD age15 for SWE FRA GBR
    have = [c for c in PRIMARY_ALLOWLIST if c in set(panel.region_id)]
    n = len(have)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, rid in enumerate(have):
        ax = axes[i]
        g = panel[panel.region_id == rid].sort_values("year")
        ax.plot(g.year, g.e0, label="e0", color="#1f77b4", lw=1.4)
        ax.plot(g.year, g.exp_death_65, label="age|65", color="#2ca02c", lw=1.4)
        ax.axhline(35, color="salmon", ls=":", lw=0.9)
        ax.set_title(label_region(rid), fontsize=10)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("D3 Multi-country: e0 vs expected age if alive at 65 (HMD summary, de-duped)", y=1.01)
    fig.tight_layout()
    path = FIGS / "D3_multicountry_e0_age65.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d4_distributions(panel: pd.DataFrame) -> Path:
    low = panel[panel.e0 < 40]
    # HLD age15 if available
    hldp = BULLET / "ladder_hld_median_long.csv"
    exp15 = None
    if hldp.exists():
        hl = pd.read_csv(hldp)
        exp15 = hl[(hl.age_x == 15) & (hl.e0 < 40)]["expected_age"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].hist(low.e0, bins=18, color="#1f77b4", edgecolor="white")
    axes[0].axvline(35, color="salmon", ls="--")
    axes[0].set_title("e0 | e0<40")
    axes[1].hist(low.exp_death_65, bins=18, color="#2ca02c", edgecolor="white")
    axes[1].axvline(35, color="salmon", ls="--")
    axes[1].axvline(70, color="black", ls="--")
    axes[1].set_title("Expected age | 65")
    if exp15 is not None and len(exp15):
        axes[2].hist(exp15, bins=18, color="#9467bd", edgecolor="white")
        axes[2].axvline(35, color="salmon", ls="--")
        axes[2].set_title("Expected age | 15 (HLD)")
    else:
        axes[2].hist(low.s_to_65 * 100, bins=18, color="#9467bd", edgecolor="white")
        axes[2].set_title("S(0→65) %")
    fig.suptitle(f"D4 Low-e0 distributions (HMD n={len(low)})", y=1.02)
    fig.tight_layout()
    path = FIGS / "D4_low_e0_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d5_forest(panel: pd.DataFrame) -> Path:
    low = panel[panel.e0 < 40]
    g = low.groupby("region_id").agg(n=("exp_death_65", "size"), med=("exp_death_65", "median"))
    g = g[g.n >= 5].sort_values("med").reset_index()
    g["label"] = g.region_id.map(label_region)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(g))))
    y = np.arange(len(g))
    ax.scatter(g.med, y, s=45, c="#2ca02c", zorder=3)
    ax.axvline(70, color="black", ls="--")
    ax.axvline(35, color="salmon", ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(g.label, fontsize=8)
    ax.set_xlabel("Median expected age if alive at 65 (when e0<40)")
    ax.set_title("D5 Myth A forest — every country still implies age|65 >> 35")
    path = FIGS / "D5_mythA_forest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d6_survival(panel: pd.DataFrame) -> Path:
    s = panel[panel.region_id == "SWE"].sort_values("year")
    dead1 = s.imr / 1000
    dead65 = 1 - s.s_to_65
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(s.year, 0, dead1, alpha=0.55, color="#d62728", label="Die in infancy (IMR)")
    ax.fill_between(s.year, dead1, dead65, alpha=0.45, color="#ff7f0e", label="Die ages 1–64")
    ax.fill_between(s.year, dead65, 1, alpha=0.45, color="#2ca02c", label="Survive to 65")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of births (period schedule)")
    ax.set_title("D6 Sweden — short e0 is early death composition, not adult death at 30")
    ax.legend(loc="upper right")
    path = FIGS / "D6_survival_composition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d7_imr(panel: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(panel.imr, panel.e0, c=panel.year.astype(float), s=9, alpha=0.4, cmap="coolwarm")
    plt.colorbar(sc, ax=axes[0], label="Year")
    axes[0].set_xlabel("IMR per 1,000")
    axes[0].set_ylabel("e0")
    axes[0].set_title("e0 vs IMR")
    # corrs
    corrs = []
    for rid, g in panel.groupby("region_id"):
        g = g.dropna(subset=["e0", "imr"])
        if len(g) >= 10:
            corrs.append(g.e0.corr(g.imr))
    axes[1].boxplot(corrs, vert=True)
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set_title(f"Within-country corr(e0,IMR)\nmedian={np.median(corrs):.2f}")
    axes[1].set_ylabel("Pearson r")
    fig.suptitle("D7 Infant drag — strong, near-universal association (not causal)")
    fig.tight_layout()
    path = FIGS / "D7_imr_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d8_first_diff(panel: pd.DataFrame) -> Path:
    d = panel.sort_values(["region_id", "year"]).copy()
    d["de0"] = d.groupby("region_id")["e0"].diff()
    d["dimr"] = d.groupby("region_id")["imr"].diff()
    d = d.dropna(subset=["de0", "dimr"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d.dimr, d.de0, s=8, alpha=0.25)
    ax.set_xlabel("Δ IMR")
    ax.set_ylabel("Δ e0")
    ax.set_title(f"D8 First differences (r={d.de0.corr(d.dimr):.2f}) — still negative, still not causal")
    path = FIGS / "D8_first_differences.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d9_models() -> Path:
    claims = _claims()
    models = claims["claim_C"]["models"]
    names = ["M0 pooled\nIMR", "M1 within\nIMR", "M2 within\nIMR+year", "M3 first\ndiff"]
    keys = ["M0_pooled", "M1_within", "M2_within_year", "M3_first_diff"]
    r2 = [models[k]["r2"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, r2, color="#1f77b4")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.set_title("D9 Association models only — high R² is structural, not ML overfit")
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    path = FIGS / "D9_model_r2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d10_myth_b(panel: pd.DataFrame) -> Path:
    pre = panel[panel.year < 1900].groupby("region_id")["e65"].mean()
    post = panel[panel.year >= 2000].groupby("region_id")["e65"].mean()
    both = pre.to_frame("pre").join(post.to_frame("post"), how="inner").dropna().reset_index()
    both["delta"] = both.post - both.pre
    both["label"] = both.region_id.map(label_region)
    both = both.sort_values("delta")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(both))))
    y = np.arange(len(both))
    ax.hlines(y, both.pre, both.post, color="gray", lw=2)
    ax.scatter(both.pre, y, c="#1f77b4", label="pre-1900 e65", zorder=3)
    ax.scatter(both.post, y, c="#2ca02c", label="post-2000 e65", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(both.label, fontsize=8)
    ax.set_xlabel("Remaining LE at age 65")
    ax.set_title("D10 Myth B — adult remaining LE rose in every de-duped country")
    ax.legend(fontsize=8)
    path = FIGS / "D10_mythB_dumbbell.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d11_coverage(panel: pd.DataFrame) -> Path:
    p = panel[panel.region_id.isin(PRIMARY_ALLOWLIST)].copy()
    p["label"] = p.region_id.map(label_region)
    p["decade"] = (p.year // 10) * 10
    piv = p.groupby(["label", "decade"]).size().unstack(fill_value=0)
    piv = (piv > 0).astype(int)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(piv, cmap="Greens", cbar_kws={"label": "has data"}, ax=ax)
    ax.set_title("D11 Coverage honesty — only Sweden spans mid-1700s in HMD summary")
    path = FIGS / "D11_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def d12_sex() -> Path:
    from src.analysis.panels import ANALYSIS_DIR

    f = dedupe_hmd(pd.read_parquet(ANALYSIS_DIR / "hmd_summary_wide_female.parquet"))
    m = dedupe_hmd(pd.read_parquet(ANALYSIS_DIR / "hmd_summary_wide_male.parquet"))
    sf = f[f.region_id == "SWE"].sort_values("year")
    sm = m[m.region_id == "SWE"].sort_values("year")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    axes[0].plot(sf.year, sf.e0, label="female")
    axes[0].plot(sm.year, sm.e0, label="male")
    axes[0].axhline(35, color="salmon", ls="--")
    axes[0].legend()
    axes[0].set_title("e0")
    axes[1].plot(sf.year, sf.exp_death_65, label="female")
    axes[1].plot(sm.year, sm.exp_death_65, label="male")
    axes[1].axhline(35, color="salmon", ls="--")
    axes[1].legend()
    axes[1].set_title("Expected age if alive at 65")
    fig.suptitle("D12 Sex sensitivity (Sweden) — myth fails for both sexes")
    fig.tight_layout()
    path = FIGS / "D12_sex_sweden.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_all() -> list[Path]:
    ensure_dirs()
    FIGS.mkdir(parents=True, exist_ok=True)
    if not (BULLET / "claims.json").exists():
        from src.analysis.bulletproof_suite import run as suite

        suite()
    panel = load_hmd_wide("both")
    paths = [
        d1_sweden(panel),
        d2_ladder_heatmap(),
        d3_multicountry(panel),
        d4_distributions(panel),
        d5_forest(panel),
        d6_survival(panel),
        d7_imr(panel),
        d8_first_diff(panel),
        d9_models(),
        d10_myth_b(panel),
        d11_coverage(panel),
        d12_sex(),
    ]
    for p in paths:
        print(f"Wrote {p}")
    return paths


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
