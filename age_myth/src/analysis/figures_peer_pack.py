"""Peer-review figure pack for myth-busting (G1–G10)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.panels import ANALYSIS_DIR, LABELS, PRIMARY_ALLOWLIST, save_panels
from src.analysis.peer_hardening import dedupe_panel, load_panel
from src.paths import OUTPUTS, PROCESSED, ensure_dirs

FIGS = OUTPUTS / "figures" / "peer_pack"
sns.set_theme(style="whitegrid", context="notebook")


def _panel() -> pd.DataFrame:
    p = ANALYSIS_DIR / "hmd_summary_wide_both.parquet"
    if not p.exists():
        save_panels()
    return dedupe_panel(pd.read_parquet(p))


def g1_sweden_storyboard(panel: pd.DataFrame) -> Path:
    s = panel[panel["region_id"] == "SWE"].sort_values("year")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(s["year"], s["e0"], color="#1f77b4", lw=2.5, label="e0 at birth")
    ax1.plot(s["year"], s["exp_death_65"], color="#2ca02c", lw=2.5, label="Expected age if alive at 65")
    ax1.axhline(35, color="salmon", ls="--", lw=1, label="Myth: died at 35")
    ax1.set_ylabel("Years")
    ax1.set_xlabel("Year")
    ax1.set_ylim(0, 95)
    ax2 = ax1.twinx()
    ax2.plot(s["year"], s["imr"], color="#d62728", lw=1.5, alpha=0.7, label="IMR (right)")
    ax2.set_ylabel("Infant deaths per 1,000 births")
    # combine legends
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=9)
    ax1.set_title(
        "Sweden storyboard (period life tables): low e0 coexists with old age if you reach 65\n"
        "High IMR explains the gap — not adults dying at 30"
    )
    # annotate early point
    r = s[s["year"] == 1800]
    if len(r):
        r = r.iloc[0]
        ax1.annotate(
            f"1800: e0={r.e0:.0f}, age|65={r.exp_death_65:.0f}, IMR={r.imr:.0f}",
            xy=(1800, r.e0),
            xytext=(1820, 55),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9,
        )
    path = FIGS / "G1_sweden_storyboard.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g2_hld_mid_adult() -> Path:
    path_h = PROCESSED / "life_expectancy_modeling_hld_median.parquet"
    h = pd.read_parquet(path_h)
    w = h.pivot_table(
        index=["region_id", "year"], columns="age", values="life_expectancy", aggfunc="mean"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    countries = ["SWE", "FRA", "GBR", "USA"]
    for rid in countries:
        if rid not in w.index.get_level_values(0):
            continue
        s = w.loc[rid].dropna(subset=[0, 15] if 15 in w.columns else [0]).reset_index()
        if 15 not in s.columns:
            continue
        axes[0].plot(s["year"], s[0], label=f"{rid} e0")
        axes[1].plot(s["year"], 15 + s[15], label=f"{rid} age|15")
    axes[0].axhline(35, color="salmon", ls="--")
    axes[1].axhline(35, color="salmon", ls="--", label="myth 35")
    axes[0].set_title("HLD median: life expectancy at birth")
    axes[1].set_title("HLD median: expected age if alive at 15 (15+e15)")
    axes[0].set_ylabel("Years")
    axes[1].set_ylabel("Years")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Mid-adult check (HLD): surviving childhood already implies ages well above 30–35\n"
        "(HLD methods heterogeneous; median across tables)",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGS / "G2_hld_mid_adult_e0_vs_age15.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # scatter low e0
    if 15 in w.columns:
        ww = w.dropna(subset=[0, 15]).copy()
        ww["exp15"] = 15 + ww[15]
        low = ww[ww[0] < 40].reset_index()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(low[0], low["exp15"], s=12, alpha=0.4)
        ax.axhline(35, color="salmon", ls="--")
        ax.axvline(35, color="salmon", ls=":")
        ax.set_xlabel("e0")
        ax.set_ylabel("Expected age if alive at 15")
        ax.set_title(f"HLD low-e0 country-years (n={len(low)}): age|15 still >> 35")
        path2 = FIGS / "G2b_hld_scatter_low_e0_age15.png"
        fig.savefig(path2, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return path


def g3_corr_boxplot(panel: pd.DataFrame) -> Path:
    from src.analysis.peer_hardening import within_country_corrs

    c = within_country_corrs(panel)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(y=c["corr_pearson"], ax=ax, color="steelblue")
    sns.stripplot(y=c["corr_pearson"], ax=ax, color="black", alpha=0.4, size=4)
    ax.axhline(0, color="gray", ls="--")
    ax.set_ylabel("Within-country corr(e0, IMR)")
    ax.set_title(
        f"Infant drag is near-universal (n={len(c)} countries)\n"
        f"median r={c['corr_pearson'].median():.2f}"
    )
    path = FIGS / "G3_within_country_corr_boxplot.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g4_first_diff(panel: pd.DataFrame) -> Path:
    d = panel.sort_values(["region_id", "year"]).copy()
    d["de0"] = d.groupby("region_id")["e0"].diff()
    d["dimr"] = d.groupby("region_id")["imr"].diff()
    d = d.dropna(subset=["de0", "dimr"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(d["dimr"], d["de0"], s=8, alpha=0.25)
    axes[0].set_xlabel("Year-to-year change in IMR")
    axes[0].set_ylabel("Year-to-year change in e0")
    axes[0].set_title(f"Pooled first differences (r={d['de0'].corr(d['dimr']):.2f})")
    s = d[d["region_id"] == "SWE"]
    axes[1].scatter(s["dimr"], s["de0"], s=12, alpha=0.5, c=s["year"].astype(float), cmap="viridis")
    axes[1].set_xlabel("Δ IMR")
    axes[1].set_ylabel("Δ e0")
    axes[1].set_title("Sweden first differences")
    fig.suptitle("First differences reduce pure time-trend confounds (still not causal)")
    fig.tight_layout()
    path = FIGS / "G4_first_differences_e0_imr.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g5_model_comparison() -> Path:
    import json

    path_j = OUTPUTS / "tables" / "peer_hardened_summary.json"
    if not path_j.exists():
        from src.analysis.peer_hardening import run

        run()
    data = json.loads(path_j.read_text(encoding="utf-8"))
    models = data["model_comparison"]["models"]
    names = [
        "M0 pooled\nIMR only",
        "M1 within\nIMR only",
        "M2 within\nIMR+year",
        "M3 first\ndiff",
        "FE full\n(sens.)",
    ]
    keys = [
        "M0_pooled_imr_only",
        "M1_within_imr_only",
        "M2_within_imr_plus_year",
        "M3_first_difference",
        "M_fe_full_sensitivity",
    ]
    r2 = [models[k]["r2"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, r2, color=["#1f77b4", "#1f77b4", "#1f77b4", "#2ca02c", "#bbbbbb"])
    bars[-1].set_hatch("//")
    ax.set_ylabel("R-squared")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Association models for e0 — high R2 is expected demography, not ML overfit\n"
        "Gray hatched = FE sensitivity (do not headline)"
    )
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    path = FIGS / "G5_model_r2_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g6_myth_a_forest(panel: pd.DataFrame) -> Path:
    low = panel[panel["e0"] < 40]
    g = (
        low.groupby("region_id")
        .agg(n=("exp_death_65", "size"), med=("exp_death_65", "median"), e0=("e0", "median"))
        .reset_index()
    )
    g = g[g["n"] >= 5].sort_values("med")
    g["label"] = g["region_id"].map(lambda x: LABELS.get(x, str(x)[:28]))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(g))))
    y = np.arange(len(g))
    ax.hlines(y, 70, g["med"], color="lightgray")
    ax.scatter(g["med"], y, s=40, c="#2ca02c", zorder=3)
    ax.axvline(70, color="black", ls="--", lw=1)
    ax.axvline(35, color="salmon", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(g["label"], fontsize=8)
    ax.set_xlabel("Median expected age at death if alive at 65 (when e0<40)")
    ax.set_title("Myth A by country: conditional old-age longevity, not death at 35")
    path = FIGS / "G6_mythA_forest_by_country.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g7_sex_gap(panel_path_both=None) -> Path:
    # load female/male panels
    fb = pd.read_parquet(ANALYSIS_DIR / "hmd_summary_wide_female.parquet")
    mb = pd.read_parquet(ANALYSIS_DIR / "hmd_summary_wide_male.parquet")
    fb = dedupe_panel(fb)
    mb = dedupe_panel(mb)
    s_f = fb[fb["region_id"] == "SWE"].sort_values("year")
    s_m = mb[mb["region_id"] == "SWE"].sort_values("year")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].plot(s_f["year"], s_f["e0"], label="female e0")
    axes[0].plot(s_m["year"], s_m["e0"], label="male e0")
    axes[0].axhline(35, color="salmon", ls="--")
    axes[0].set_title("Sweden e0 by sex")
    axes[0].legend()
    axes[1].plot(s_f["year"], s_f["exp_death_65"], label="female age|65")
    axes[1].plot(s_m["year"], s_m["exp_death_65"], label="male age|65")
    axes[1].axhline(35, color="salmon", ls="--")
    axes[1].set_title("Sweden expected age if alive at 65 by sex")
    axes[1].legend()
    fig.suptitle("Sex sensitivity: myth fails for both males and females")
    fig.tight_layout()
    path = FIGS / "G7_sweden_sex_gap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g8_survival_stack(panel: pd.DataFrame) -> Path:
    s = panel[panel["region_id"] == "SWE"].sort_values("year")
    # approximate share dead by 1 from IMR/1000; dead by 65 = 1 - s_to_65
    dead1 = s["imr"] / 1000.0
    dead65 = 1 - s["s_to_65"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(s["year"], 0, dead1, alpha=0.5, label="Approx. die in infancy (IMR)", color="#d62728")
    ax.fill_between(
        s["year"], dead1, dead65, alpha=0.4, label="Die ages 1-64 (period table)", color="#ff7f0e"
    )
    ax.fill_between(s["year"], dead65, 1, alpha=0.4, label="Survive to 65", color="#2ca02c")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of births (period schedule)")
    ax.set_xlabel("Year")
    ax.set_title("Sweden: most of the 'short life expectancy' is early death, not adult death at 30")
    ax.legend(loc="upper right", fontsize=9)
    path = FIGS / "G8_sweden_survival_composition.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g9_eurostat_gap() -> Path:
    eu = pd.read_parquet(ANALYSIS_DIR / "eurostat_long.parquet")
    countries = ["SE", "FR", "DE", "IT", "ES"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in countries:
        e0 = eu[(eu.region_id == c) & (eu.age == 0) & (eu.sex == "both")]
        e15 = eu[(eu.region_id == c) & (eu.age == 15) & (eu.sex == "both")]
        m = e0[["year", "life_expectancy"]].merge(
            e15[["year", "life_expectancy"]], on="year", suffixes=("_0", "_15")
        )
        if m.empty:
            continue
        # remaining at 15; expected age = 15+e15
        ax.plot(m["year"], (15 + m["life_expectancy_15"]) - m["life_expectancy_0"], label=c)
    ax.set_xlabel("Year")
    ax.set_ylabel("(15+e15) - e0  [years]")
    ax.set_title("Eurostat modern era: gap between birth e0 and expected age if alive at 15")
    ax.legend()
    path = FIGS / "G9_eurostat_e0_age15_gap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g10_coverage(panel: pd.DataFrame) -> Path:
    primary = panel[panel["region_id"].isin(PRIMARY_ALLOWLIST)].copy()
    primary["label"] = primary["region_id"].map(lambda x: LABELS.get(x, x))
    # binary presence of e0 by decade
    primary["decade"] = (primary["year"] // 10) * 10
    piv = (
        primary.groupby(["label", "decade"])
        .size()
        .unstack(fill_value=0)
    )
    piv = (piv > 0).astype(int)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(piv, cmap="Greens", cbar_kws={"label": "has data"}, ax=ax)
    ax.set_title("Coverage honesty: only Sweden reaches mid-18th century in HMD summary")
    ax.set_xlabel("Decade")
    ax.set_ylabel("")
    path = FIGS / "G10_coverage_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g_myth_b_deduped(panel: pd.DataFrame) -> Path:
    d = panel
    pre = d[d["year"] < 1900].groupby("region_id")["e65"].mean()
    post = d[d["year"] >= 2000].groupby("region_id")["e65"].mean()
    both = pre.to_frame("pre").join(post.to_frame("post"), how="inner").dropna().reset_index()
    both["delta"] = both["post"] - both["pre"]
    both["label"] = both["region_id"].map(lambda x: LABELS.get(x, str(x)[:30]))
    both = both.sort_values("delta")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(both))))
    y = np.arange(len(both))
    ax.hlines(y, both["pre"], both["post"], color="gray", lw=2)
    ax.scatter(both["pre"], y, label="pre-1900 mean e65", color="#1f77b4", zorder=3)
    ax.scatter(both["post"], y, label="post-2000 mean e65", color="#2ca02c", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(both["label"], fontsize=8)
    ax.set_xlabel("Remaining LE at age 65")
    ax.set_title("Myth B (de-duplicated): adult remaining LE rose in every country shown")
    ax.legend(fontsize=8)
    path = FIGS / "G_mythB_dumbbell_deduped.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_all() -> list[Path]:
    ensure_dirs()
    FIGS.mkdir(parents=True, exist_ok=True)
    # ensure hardened stats exist for G5
    from src.analysis.peer_hardening import run as harden

    if not (OUTPUTS / "tables" / "peer_hardened_summary.json").exists():
        harden()

    panel = _panel()
    paths = [
        g1_sweden_storyboard(panel),
        g2_hld_mid_adult(),
        g3_corr_boxplot(panel),
        g4_first_diff(panel),
        g5_model_comparison(),
        g6_myth_a_forest(panel),
        g7_sex_gap(),
        g8_survival_stack(panel),
        g9_eurostat_gap(),
        g10_coverage(panel),
        g_myth_b_deduped(panel),
    ]
    for p in paths:
        print(f"Wrote {p}")
    return paths


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
