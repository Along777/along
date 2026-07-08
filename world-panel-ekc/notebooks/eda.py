#!/usr/bin/env python3
"""
eda.py
======
Exploratory data analysis for the world country-year panel produced by
build_panel.py. Reads output/panel_wide.csv and writes figures, summary
tables, and a markdown report into output/eda/.

Requires: pandas, numpy, matplotlib, tabulate (for markdown tables), and
optionally seaborn.

Run:
    python eda.py                      # uses output/panel_wide.csv
    python eda.py path/to/panel.csv    # custom path

Produces in output/eda/:
    coverage_over_time.png     share of countries reporting each indicator/yr
    distributions.png          histograms (log scale where skewed)
    correlation.png            cross-sectional correlation, latest good year
    scatter_relationships.png  GDP vs CO2 / life exp / internet, by region
    trends.png                 region-mean trends for key indicators
    convergence.png            beta convergence: initial GDP vs subsequent growth
    kuznets.png                environmental Kuznets curve check (GDP vs CO2 pc)
    governance_corr.csv        WGI governance indicators vs log GDP pc, snapshot yr
    top_movers.csv             biggest % gainers/losers, GDP pc & CO2 pc
    summary_latest.csv         describe() for the latest good year
    summary_by_income.csv      means by income group, latest good year
    missingness.csv            % missing per indicator
    eda_report.md              narrative tying it together
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

META_COLS = ["iso3", "country_name", "region", "income_group", "year",
             "is_micro_state"]
# indicators that are strongly right-skewed -> plot on log scale
LOG_INDICATORS = {
    "gdp_pc_ppp_const", "population", "energy_use_pc", "co2_pc",
    "co2_total_kt", "patent_apps_resident",
}
# WGI governance estimates, roughly N(0,1) -- useful to group for reporting
GOVERNANCE_INDICATORS = [
    "control_corruption", "gov_effectiveness", "political_stability",
    "regulatory_quality", "rule_of_law", "voice_accountability",
]
# Thematic grouping for the correlation heatmap -- a single 38x38 grid is
# illegible, so we render one readable sub-heatmap per theme instead.
CORR_THEMES = {
    "Economy & demographics": [
        "gdp_pc_ppp_const", "gdp_growth", "unemployment", "gcf_pct_gdp",
        "gross_fixed_capital_pct_gdp", "gross_savings_pct_gdp",
        "trade_pct_gdp", "fdi_net_inflow_pct_gdp", "population", "urban_pct",
        "fertility_rate", "dependency_ratio",
    ],
    "Energy & environment": [
        "co2_pc", "co2_total_mt", "pm25_exposure", "renew_share",
        "energy_use_pc", "energy_intensity", "fossil_fuel_pct",
        "elec_access", "clean_cooking_access",
    ],
    "Governance, human capital & innovation": [
        "control_corruption", "gov_effectiveness", "political_stability",
        "regulatory_quality", "rule_of_law", "voice_accountability",
        "secondary_enroll", "tertiary_enroll", "health_exp_pct_gdp",
        "rnd_exp_pct_gdp", "hightech_exports_pct", "patent_apps_resident",
        "ict_service_exports_pct", "internet_users", "life_expectancy",
        "gini", "poverty_215",
    ],
}

IN_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/panel_wide.csv")
OUT_DIR = Path("output/eda")


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def indicator_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def pick_year(df: pd.DataFrame, cols: list[str], min_obs: int = 25) -> int:
    """Latest year where every col in `cols` has >= min_obs countries.

    Used for diagnostics that specifically need those columns (e.g. the
    Kuznets check needs gdp_pc_ppp_const + co2_pc). NOT used for the
    all-indicator snapshot -- see pick_snapshot_year."""
    for yr in sorted(df["year"].unique(), reverse=True):
        sub = df[df["year"] == yr]
        if all(sub[c].notna().sum() >= min_obs for c in cols):
            return int(yr)
    return int(df["year"].max())


def pick_snapshot_year(df: pd.DataFrame, inds: list[str], min_obs: int = 25,
                        max_bad_frac: float = 0.1) -> int:
    """Latest year where nearly all indicators have decent coverage.

    A handful of hardcoded "core" columns (as pick_year does) is misleading
    once the indicator set is wide: some indicators (WB reporting lag) can be
    1-3 years behind the core ones, so picking the very latest year based on
    3 columns leaves many other indicators completely blank in that year's
    distribution/correlation plots even though recent data exists for them.
    A quantile-based cutoff still lets exactly-zero-coverage indicators slip
    through when they're a small enough fraction, so instead directly cap the
    fraction of indicators allowed to fall below `min_obs`.
    """
    for yr in sorted(df["year"].unique(), reverse=True):
        sub = df[df["year"] == yr]
        counts = sub[inds].notna().sum()
        bad_frac = (counts < min_obs).mean()
        if bad_frac <= max_bad_frac:
            return int(yr)
    return int(df["year"].max())


def have(df, *cols) -> bool:
    return all(c in df.columns for c in cols)


# --------------------------------------------------------------------------- #
# 1. coverage                                                                  #
# --------------------------------------------------------------------------- #

def coverage_over_time(df, inds, out):
    n_countries = df.groupby("year")["iso3"].nunique()
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in inds:
        share = df[df[c].notna()].groupby("year")["iso3"].nunique() / n_countries
        ax.plot(share.index, share.values, marker=".", label=c, linewidth=1.4)
    ax.set_title("Indicator coverage over time (share of countries reporting)")
    ax.set_xlabel("year"); ax.set_ylabel("share of countries")
    ax.set_ylim(0, 1.02)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def missingness_table(df, inds, out):
    miss = (df[inds].isna().mean().sort_values(ascending=False)
            .rename("missing_share").reset_index()
            .rename(columns={"index": "indicator"}))
    miss.to_csv(out, index=False)
    return miss


# --------------------------------------------------------------------------- #
# 2. distributions                                                             #
# --------------------------------------------------------------------------- #

def distributions(df, inds, year, out):
    sub = df[df["year"] == year]
    n = len(inds); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, inds):
        vals = sub[c].dropna()
        logged = c in LOG_INDICATORS and (vals > 0).all() and len(vals) > 0
        data = np.log10(vals) if logged else vals
        if len(data):
            ax.hist(data, bins=20, color="#4c72b0", edgecolor="white")
        ax.set_title(f"{c}{' (log10)' if logged else ''}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Distributions, {year}", y=1.005)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. correlation                                                               #
# --------------------------------------------------------------------------- #

def _draw_corr_heatmap(ax, corr, labels, title):
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.6 else "white")
    ax.set_title(title, fontsize=10)
    return im


def correlation(df, inds, year, out):
    """Full cross-sectional correlation matrix (all inds, for the report's
    "corr with GDP" table), rendered as readable per-theme sub-heatmaps
    rather than one illegible NxN grid."""
    sub = df[df["year"] == year][inds]
    corr = sub.corr()

    groups = [(name, [c for c in cols if c in inds])
              for name, cols in CORR_THEMES.items()]
    grouped = {c for _, cols in groups for c in cols}
    leftover = [c for c in inds if c not in grouped]
    if leftover:
        groups.append(("Other", leftover))
    groups = [(name, cols) for name, cols in groups if cols]

    fig, axes = plt.subplots(1, len(groups),
                              figsize=(sum(0.55 * len(c) + 2.5 for _, c in groups),
                                       max(0.55 * len(c) for _, c in groups) + 2))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (name, cols) in zip(axes, groups):
        im = _draw_corr_heatmap(ax, corr.loc[cols, cols], cols, name)
    fig.suptitle(f"Cross-sectional correlation, {year} (by theme)", y=1.02)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return corr


# --------------------------------------------------------------------------- #
# 4. relationship scatters                                                     #
# --------------------------------------------------------------------------- #

def scatter_relationships(df, year, out):
    x = "gdp_pc_ppp_const"
    pairs = [(x, "co2_pc", True), (x, "life_expectancy", False),
             (x, "internet_users", False)]
    pairs = [(a, b, lg) for a, b, lg in pairs if have(df, a, b)]
    if not pairs:
        return
    sub = df[df["year"] == year]
    regions = sorted(sub["region"].dropna().unique()) if "region" in sub else []
    cmap = plt.get_cmap("tab10")
    colors = {r: cmap(i % 10) for i, r in enumerate(regions)}

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, (a, b, ylog) in zip(axes, pairs):
        d = sub[[a, b] + (["region"] if regions else [])].dropna(subset=[a, b])
        if regions:
            for r in regions:
                dr = d[d["region"] == r]
                ax.scatter(dr[a], dr[b], s=18, alpha=0.75,
                           color=colors[r], label=r)
        else:
            ax.scatter(d[a], d[b], s=18, alpha=0.75)
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel(f"{a} (log)"); ax.set_ylabel(b)
        ax.set_title(f"{b} vs GDP p.c., {year}", fontsize=10)
    if regions:
        axes[0].legend(fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. regional trends                                                           #
# --------------------------------------------------------------------------- #

def trends(df, out):
    want = [c for c in ["co2_pc", "gdp_pc_ppp_const", "internet_users",
                        "renew_share"] if c in df.columns]
    if "region" not in df.columns or not want:
        return
    fig, axes = plt.subplots(1, len(want), figsize=(5 * len(want), 4))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, want):
        g = df.groupby(["region", "year"])[c].mean().reset_index()
        for r, dr in g.groupby("region"):
            ax.plot(dr["year"], dr[c], label=r, linewidth=1.3)
        ax.set_title(f"Region mean: {c}", fontsize=10)
        ax.set_xlabel("year")
    axes[0].legend(fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. beta convergence                                                          #
# --------------------------------------------------------------------------- #

def convergence_plot(df, out):
    """Barro-style beta convergence: initial log GDP pc vs subsequent avg growth.

    Returns the OLS slope (None if not computable). Negative slope = poorer
    countries grew faster on average = convergence.
    """
    if "gdp_pc_ppp_const" not in df.columns:
        return None
    d = df[["iso3", "year", "gdp_pc_ppp_const"]].dropna()
    d = d[d["gdp_pc_ppp_const"] > 0]
    if d.empty:
        return None

    first = d.loc[d.groupby("iso3")["year"].idxmin()].set_index("iso3")
    last = d.loc[d.groupby("iso3")["year"].idxmax()].set_index("iso3")
    span = last["year"] - first["year"]
    valid = span[span >= 5].index  # need a few years to talk about "growth"
    if len(valid) < 5:
        return None

    log_gdp0 = np.log(first.loc[valid, "gdp_pc_ppp_const"])
    n_years = span.loc[valid]
    avg_growth = (
        np.log(last.loc[valid, "gdp_pc_ppp_const"])
        - np.log(first.loc[valid, "gdp_pc_ppp_const"])
    ) / n_years

    slope, intercept = np.polyfit(log_gdp0, avg_growth, 1)
    xs = np.linspace(log_gdp0.min(), log_gdp0.max(), 50)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(log_gdp0, avg_growth, s=20, alpha=0.7, color="#4c72b0")
    ax.plot(xs, slope * xs + intercept, color="firebrick", linewidth=1.5)
    ax.set_xlabel("log GDP per capita, earliest available year")
    ax.set_ylabel("average annual log growth to latest available year")
    ax.set_title(f"Beta convergence (slope = {slope:.4f})")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return float(slope)


# --------------------------------------------------------------------------- #
# 7. environmental Kuznets curve check                                        #
# --------------------------------------------------------------------------- #

def kuznets_check(df, year, out):
    """Quadratic fit of CO2 pc on log GDP pc; report inverted-U turning point."""
    if not have(df, "gdp_pc_ppp_const", "co2_pc"):
        return None
    sub = df[df["year"] == year][["gdp_pc_ppp_const", "co2_pc"]].dropna()
    sub = sub[sub["gdp_pc_ppp_const"] > 0]
    if len(sub) < 20:
        return None

    x = np.log(sub["gdp_pc_ppp_const"])
    y = sub["co2_pc"]
    a, b, c = np.polyfit(x, y, 2)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = a * xs**2 + b * xs + c

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(x, y, s=18, alpha=0.6, color="#55a868")
    ax.plot(xs, ys, color="firebrick", linewidth=1.5)
    ax.set_xlabel("log GDP per capita"); ax.set_ylabel("CO2 emissions per capita (t)")
    ax.set_title(f"Environmental Kuznets curve check, {year}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

    if a < 0:
        turning_log_gdp = -b / (2 * a)
        if x.min() <= turning_log_gdp <= x.max():
            return float(np.exp(turning_log_gdp))
    return None


# --------------------------------------------------------------------------- #
# 8. governance vs GDP                                                        #
# --------------------------------------------------------------------------- #

def governance_gdp_table(df, year, out):
    gov_cols = [c for c in GOVERNANCE_INDICATORS if c in df.columns]
    if not gov_cols or "gdp_pc_ppp_const" not in df.columns:
        return None
    sub = df[df["year"] == year].copy()
    sub = sub[sub["gdp_pc_ppp_const"] > 0]
    if sub.empty:
        return None
    sub["log_gdp_pc"] = np.log(sub["gdp_pc_ppp_const"])
    corr = (
        sub[gov_cols + ["log_gdp_pc"]].corr()["log_gdp_pc"]
        .drop(labels=["log_gdp_pc"])
        .sort_values(ascending=False)
        .rename("corr_with_log_gdp_pc")
    )
    corr.to_csv(out)
    return corr


# --------------------------------------------------------------------------- #
# 9. top movers                                                               #
# --------------------------------------------------------------------------- #

def top_movers(df, out, n=5):
    """Countries with the largest total % change (first -> last obs) on
    gdp_pc_ppp_const and co2_pc."""
    frames = []
    for col in ["gdp_pc_ppp_const", "co2_pc"]:
        if col not in df.columns:
            continue
        d = df[["iso3", "year", col]].dropna()
        d = d[d[col] > 0]
        if d.empty:
            continue
        first = d.loc[d.groupby("iso3")["year"].idxmin()].set_index("iso3")[col]
        last = d.loc[d.groupby("iso3")["year"].idxmax()].set_index("iso3")[col]
        span = (
            d.loc[d.groupby("iso3")["year"].idxmax()].set_index("iso3")["year"]
            - d.loc[d.groupby("iso3")["year"].idxmin()].set_index("iso3")["year"]
        )
        pct = ((last / first) - 1) * 100
        pct = pct[span >= 5]
        if pct.empty:
            continue
        top = pct.sort_values(ascending=False).head(n)
        bottom = pct.sort_values(ascending=True).head(n)
        for iso3, v in pd.concat([top, bottom]).items():
            frames.append({"indicator": col, "iso3": iso3, "pct_change": round(v, 1)})
    if not frames:
        return None
    result = pd.DataFrame(frames)
    result.to_csv(out, index=False)
    return result


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #

def write_report(df, inds, year, miss, corr, out,
                  conv_slope=None, kuznets_turn=None, gov_corr=None, movers=None):
    n_c = df["iso3"].nunique()
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    t_per_country = df.groupby("iso3")["year"].nunique()
    lines = []
    lines.append("# World panel — EDA report\n")
    lines.append(f"- Countries (N): **{n_c}**")
    lines.append(f"- Years: **{yr_min}–{yr_max}**")
    lines.append(f"- Indicators: **{len(inds)}**")
    lines.append(f"- Cross-sectional snapshot year (best coverage): **{year}**")
    lines.append(
        f"- Panel structure (T = years observed per country): "
        f"min **{int(t_per_country.min())}**, "
        f"median **{int(t_per_country.median())}**, "
        f"max **{int(t_per_country.max())}** "
        f"({'balanced' if t_per_country.nunique() == 1 else 'unbalanced'} panel)\n"
    )

    lines.append("## Coverage\n")
    lines.append("Missing share by indicator (whole panel):\n")
    lines.append(miss.to_markdown(index=False))
    lines.append("\n![coverage](coverage_over_time.png)\n")

    lines.append("## Distributions\n")
    lines.append("![distributions](distributions.png)\n")

    lines.append("## Correlations\n")
    if corr is not None and "gdp_pc_ppp_const" in corr.columns:
        s = corr["gdp_pc_ppp_const"].drop(labels=["gdp_pc_ppp_const"],
                                          errors="ignore").sort_values()
        lines.append("Strongest (anti)correlations with GDP per capita "
                     f"in {year}:\n")
        lines.append(s.round(2).to_frame("corr_with_gdp_pc").to_markdown())
    lines.append("\n![correlation](correlation.png)\n")

    lines.append("## Key relationships\n")
    lines.append("![scatter](scatter_relationships.png)\n")

    lines.append("## Regional trends\n")
    lines.append("![trends](trends.png)\n")

    lines.append("## Beta convergence\n")
    if conv_slope is not None:
        verdict = "convergence (poorer countries grew faster)" if conv_slope < 0 \
            else "divergence (richer countries grew faster)"
        lines.append(f"OLS slope of avg. growth on initial log GDP p.c.: "
                     f"**{conv_slope:.4f}** -> {verdict}.\n")
    lines.append("![convergence](convergence.png)\n")

    lines.append("## Environmental Kuznets curve check\n")
    if kuznets_turn is not None:
        lines.append(f"Inverted-U shape detected; CO2-per-capita turning point "
                     f"around **${kuznets_turn:,.0f}** GDP p.c. "
                     f"(quadratic fit, {year}).\n")
    else:
        lines.append("No inverted-U turning point detected within the observed "
                     f"GDP range in {year} (or insufficient data).\n")
    lines.append("![kuznets](kuznets.png)\n")

    if gov_corr is not None and len(gov_corr):
        lines.append("## Governance vs. GDP per capita\n")
        lines.append(f"Correlation of WGI governance estimates with log GDP p.c. "
                     f"in {year}:\n")
        lines.append(gov_corr.round(2).to_frame().to_markdown())
        lines.append("")

    if movers is not None and len(movers):
        lines.append("## Interesting facts: biggest movers\n")
        lines.append("Largest total % change from each country's first to last "
                     "available observation (min. 5-year span):\n")
        lines.append(movers.to_markdown(index=False))
        lines.append("")

    lines.append("## Caveats\n")
    lines.append("- Cross-sectional correlations are **pooled** and confound "
                 "between-country and within-country variation; treat as "
                 "descriptive only, not causal.")
    lines.append("- Coverage is uneven: indicators with high missing share "
                 "(see table) will shrink any complete-case model window.")
    lines.append("- `energy_use_pc` (World Bank) effectively ends ~2014.")
    lines.append("- WGI governance indicators (`control_corruption`, "
                 "`gov_effectiveness`, `political_stability`, "
                 "`regulatory_quality`, `rule_of_law`, "
                 "`voice_accountability`) are **structurally missing**, not "
                 "randomly missing, for 1997/1999/2001 -- the survey was "
                 "biennial before 2002. Don't naively interpolate or use "
                 "year fixed effects across that gap without accounting "
                 "for it.")
    lines.append("- `gini` and `rnd_exp_pct_gdp` never reach ~50% "
                 "cross-sectional coverage in *any* single year (survey-"
                 "based, irregular timing) -- usable pooled across years, "
                 "not as a single-year cross-section.")
    lines.append("- Micro-states/territories (population < 1,000,000, "
                 "flagged via `is_micro_state` in the panel) disproportion"
                 "ately drive missingness and occasional implausible values "
                 "in per-capita indicators; consider filtering them out "
                 "(`df[~df.is_micro_state]`) for cross-country regressions.")
    out.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main():
    if not IN_PATH.exists():
        sys.exit(f"Input not found: {IN_PATH}. Run build_panel.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)
    inds = indicator_cols(df)
    if not inds:
        sys.exit("No numeric indicator columns found.")

    year = pick_snapshot_year(df, inds)
    print(f"{df['iso3'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}, "
          f"{len(inds)} indicators; snapshot year = {year}")

    coverage_over_time(df, inds, OUT_DIR / "coverage_over_time.png")
    miss = missingness_table(df, inds, OUT_DIR / "missingness.csv")
    distributions(df, inds, year, OUT_DIR / "distributions.png")
    corr = correlation(df, inds, year, OUT_DIR / "correlation.png")
    scatter_relationships(df, year, OUT_DIR / "scatter_relationships.png")
    trends(df, OUT_DIR / "trends.png")
    conv_slope = convergence_plot(df, OUT_DIR / "convergence.png")
    kuznets_turn = kuznets_check(df, year, OUT_DIR / "kuznets.png")
    gov_corr = governance_gdp_table(df, year, OUT_DIR / "governance_corr.csv")
    movers = top_movers(df, OUT_DIR / "top_movers.csv")

    snap = df[df["year"] == year]
    snap[inds].describe().T.to_csv(OUT_DIR / "summary_latest.csv")
    if "income_group" in df.columns:
        (snap.groupby("income_group")[inds].mean().T
         .to_csv(OUT_DIR / "summary_by_income.csv"))

    write_report(df, inds, year, miss, corr, OUT_DIR / "eda_report.md",
                 conv_slope=conv_slope, kuznets_turn=kuznets_turn,
                 gov_corr=gov_corr, movers=movers)
    print(f"Wrote EDA outputs to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
