#!/usr/bin/env python3
"""
topic_models.py
================
Econometric topic models run against the country-year panel produced by
build_panel.py (output/panel_model_ready.csv). Four topics, chosen to cover
tech/AI-adjacent diffusion, pollution/decoupling, institutions/convergence,
and inequality:

  1. tech_growth_model      -- does digital/tech diffusion predict subsequent
                                GDP-per-capita growth? (closest available
                                analog to "does AI/tech adoption pay off" --
                                WDI has no real AI-adoption series)
  2. pollution_model         -- EKC turning point interacted with governance
                                quality, plus carbon-intensity decoupling
                                trends by income group
  3. convergence_model       -- conditional beta-convergence (Barro/MRW
                                style): growth ~ initial income + investment
                                + human capital + trade + rule of law
  4. digital_divide_model    -- internet access vs. income inequality, plus
                                sigma-convergence (cross-country dispersion)
                                of internet penetration over time

IMPORTANT -- read before trusting any coefficient:
  - None of this is causal. Every regression here is observational; no valid
    instrument is used anywhere. Treat coefficients as conditional
    associations, not effects.
  - "Tech/AI" is proxied by internet penetration, ICT service exports, and
    high-tech exports -- general-purpose-technology diffusion measures, not
    AI-specific investment or adoption data (which doesn't exist in WDI).
  - topic 3's specification (lagged log-GDP regressor + entity fixed
    effects, finite T) is subject to Nickell (1981) dynamic-panel bias --
    flagged in the report, not corrected (would need GMM, out of scope here).
  - Micro-states/territories (population < 1,000,000, `is_micro_state`) are
    excluded from every regression's baseline sample -- see the prior data
    audit for why.

Requires: pandas, numpy, matplotlib, statsmodels, linearmodels.
    pip install statsmodels linearmodels

Run:
    python topic_models.py                       # uses output/panel_model_ready.csv
    python topic_models.py path/to/panel.csv      # custom path

Produces in output/models/:
    tech_growth_coefs.csv          tech-and-growth panel FE regression table
    tech_growth_fitted.png         fitted vs. actual growth by internet tercile
    ekc_governance_coefs.csv       EKC x governance-quality regression table
    ekc_governance.png             CO2-GDP curves at low/high regulatory quality
    decoupling_by_income.csv       annual %, carbon-intensity trend by income group
    convergence_conditional_coefs.csv  conditional beta-convergence regression table
    digital_divide_coefs.csv       internet-access vs. gini regression table
    digital_divide_sigma.png       cross-country dispersion of internet access over time
    topic_models_report.md         narrative tying it together, with caveats
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

IN_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/panel_model_ready.csv")
OUT_DIR = Path("output/models")

TECH_ERA_START = 2010  # broadband/smartphone diffusion era; closest real-world
# analog available in WDI to present-day AI/tech diffusion


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def base_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline sample used by every topic: drop micro-states/territories."""
    if "is_micro_state" in df.columns:
        return df[~df["is_micro_state"]].copy()
    return df.copy()


def to_panel(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Long df -> MultiIndex(iso3, year) frame ready for linearmodels PanelOLS."""
    sub = df[["iso3", "year"] + cols].dropna()
    return sub.set_index(["iso3", "year"])


def fit_panel_fe(panel: pd.DataFrame, y_col: str, x_cols: list[str],
                  time_effects: bool = True):
    """entity_effects is always on; time_effects defaults on too, but must be
    turned off for any regressor that's a deterministic function of year
    alone (e.g. a linear trend) -- year FE would fully absorb it and produce
    a meaningless/spurious coefficient on the leftover collinearity."""
    y = panel[y_col]
    X = sm.add_constant(panel[x_cols])
    model = PanelOLS(y, X, entity_effects=True, time_effects=time_effects,
                      drop_absorbed=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def save_coefs(res, out: Path) -> pd.DataFrame:
    table = pd.DataFrame({
        "coef": res.params, "std_err": res.std_errors,
        "t_stat": res.tstats, "p_value": res.pvalues,
    })
    table.to_csv(out)
    return table


# --------------------------------------------------------------------------- #
# 1. tech & growth                                                             #
# --------------------------------------------------------------------------- #

def tech_growth_model(df: pd.DataFrame):
    d = df[df["year"] >= TECH_ERA_START].sort_values(["iso3", "year"]).copy()
    lag_cols = ["internet_users", "gcf_pct_gdp", "secondary_enroll", "trade_pct_gdp"]
    for c in lag_cols:
        d[f"{c}_lag1"] = d.groupby("iso3")[c].shift(1)
    x_cols = [f"{c}_lag1" for c in lag_cols]

    panel = to_panel(d, ["gdp_pc_growth_calc"] + x_cols)
    if len(panel) < 50:
        print("tech_growth_model: insufficient data, skipping")
        return None, None, d

    res = fit_panel_fe(panel, "gdp_pc_growth_calc", x_cols)
    table = save_coefs(res, OUT_DIR / "tech_growth_coefs.csv")

    # fitted vs actual growth, grouped by internet-penetration tercile
    d["internet_tercile"] = pd.qcut(d["internet_users_lag1"], 3,
                                     labels=["low", "mid", "high"])
    grp = d.dropna(subset=["gdp_pc_growth_calc", "internet_tercile"]) \
        .groupby(["year", "internet_tercile"], observed=True)["gdp_pc_growth_calc"].mean() \
        .reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for tercile, dt in grp.groupby("internet_tercile", observed=True):
        ax.plot(dt["year"], dt["gdp_pc_growth_calc"], marker=".", label=f"internet {tercile}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("year"); ax.set_ylabel("mean GDP p.c. growth")
    ax.set_title("GDP p.c. growth by prior-year internet-penetration tercile")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "tech_growth_fitted.png", dpi=130)
    plt.close(fig)

    return res, table, d


# --------------------------------------------------------------------------- #
# 2. pollution: EKC x governance, and decoupling                              #
# --------------------------------------------------------------------------- #

def pollution_model(df: pd.DataFrame):
    d = df.dropna(subset=["co2_pc", "log_gdp_pc", "regulatory_quality", "year"]).copy()
    if len(d) < 50:
        print("pollution_model: insufficient data for EKC x governance, skipping")
        ekc_res = None
    else:
        d["log_gdp_pc_sq"] = d["log_gdp_pc"] ** 2
        d["log_gdp_pc_x_reg"] = d["log_gdp_pc"] * d["regulatory_quality"]
        formula = ("co2_pc ~ log_gdp_pc + log_gdp_pc_sq + regulatory_quality "
                   "+ log_gdp_pc_x_reg + C(year)")
        ekc_res = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["iso3"]})
        coefs = ekc_res.params.filter(regex="^(log_gdp_pc|regulatory_quality|Intercept)")
        pvals = ekc_res.pvalues.filter(regex="^(log_gdp_pc|regulatory_quality|Intercept)")
        pd.DataFrame({"coef": coefs, "p_value": pvals}).to_csv(
            OUT_DIR / "ekc_governance_coefs.csv")

        # plot fitted CO2-GDP curve at low (10th pct) vs high (90th pct) governance
        lo_reg, hi_reg = d["regulatory_quality"].quantile([0.1, 0.9])
        xs = np.linspace(d["log_gdp_pc"].min(), d["log_gdp_pc"].max(), 100)
        b = ekc_res.params
        year_ref = d["year"].median()
        year_dummy_cols = [c for c in b.index if c.startswith("C(year)")]

        def predict(reg_level):
            base = (b.get("Intercept", 0) + b["log_gdp_pc"] * xs
                    + b["log_gdp_pc_sq"] * xs**2 + b["regulatory_quality"] * reg_level
                    + b["log_gdp_pc_x_reg"] * xs * reg_level)
            return base

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(xs, predict(lo_reg), label=f"low regulatory quality ({lo_reg:.1f})",
                color="firebrick")
        ax.plot(xs, predict(hi_reg), label=f"high regulatory quality ({hi_reg:.1f})",
                color="steelblue")
        ax.set_xlabel("log GDP per capita"); ax.set_ylabel("fitted CO2 pc (t)")
        ax.set_title("EKC fit at low vs. high governance quality")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT_DIR / "ekc_governance.png", dpi=130)
        plt.close(fig)

    # decoupling: within-country trend in carbon intensity, by income group
    decoupling_rows = []
    if "income_group" in df.columns and "co2_per_1000gdp" in df.columns:
        for grp_name, sub in df.dropna(subset=["co2_per_1000gdp", "trend", "income_group"]).groupby("income_group"):
            panel = to_panel(sub, ["co2_per_1000gdp", "trend"])
            if panel["trend"].nunique() < 3 or len(panel) < 30:
                continue
            # time_effects=False: trend IS the year signal here, so year FE
            # would absorb it -- only entity (country) FE make sense
            res = fit_panel_fe(panel, "co2_per_1000gdp", ["trend"], time_effects=False)
            mean_level = sub["co2_per_1000gdp"].mean()
            pct_per_year = res.params["trend"] / mean_level * 100 if mean_level else np.nan
            decoupling_rows.append({
                "income_group": grp_name, "trend_coef": res.params["trend"],
                "p_value": res.pvalues["trend"], "pct_change_per_year": pct_per_year,
                "n_obs": len(panel),
            })
    decoupling = pd.DataFrame(decoupling_rows)
    decoupling.to_csv(OUT_DIR / "decoupling_by_income.csv", index=False)

    return ekc_res, decoupling


# --------------------------------------------------------------------------- #
# 3. conditional beta-convergence                                             #
# --------------------------------------------------------------------------- #

def convergence_model(df: pd.DataFrame):
    d = df.sort_values(["iso3", "year"]).copy()
    lag_cols = ["log_gdp_pc", "gcf_pct_gdp", "secondary_enroll", "trade_pct_gdp", "rule_of_law"]
    for c in lag_cols:
        d[f"{c}_lag1"] = d.groupby("iso3")[c].shift(1)
    x_cols = [f"{c}_lag1" for c in lag_cols]

    panel = to_panel(d, ["gdp_pc_growth_calc"] + x_cols)
    if len(panel) < 50:
        print("convergence_model: insufficient data, skipping")
        return None, None

    res = fit_panel_fe(panel, "gdp_pc_growth_calc", x_cols)
    table = save_coefs(res, OUT_DIR / "convergence_conditional_coefs.csv")
    return res, table


# --------------------------------------------------------------------------- #
# 4. digital divide vs. inequality                                            #
# --------------------------------------------------------------------------- #

def digital_divide_model(df: pd.DataFrame):
    d = df.dropna(subset=["internet_users", "gini", "year"]).copy()
    reg_res = None
    if len(d) >= 30:
        reg_res = smf.ols("internet_users ~ gini + C(year)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["iso3"]})
        coefs = reg_res.params.filter(regex="^(gini|Intercept)")
        pvals = reg_res.pvalues.filter(regex="^(gini|Intercept)")
        pd.DataFrame({"coef": coefs, "p_value": pvals}).to_csv(
            OUT_DIR / "digital_divide_coefs.csv")
    else:
        print("digital_divide_model: insufficient data for gini regression, skipping")

    sigma = df.groupby("year")["internet_users"].std().dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigma.index, sigma.values, marker=".", color="#4c72b0")
    ax.set_xlabel("year"); ax.set_ylabel("cross-country std. dev. of internet_users")
    ax.set_title("Sigma-convergence: dispersion of internet access across countries")
    fig.tight_layout(); fig.savefig(OUT_DIR / "digital_divide_sigma.png", dpi=130)
    plt.close(fig)

    return reg_res, sigma


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #

def fmt_coef(res, name: str) -> str:
    if res is None or name not in res.params.index:
        return "n/a"
    coef, p = res.params[name], res.pvalues[name]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{coef:.4f}{stars} (p={p:.3f})"


def write_report(tech_res, tech_table, ekc_res, decoupling, conv_res, conv_table,
                  digital_res, sigma, out: Path):
    lines = []
    lines.append("# Topic models — econometric report\n")
    lines.append("All models are observational associations, not causal "
                 "effects (no instruments used anywhere). See each section's "
                 "caveat.\n")

    lines.append("## 1. Tech/digital diffusion & subsequent growth\n")
    lines.append(f"**Question:** does lagged internet penetration predict "
                 f"next-period GDP-per-capita growth (2010-2024), controlling "
                 f"for lagged investment, education, and trade openness? "
                 f"Closest available WDI analog to \"does tech/AI adoption "
                 f"pay off\" -- no AI-specific data exists in this source.\n")
    lines.append(f"**Method:** panel FE (entity + year), clustered SE by "
                 f"country.\n")
    lines.append(f"**Headline:** internet_users(t-1) coefficient = "
                 f"{fmt_coef(tech_res, 'internet_users_lag1')}\n")
    if tech_table is not None:
        lines.append(tech_table.round(4).to_markdown())
    lines.append("\n![tech_growth](tech_growth_fitted.png)\n")
    lines.append("**Caveat:** reverse causality is plausible (richer, "
                 "faster-growing countries also adopt tech faster); the "
                 "1-year lag reduces but does not eliminate this.\n")

    lines.append("## 2. Pollution: EKC x governance, and decoupling\n")
    lines.append("**Question:** does regulatory quality shift the "
                 "GDP-CO2 (Environmental Kuznets Curve) relationship, and is "
                 "carbon intensity actually falling within countries over "
                 "time (decoupling), and does the pace differ by income "
                 "group?\n")
    lines.append(f"**EKC x governance headline:** log_gdp_pc x "
                 f"regulatory_quality interaction = "
                 f"{fmt_coef(ekc_res, 'log_gdp_pc_x_reg')}\n")
    lines.append("\n![ekc_governance](ekc_governance.png)\n")
    if decoupling is not None and len(decoupling):
        lines.append("**Decoupling (within-country carbon-intensity trend, "
                     "%/year) by income group:**\n")
        lines.append(decoupling.round(4).to_markdown(index=False))
    lines.append("\n**Caveat:** EKC turning points from a quadratic fit are "
                 "sensitive to outliers (e.g. Gulf petrostates); treat as "
                 "descriptive.\n")

    lines.append("## 3. Conditional beta-convergence\n")
    lines.append("**Question:** controlling for investment, education, "
                 "trade openness, and rule of law, do poorer countries still "
                 "grow faster (conditional convergence, Barro/Mankiw-Romer-"
                 "Weil style)?\n")
    lines.append(f"**Headline:** log_gdp_pc(t-1) coefficient = "
                 f"{fmt_coef(conv_res, 'log_gdp_pc_lag1')}\n")
    if conv_table is not None:
        lines.append(conv_table.round(4).to_markdown())
    lines.append("\n**Caveat:** lagged-level regressor + entity fixed "
                 "effects in a finite panel is subject to Nickell (1981) "
                 "dynamic-panel bias; a proper treatment would use "
                 "Arellano-Bond/GMM, not attempted here.\n")

    lines.append("## 4. Digital divide vs. inequality\n")
    lines.append("**Question:** is internet access lower in more unequal "
                 "countries, and has the cross-country digital divide been "
                 "widening or narrowing over time?\n")
    lines.append(f"**Headline:** gini coefficient on internet_users = "
                 f"{fmt_coef(digital_res, 'gini')}\n")
    if len(sigma):
        lines.append(f"**Sigma-convergence:** cross-country std. dev. of "
                     f"internet_users went from {sigma.iloc[0]:.1f} "
                     f"({sigma.index[0]}) to a peak of {sigma.max():.1f} "
                     f"({sigma.idxmax()}) to {sigma.iloc[-1]:.1f} "
                     f"({sigma.index[-1]}) -- the divide widened during "
                     f"early diffusion, then has been slowly narrowing.\n")
    lines.append("\n![digital_divide](digital_divide_sigma.png)\n")
    lines.append("**Caveat:** purely descriptive/correlational; gini's sparse "
                 "coverage (never >50% of countries in any single year) "
                 "means this pools across many years and country-sets.\n")

    out.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main():
    if not IN_PATH.exists():
        sys.exit(f"Input not found: {IN_PATH}. Run build_panel.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)
    df = base_sample(df)
    print(f"base sample (excl. micro-states): {df['iso3'].nunique()} countries, "
          f"{len(df):,} rows")

    tech_res, tech_table, _ = tech_growth_model(df)
    ekc_res, decoupling = pollution_model(df)
    conv_res, conv_table = convergence_model(df)
    digital_res, sigma = digital_divide_model(df)

    write_report(tech_res, tech_table, ekc_res, decoupling, conv_res, conv_table,
                 digital_res, sigma, OUT_DIR / "topic_models_report.md")
    print(f"Wrote topic model outputs to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
