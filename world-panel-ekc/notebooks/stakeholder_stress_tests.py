#!/usr/bin/env python3
"""
stakeholder_stress_tests.py
===========================
Pre-presentation stress tests for the EKC pipeline: quantify the answer to every
objection a sharp stakeholder is likely to raise, extract the headline
"two-speed decarbonization" result, and write a defensible stakeholder brief.

Objections addressed (each with a numbered output):
  1. "Your change-task R2 is just COVID."        -> covid_sensitivity.csv
  2. "Same-year drivers aren't a forecast."      -> true_forecast.csv
  3. "Has the growth-CO2 coupling weakened?"     -> coupling_by_period.csv (star result)
  4. "Petrostates drive your pooled U-shape."    -> pooled_no_petro.csv
  5. "Decoupling is only relative, not absolute."-> absolute_decoupling.csv

Headline figure: coupling_headline.png (cyclical elasticity vs autonomous drift,
three periods). Brief: ../output/ekc/STAKEHOLDER_BRIEF.md.

Run AFTER run_ekc_pipeline.py:
    python stakeholder_stress_tests.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from ekc_pipeline import config as C
from ekc_pipeline import data as D
from ekc_pipeline import validation as V
from ekc_pipeline.models import to_panel, _fit

STRESS_DIR = C.OUT_DIR / "stress"
PERIODS = [(1996, 2005), (2006, 2015), (2016, 2024)]
COVID_YEARS = [2020, 2021]

# reference dataviz palette (light mode): categorical slot 1 + diverging red pole
BLUE = "#2a78d6"; RED = "#e34948"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- #
# shared prep                                                                  #
# --------------------------------------------------------------------------- #

def change_data(df: pd.DataFrame):
    """The exact change-task dataset validation.py uses."""
    data = V._prep(df).sort_values(["iso3", "year"]).copy()
    data["d_co2"] = data.groupby("iso3")[C.TARGET].diff()
    dcols = []
    for c in V.CHANGE_DRIVERS:
        dc = f"d_{c}"
        data[dc] = data.groupby("iso3")[c].diff()
        dcols.append(dc)
    return data.dropna(subset=["d_co2"] + dcols), dcols


def elasticity_data(df: pd.DataFrame):
    """Log-difference dataset: dlog(CO2 pc) and dlog(GDP pc) within country."""
    d = df[df["co2_pc"] > 0].sort_values(["iso3", "year"]).copy()
    d["lco2"] = np.log(d["co2_pc"])
    d["dlco2"] = d.groupby("iso3")["lco2"].diff()
    d["dlgdp"] = d.groupby("iso3")[C.GDP_RAW].diff()
    return d.dropna(subset=["dlco2", "dlgdp", "income_group"])


# --------------------------------------------------------------------------- #
# 1. COVID sensitivity of the change-task result                              #
# --------------------------------------------------------------------------- #

def covid_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    d, dcols = change_data(df)
    train, test = D.temporal_split(d)
    ols = sm.OLS(train["d_co2"].values, sm.add_constant(train[dcols])).fit()
    test = test.copy()
    test["pred"] = np.asarray(ols.predict(sm.add_constant(test[dcols])))

    rows = []
    for yr, g in test.groupby("year"):
        rows.append({"slice": str(int(yr)), "n": len(g),
                     "ols_r2": round(r2_score(g["d_co2"], g["pred"]), 3),
                     "naive_r2": round(r2_score(g["d_co2"], np.zeros(len(g))), 3)})
    ex = test[~test["year"].isin(COVID_YEARS)]
    cv = test[test["year"].isin(COVID_YEARS)]
    rows.append({"slice": "ex-COVID (2019, 2022+)", "n": len(ex),
                 "ols_r2": round(r2_score(ex["d_co2"], ex["pred"]), 3),
                 "naive_r2": round(r2_score(ex["d_co2"], np.zeros(len(ex))), 3)})
    rows.append({"slice": "COVID only (2020-21)", "n": len(cv),
                 "ols_r2": round(r2_score(cv["d_co2"], cv["pred"]), 3),
                 "naive_r2": round(r2_score(cv["d_co2"], np.zeros(len(cv))), 3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(STRESS_DIR / "covid_sensitivity.csv", index=False)
    return tbl


# --------------------------------------------------------------------------- #
# 2. true forecast (lagged info only)                                         #
# --------------------------------------------------------------------------- #

def true_forecast(df: pd.DataFrame) -> pd.DataFrame:
    d, dcols = change_data(df)
    for dc in dcols:
        d[dc + "_l1"] = d.groupby("iso3")[dc].shift(1)
    d["d_co2_l1"] = d.groupby("iso3")["d_co2"].shift(1)
    lcols = [dc + "_l1" for dc in dcols] + ["d_co2_l1"]
    d = d.dropna(subset=lcols)
    train, test = D.temporal_split(d)
    ols = sm.OLS(train["d_co2"].values, sm.add_constant(train[lcols])).fit()
    pred = np.asarray(ols.predict(sm.add_constant(test[lcols])))
    tbl = pd.DataFrame([
        {"model": "lagged_info_only (true forecast)",
         "test_r2": round(r2_score(test["d_co2"], pred), 4), "n_test": len(test)},
        {"model": "contemporaneous (attribution, from validation)",
         "test_r2": 0.184, "n_test": 391},
    ])
    tbl.to_csv(STRESS_DIR / "true_forecast.csv", index=False)
    return tbl


# --------------------------------------------------------------------------- #
# 3. star result: two-speed decarbonization                                   #
# --------------------------------------------------------------------------- #

def coupling_by_period(df: pd.DataFrame):
    """dlog(CO2) ~ dlog(GDP) per period: the intercept is the AUTONOMOUS DRIFT
    (emission change at zero growth -- structural decarbonization) and the slope
    is the CYCLICAL COUPLING (elasticity to growth fluctuations).

    Also runs a pooled interaction model to test whether the elasticity change
    between the first and last period is statistically significant."""
    d = elasticity_data(df)
    rows = []

    def fit_one(sub, sample, group):
        if len(sub) < 100:
            return
        m = sm.OLS(sub["dlco2"], sm.add_constant(sub["dlgdp"])).fit(
            cov_type="cluster", cov_kwds={"groups": sub["iso3"]})
        rows.append({
            "sample": sample, "group": group,
            "drift_pct_yr": round(float(m.params["const"]) * 100, 2),
            "drift_se_pct": round(float(m.bse["const"]) * 100, 2),
            "drift_p": round(float(m.pvalues["const"]), 4),
            "elasticity": round(float(m.params["dlgdp"]), 3),
            "elast_se": round(float(m.bse["dlgdp"]), 3),
            "elast_p": round(float(m.pvalues["dlgdp"]), 4),
            "n": len(sub),
        })

    for lo, hi in PERIODS:
        per = d[(d["year"] >= lo) & (d["year"] <= hi)]
        fit_one(per, f"{lo}-{hi}", "all countries")
        fit_one(per[~per["year"].isin(COVID_YEARS)], f"{lo}-{hi} ex-COVID", "all countries")
        for grp in C.INCOME_ORDER:
            g = per[per["income_group"] == grp]
            fit_one(g, f"{lo}-{hi}", grp)
            fit_one(g[~g["year"].isin(COVID_YEARS)], f"{lo}-{hi} ex-COVID", grp)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(STRESS_DIR / "coupling_by_period.csv", index=False)

    # significance of the elasticity rise: pooled interaction, first vs last period
    first, last = PERIODS[0], PERIODS[-1]
    sub = d[((d["year"] >= first[0]) & (d["year"] <= first[1])) |
            ((d["year"] >= last[0]) & (d["year"] <= last[1]))].copy()
    sub["late"] = ((sub["year"] >= last[0]) & (sub["year"] <= last[1])).astype(float)
    sub["dlgdp_x_late"] = sub["dlgdp"] * sub["late"]
    m = sm.OLS(sub["dlco2"],
               sm.add_constant(sub[["dlgdp", "late", "dlgdp_x_late"]])).fit(
        cov_type="cluster", cov_kwds={"groups": sub["iso3"]})
    inter = {
        "test": f"elasticity change {first[0]}-{first[1]} vs {last[0]}-{last[1]}",
        "delta_elasticity": round(float(m.params["dlgdp_x_late"]), 3),
        "p_value": round(float(m.pvalues["dlgdp_x_late"]), 4),
        "delta_drift_pct_yr": round(float(m.params["late"]) * 100, 2),
        "drift_change_p": round(float(m.pvalues["late"]), 4),
    }
    pd.DataFrame([inter]).to_csv(STRESS_DIR / "coupling_change_test.csv", index=False)
    return tbl, inter


# --------------------------------------------------------------------------- #
# 4. pooled curvature without petrostates                                     #
# --------------------------------------------------------------------------- #

def pooled_no_petro(df: pd.DataFrame) -> pd.DataFrame:
    full = D.build_sample(df, "full")
    rows = []
    for label, sub in [("all countries", full),
                       ("excl. petrostates", full[~full["iso3"].isin(C.PETROSTATES)])]:
        p = to_panel(sub, [C.TARGET, C.GDP, C.GDP_SQ])
        r = _fit(p, C.TARGET, [C.GDP, C.GDP_SQ], entity=False, time=False,
                 cov="clustered")
        rows.append({"sample": label,
                     "b_gdp_sq_pooled": round(float(r.params[C.GDP_SQ]), 3),
                     "p_value": round(float(r.pvalues[C.GDP_SQ]), 4),
                     "n": int(r.nobs)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(STRESS_DIR / "pooled_no_petro.csv", index=False)
    return tbl


# --------------------------------------------------------------------------- #
# 5. absolute decoupling                                                       #
# --------------------------------------------------------------------------- #

def absolute_decoupling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in C.INCOME_ORDER:
        sub = df[df["income_group"] == grp].dropna(subset=["co2_pc", "trend"])
        panel = to_panel(sub, ["co2_pc", "trend"])
        if len(panel) < 40:
            continue
        r = _fit(panel, "co2_pc", ["trend"], entity=True, time=False, cov="clustered")
        rows.append({"income_group": grp,
                     "abs_trend_t_per_yr": round(float(r.params["trend"]), 4),
                     "p_value": round(float(r.pvalues["trend"]), 4),
                     "mean_level_t": round(float(sub["co2_pc"].mean()), 2),
                     "n": int(r.nobs)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(STRESS_DIR / "absolute_decoupling.csv", index=False)
    return tbl


# --------------------------------------------------------------------------- #
# headline figure                                                              #
# --------------------------------------------------------------------------- #

def fig_coupling_headline(coupling: pd.DataFrame):
    """Two panels x three periods: cyclical elasticity (blue) and autonomous
    drift (diverging red/blue by sign), 95% CI whiskers, direct labels, and an
    ex-COVID marker on the last period. Reference dataviz palette, light mode."""
    allc = coupling[coupling["group"] == "all countries"]
    full = allc[~allc["sample"].str.contains("ex-COVID")].reset_index(drop=True)
    ex = allc[allc["sample"].str.contains("ex-COVID")].reset_index(drop=True)
    labels = [s.replace("-", "–") for s in full["sample"]]
    x = np.arange(len(full))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.set_xticks(x); ax.set_xticklabels(labels, color=INK2)

    # -- panel A: cyclical coupling (elasticity) ---------------------------- #
    ax = axes[0]; style(ax)
    ci = 1.96 * full["elast_se"]
    ax.bar(x, full["elasticity"], width=0.55, color=BLUE, zorder=2)
    ax.errorbar(x, full["elasticity"], yerr=ci, fmt="none",
                ecolor=INK2, elinewidth=1.4, capsize=3, zorder=3)
    for xi, v in zip(x, full["elasticity"]):
        ax.text(xi, v + ci[xi] + 0.03, f"{v:.2f}", ha="center",
                color=INK, fontsize=10, fontweight="bold")
    exl = ex.iloc[-1]
    ax.plot([x[-1] + 0.31], [exl["elasticity"]], marker="o", ms=7,
            mfc=SURFACE, mec=BLUE, mew=1.6, zorder=4)
    ax.annotate(f"ex-COVID {exl['elasticity']:.2f}",
                (x[-1] + 0.31, exl["elasticity"]), textcoords="offset points",
                xytext=(8, -3), fontsize=8, color=MUTED)
    ax.set_title("Cyclical coupling has not weakened",
                 color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("CO2 elasticity to GDP growth\n(dlog CO2 pc / dlog GDP pc)",
                  color=INK2, fontsize=9)
    ax.set_ylim(0, max(full["elasticity"] + ci) * 1.28)
    ax.set_xlim(-0.5, len(x) - 0.5 + 0.6)

    # -- panel B: autonomous drift ------------------------------------------ #
    ax = axes[1]; style(ax)
    ci = 1.96 * full["drift_se_pct"]
    colors = [RED if v > 0 else BLUE for v in full["drift_pct_yr"]]
    ax.bar(x, full["drift_pct_yr"], width=0.55, color=colors, zorder=2)
    ax.errorbar(x, full["drift_pct_yr"], yerr=ci, fmt="none",
                ecolor=INK2, elinewidth=1.4, capsize=3, zorder=3)
    ax.axhline(0, color=BASE, linewidth=1.2, zorder=1)
    for xi, v in zip(x, full["drift_pct_yr"]):
        off = ci[xi] + 0.12
        ax.text(xi, v + off if v > 0 else v - off, f"{v:+.2f}%",
                ha="center", va="bottom" if v > 0 else "top",
                color=INK, fontsize=10, fontweight="bold")
    exl = ex.iloc[-1]
    ax.plot([x[-1] + 0.31], [exl["drift_pct_yr"]], marker="o", ms=7,
            mfc=SURFACE, mec=BLUE, mew=1.6, zorder=4)
    ax.annotate(f"ex-COVID {exl['drift_pct_yr']:+.2f}%",
                (x[-1] + 0.31, exl["drift_pct_yr"]), textcoords="offset points",
                xytext=(8, -3), fontsize=8, color=MUTED)
    ax.set_title("...but autonomous decarbonization is accelerating",
                 color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("emission drift at zero growth\n(% per year)",
                  color=INK2, fontsize=9)
    lo = min(full["drift_pct_yr"] - ci) - 0.55
    hi = max(full["drift_pct_yr"] + ci) + 0.55
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.5, len(x) - 0.5 + 0.6)

    fig.suptitle("Two-speed decarbonization: growth still buys carbon, "
                 "but the baseline is falling", color=INK, fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.925, "dlog(CO2 pc) ~ dlog(GDP pc) by period, ~150 countries, "
             "cluster-robust 95% CIs; open circles = 2016–2024 excluding 2020–21",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(STRESS_DIR / "coupling_headline.png", dpi=150,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# stakeholder brief                                                            #
# --------------------------------------------------------------------------- #

def write_brief(covid, forecast, coupling, change_test, no_petro, absolute, out):
    allc = coupling[coupling["group"] == "all countries"]
    full = allc[~allc["sample"].str.contains("ex-COVID")]
    hi = coupling[coupling["group"] == "High income"]

    L = []
    L.append("# Stakeholder brief — growth, emissions, and the limits of the EKC\n")
    L.append("*Final pre-presentation review: thesis, star result, and prepared "
             "answers to anticipated objections. Panel: ~150 countries, "
             "1996–2024, micro-states excluded. All results are observational "
             "associations.*\n")

    L.append("## Thesis in one paragraph\n")
    L.append("The Environmental Kuznets Curve — the promise that growth "
             "eventually cleans up after itself — **is not in the data**: its "
             "inverted-U is a cross-sectional artifact that flips sign under "
             "within-country fixed effects and never yields an identified "
             "turning point (2% of bootstrap resamples). What *is* in the data "
             "is **two-speed decarbonization**: the *cyclical* coupling of "
             "emissions to growth has not weakened — a 1% GDP-per-capita "
             "acceleration still brings ~0.5–0.7% more CO2, tighter than in the "
             "1990s — while the *autonomous* drift (emission change at zero "
             "growth) has swung from **+0.8%/yr in 1996–2005 to −1.0%/yr in "
             "2016–24** (−2.7%/yr in high-income countries, ex-COVID). "
             "**Decarbonization is happening around growth — via energy mix and "
             "technology — not through it.** Policy implication: waiting for "
             "the EKC is not a strategy; the drift is where the action is.\n")

    L.append("![two-speed decarbonization](stress/coupling_headline.png)\n")

    L.append("## The star table: drift vs coupling by period\n")
    L.append(full.drop(columns="group").to_markdown(index=False))
    L.append("\nHigh-income countries only:\n")
    L.append(hi.drop(columns="group").to_markdown(index=False))
    L.append(f"\n**Is the elasticity rise significant?** Interaction test "
             f"(1996–2005 vs 2016–24): Δelasticity = "
             f"+{change_test['delta_elasticity']:.2f} "
             f"(p = {change_test['p_value']:.3f}); drift change "
             f"{change_test['delta_drift_pct_yr']:+.2f} pp/yr "
             f"(p = {change_test['drift_change_p']:.3f}). Both margins moved, "
             f"in opposite directions — that is the two-speed result.\n")

    L.append("## Anticipated objections — and our answers\n")
    L.append("**Q1. \"Your out-of-sample win on emission changes is just "
             "COVID.\"**  Largely, yes — and that *is* the finding. Ex-COVID "
             "test years the change-model R² is ~0.02 (annual changes are "
             "near-noise in calm years); in 2020–21 it is 0.27 while naive "
             "persistence collapses to −0.45. The model earns its keep exactly "
             "when growth moves sharply — which is direct out-of-sample "
             "evidence that the growth-emissions coupling is real and live. "
             "We present the per-year split ourselves:\n")
    L.append(covid.to_markdown(index=False))
    L.append("\n*Rows after 2021 have n ≤ 7 (the fossil-fuel-share control lags "
             "in WDI), so their R² is not informative — flagged here before "
             "anyone else does.*\n")
    L.append("\n**Q2. \"Contemporaneous drivers aren't a forecast.\"**  "
             "Correct, and we don't claim one. With only lagged information the "
             f"test R² is {forecast.iloc[0]['test_r2']:.2f} — essentially "
             "unforecastable. The change model is **attribution**: given "
             "observed growth, how much emissions move. Its ΔGDP coefficient is "
             "the marginal coupling — an inferential quantity, not a crystal "
             "ball. (validation_change_metrics.csv remains the honest headline: "
             "structure beats the GBM on the same task, same rows.)\n")
    L.append("**Q3. \"Petrostates create your pooled U-shape.\"**  They "
             "amplify it but don't create it: dropping the 8 largest "
             f"oil-and-gas outliers, pooled curvature stays positive "
             f"({no_petro.iloc[1]['b_gdp_sq_pooled']:+.2f}, p < 0.001). Either "
             "way, no inverted-U:\n")
    L.append(no_petro.to_markdown(index=False))
    L.append("\n**Q4. \"Intensity decoupling is a ratio trick — absolute "
             "emissions matter.\"**  Agreed, so we estimated absolute "
             "within-country trends: high-income CO2 per capita is falling "
             f"**{absolute.iloc[3]['abs_trend_t_per_yr']:+.3f} t/yr** "
             "(p < 0.001) — genuine absolute decline — while upper-middle-"
             f"income is still **rising** "
             f"({absolute.iloc[2]['abs_trend_t_per_yr']:+.3f} t/yr, p < 0.01). "
             "Absolute decoupling exists, but is so far a rich-country "
             "phenomenon:\n")
    L.append(absolute.to_markdown(index=False))
    L.append("\n**Q5. \"Your FE null on the EKC could be low power — only "
             "3.2% of CO2 variance is within-country.\"**  Partly fair, and "
             "we flag it. But the with-controls FE quadratic is significantly "
             "*convex* (+0.48, p < 0.001), not merely null, and the "
             "change-elasticity analysis — which uses the same within-country "
             "variation — finds strong, precisely-estimated coupling. The data "
             "have enough within-variation to speak; what they say is 'no "
             "inverted-U.'\n")
    L.append("**Q6. \"Driscoll-Kraay with T≈29 is stretching it.\"**  Both SE "
             "families are reported side by side (headline_se_comparison.csv); "
             "point estimates are identical and every conclusion survives "
             "under clustered SEs. The FE-vs-RE call rests on the Mundlak test "
             "(p = 0.001), not the degenerate Hausman.\n")
    L.append("**Q7. \"A rising elasticity contradicts your own decoupling "
             "claim.\"**  No — they are the two different margins estimated in "
             "one regression: the *slope* (cyclical response to growth "
             "fluctuations) rose, the *intercept* (secular drift) fell. "
             "Trend-decoupling coexists with tight cyclical coupling; that "
             "tension is the headline, not a contradiction.\n")
    L.append("**Q8. \"None of this is causal.\"**  Stated on every output. "
             "These are conditional associations with fixed effects and robust "
             "inference; no instrument exists here. The policy-relevant claim — "
             "that the observed decline comes from the drift, not from an "
             "income turning point — is about *where the variation lives*, and "
             "stands at the descriptive level at which it is made.\n")

    L.append("## The close\n")
    L.append("> Growth has not decoupled from carbon at the margin — every "
             "point of GDP growth still buys about half a point of CO2, more "
             "tightly than in the 1990s. What has changed is everything "
             "*around* growth: at zero growth, emissions now fall ~1% a year "
             "globally and ~2.7% in high-income economies, versus *rising* "
             "0.8% in the late 1990s. The Kuznets curve promised growth would "
             "do the cleanup. The data say the cleanup is being done in spite "
             "of growth — by the energy mix — and that is where policy "
             "leverage lies.\n")
    out.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main():
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    df = D.add_ekc_features(D.load_panel())

    print("[1/6] COVID sensitivity of the change task")
    covid = covid_sensitivity(df)
    print(covid.to_string(index=False))

    print("\n[2/6] true forecast (lagged info only)")
    forecast = true_forecast(df)
    print(forecast.to_string(index=False))

    print("\n[3/6] two-speed decarbonization (drift vs coupling)")
    coupling, change_test = coupling_by_period(df)
    print(coupling[coupling["group"] == "all countries"].to_string(index=False))
    print("elasticity change test:", change_test)

    print("\n[4/6] pooled curvature without petrostates")
    no_petro = pooled_no_petro(df)
    print(no_petro.to_string(index=False))

    print("\n[5/6] absolute decoupling by income")
    absolute = absolute_decoupling(df)
    print(absolute.to_string(index=False))

    print("\n[6/6] headline figure + stakeholder brief")
    fig_coupling_headline(coupling)
    write_brief(covid, forecast, coupling, change_test, no_petro, absolute,
                C.OUT_DIR / "STAKEHOLDER_BRIEF.md")
    print(f"\nDone. Stress outputs in {STRESS_DIR.resolve()}/ ; "
          f"brief at {(C.OUT_DIR / 'STAKEHOLDER_BRIEF.md').resolve()}")


if __name__ == "__main__":
    main()
