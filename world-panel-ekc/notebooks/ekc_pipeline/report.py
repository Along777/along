"""
report.py
=========
Figures + the narrative EKC_REPORT.md. Kept separate from the modeling modules
so the analysis can be re-run without regenerating prose, and vice versa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import data as D


# --------------------------------------------------------------------------- #
# figures                                                                       #
# --------------------------------------------------------------------------- #

def fig_target_distribution(df: pd.DataFrame):
    full = D.build_sample(df, "full")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(full[C.TARGET].dropna(), bins=40, color="#4c72b0", edgecolor="white")
    axes[0].set_title("CO2 per capita (levels)"); axes[0].set_xlabel("t CO2 / capita")
    axes[1].hist(np.log1p(full[C.TARGET].dropna()), bins=40, color="#55a868",
                 edgecolor="white")
    axes[1].set_title("log(1 + CO2 per capita)"); axes[1].set_xlabel("log t CO2 / capita")
    fig.suptitle("Target distribution: heavy right skew, petrostate tail")
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "target_distribution.png", dpi=130)
    plt.close(fig)


def fig_within_between(df: pd.DataFrame):
    rows = []
    for col in [C.TARGET, C.GDP, C.MODERATOR]:
        wb = D.within_between(df, col)
        rows.append((col, wb["within_share"], 1 - wb["within_share"]))
    labels = [r[0] for r in rows]
    within = [r[1] for r in rows]
    between = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, within, label="within-country", color="#4c72b0")
    ax.bar(labels, between, bottom=within, label="between-country", color="#c44e52")
    ax.set_ylabel("share of variance"); ax.set_ylim(0, 1)
    ax.set_title("Within- vs between-country variation\n(FE identifies off the within share)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "within_between_variation.png", dpi=130)
    plt.close(fig)


def fig_ekc_curves_by_income(df: pd.DataFrame, het_income: pd.DataFrame):
    full = D.build_sample(df, "full")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, grp in enumerate(C.INCOME_ORDER):
        sub = full[full["income_group"] == grp]
        if sub.empty:
            continue
        # plot on the raw log-GDP scale (C.GDP is now centered) for readability
        ax.scatter(sub[C.GDP_RAW], sub[C.TARGET], s=8, alpha=0.25,
                   color=cmap(i / 3))
    # overlay a pooled quadratic fit line for orientation
    d = full.dropna(subset=[C.GDP_RAW, C.TARGET])
    xs = np.linspace(d[C.GDP_RAW].min(), d[C.GDP_RAW].max(), 100)
    b = np.polyfit(d[C.GDP_RAW], d[C.TARGET], 2)
    ax.plot(xs, np.polyval(b, xs), color="firebrick", linewidth=2,
            label="pooled quadratic fit")
    ax.set_xlabel("log GDP per capita"); ax.set_ylabel("CO2 per capita (t)")
    ax.set_title("CO2 vs income, colored by income group\n"
                 "(pooled curve is U-shaped, not the textbook inverted-U)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "ekc_curves_by_income.png", dpi=130)
    plt.close(fig)


def fig_governance_interaction(df: pd.DataFrame, pooled_res):
    """Fitted CO2-income curve at low vs high regulatory quality, holding
    controls at their means, from the POOLED (between-country) governance model
    -- this is where the moderation is present. The FE contrast (where it
    vanishes) is reported in the governance_moderation table, not this figure."""
    if pooled_res is None:
        return
    full = D.build_sample(df, "full")
    b = pooled_res.params
    # xs spans CENTERED log-GDP (model is fit on centered terms); plot on the
    # raw log-GDP axis by shifting back with the centering mean
    xs = np.linspace(full[C.GDP].min(), full[C.GDP].max(), 100)
    xs_raw = xs + (C.GDP_MEAN or 0.0)
    lo_g, hi_g = full[C.MODERATOR].quantile([0.1, 0.9])
    ctrl_mean = {c: full[c].mean() for c in C.CONTROLS}
    ctrl_contrib = sum(b.get(c, 0.0) * ctrl_mean[c] for c in C.CONTROLS)

    def curve(g):
        return (b.get("const", 0.0) + b[C.GDP] * xs + b[C.GDP_SQ] * xs**2
                + b.get(C.MODERATOR, 0.0) * g + b.get(C.GDP_X_GOV, 0.0) * xs * g
                + ctrl_contrib)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(xs_raw, curve(lo_g), color="firebrick", linewidth=2,
            label=f"low regulatory quality ({lo_g:.1f})")
    ax.plot(xs_raw, curve(hi_g), color="steelblue", linewidth=2,
            label=f"high regulatory quality ({hi_g:.1f})")
    ax.set_xlabel("log GDP per capita"); ax.set_ylabel("fitted CO2 per capita (t)")
    ax.set_title("Between countries, higher governance = lower emissions curve\n"
                 "(pooled/cross-sectional; vanishes within-country — see table)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "governance_interaction.png", dpi=130)
    plt.close(fig)


def fig_decoupling(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, grp in enumerate(C.INCOME_ORDER):
        sub = df[df["income_group"] == grp]
        g = sub.groupby("year")[C.INTENSITY].mean().dropna()
        if len(g):
            ax.plot(g.index, g.values, marker=".", color=cmap(i / 3), label=grp)
    ax.set_xlabel("year"); ax.set_ylabel("CO2 per $1000 GDP (t)")
    ax.set_title("Carbon-intensity decoupling by income group")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "decoupling_trends.png", dpi=130)
    plt.close(fig)


def fig_turning_point_bootstrap(samples: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for name, vals in samples.items():
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        # need at least a few distinct values for a histogram to make sense
        if len(vals) >= 5 and np.ptp(vals) > 0:
            ax.hist(vals, bins=30, alpha=0.5, label=f"{name} (n={len(vals)})")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No identified interior turning point\n"
                          "in the plausible income range\n"
                          "(the EKC peak is not identified)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
    else:
        ax.legend(fontsize=8)
    ax.set_xlabel("bootstrap turning point (GDP per capita, $)")
    ax.set_ylabel("frequency")
    ax.set_title("Block-bootstrap turning-point distribution\n"
                 "(wide/absent mass => the EKC peak is not well identified)")
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "turning_point_bootstrap.png", dpi=130)
    plt.close(fig)


def fig_feature_importance(imp_tbl: pd.DataFrame):
    col = "perm_importance" if "perm_importance" in imp_tbl.columns else "importance"
    top = imp_tbl.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top[col], color="#4c72b0")
    ax.set_xlabel("held-out permutation importance (RMSE increase)")
    ax.set_title("What predicts CO2 per capita? (GBM permutation importance)")
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)


def fig_oos_predictions(preds: pd.DataFrame):
    """Level-task fit. Both models tie a naive persistence baseline here -- the
    plot is a persistence check, not a model win (see the change-task figure)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    color = {g: cmap(i / 3) for i, g in enumerate(C.INCOME_ORDER)}
    for ax, col, title in [(axes[0], "pred_struct", "Structural (country-FE OLS)"),
                            (axes[1], "pred_gbm", "Gradient boosting")]:
        d = preds.dropna(subset=[col])
        for g in C.INCOME_ORDER:
            dg = d[d["income_group"] == g]
            ax.scatter(dg[C.TARGET], dg[col], s=12, alpha=0.5,
                       color=color[g], label=g)
        lim = [0, max(d[C.TARGET].max(), d[col].max()) * 1.05]
        ax.plot(lim, lim, color="grey", linewidth=0.8)
        ax.set_xlabel("actual CO2 pc (t)"); ax.set_title(title)
    axes[0].set_ylabel("predicted CO2 pc (t)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Levels holdout (2019-2024): trivially predictable — "
                 "both models merely tie a naive persistence baseline")
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "oos_predictions.png", dpi=130)
    plt.close(fig)


def fig_oos_change_predictions(preds: pd.DataFrame):
    """The real test: predicting the within-country ANNUAL CHANGE in CO2 pc.
    Naive 'zero change' scores ~0, so any lift is genuine signal."""
    if preds is None or preds.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    color = {g: cmap(i / 3) for i, g in enumerate(C.INCOME_ORDER)}
    for ax, col, title in [(axes[0], "pred_ols", "Structural (differenced OLS)"),
                            (axes[1], "pred_gbm", "Gradient boosting")]:
        d = preds.dropna(subset=[col])
        for g in C.INCOME_ORDER:
            dg = d[d["income_group"] == g]
            ax.scatter(dg["d_co2"], dg[col], s=12, alpha=0.5,
                       color=color[g], label=g)
        lim = [d["d_co2"].min() * 1.05, d["d_co2"].max() * 1.05]
        ax.plot(lim, lim, color="grey", linewidth=0.8)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlabel("actual ΔCO2 pc (t/yr)"); ax.set_title(title)
    axes[0].set_ylabel("predicted ΔCO2 pc (t/yr)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Change holdout (2019-2024): the real test — "
                 "structural OLS extracts more signal than the black box")
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "oos_change_predictions.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report                                                                        #
# --------------------------------------------------------------------------- #

def write_report(spec_ladder, diagnostics, vif, turning_points, het_income,
                  decoupling, het_region, robustness, gov_moderation,
                  se_comparison, val, out):
    L = []
    L.append("# Does growth have to cost the planet?\n")
    L.append("## Environmental Kuznets Curve, governance, and carbon "
             "decoupling — a panel study\n")
    L.append("*Country-year panel, ~150 countries, 1996–2024. All estimates "
             "are observational associations, not causal effects.*\n")

    L.append("## Research questions & hypotheses\n")
    L.append("- **H1 (EKC):** does CO2 per capita follow an inverted-U in "
             "income — rising, then falling past a turning point?")
    L.append("- **H2 (governance):** does regulatory quality shift/flatten "
             "that relationship?")
    L.append("- **H3 (decoupling):** are countries cutting carbon intensity "
             "within their own borders over time, and does the pace depend on "
             "income level?\n")

    L.append("## Headline finding: the textbook EKC is fragile\n")
    L.append("The specification ladder below walks from the naive pooled "
             "quadratic to two-way fixed effects with controls. **The sign of "
             "the squared-income term is not stable** — the textbook "
             "inverted-U is an artifact of comparing rich and poor countries "
             "cross-sectionally (and of petrostate outliers), not a robust "
             "within-country law.\n")
    L.append("*Note: the income polynomial is mean-centered, so the linear "
             "coefficient is the slope at mean income and the linear/quadratic "
             "terms are not collinear (VIF ~7/1.5, not ~370). Centering leaves "
             "the squared-term coefficient — and this sign-flip finding — "
             "unchanged; it only makes the coefficients interpretable. "
             "`se_type` gives the standard-error family per row.*\n")
    L.append(spec_ladder.to_markdown(index=False))
    L.append("\n![ekc curves](ekc_curves_by_income.png)\n")
    L.append("![target dist](target_distribution.png)\n")

    L.append("## Turning point: is there even a peak?\n")
    L.append("For each FE quadratic we estimate the implied turning point and "
             "a **country block-bootstrap** 95% CI. `inverted_u_boot_share` is "
             "the fraction of bootstrap resamples that produced an inverted-U "
             "at all — low values mean the peak is not identified.\n")
    L.append(turning_points.to_markdown(index=False))
    L.append("\n![turning point](turning_point_bootstrap.png)\n")

    L.append("## H2: governance moderation is a *between-country* pattern\n")
    L.append("The `log_gdp_pc × regulatory_quality` interaction is strongly "
             "negative in the pooled (between-country) specification — higher-"
             "governance countries sit on a lower emissions-income curve — but "
             "it **collapses to near-zero and insignificant once country fixed "
             "effects are added**. Governance moderation is a cross-country "
             "stylized fact, not a within-country lever, the same fragility "
             "theme as the EKC shape itself.\n")
    L.append(gov_moderation.to_markdown(index=False))
    L.append("\n![governance](governance_interaction.png)\n")

    L.append("## H3: decoupling is real but income-conditional\n")
    L.append("Within-country trend in carbon intensity (CO2 per $1000 GDP), "
             "by income group. Negative = decoupling.\n")
    L.append(decoupling.to_markdown(index=False))
    L.append("\n![decoupling](decoupling_trends.png)\n")
    L.append("\n### EKC curvature by income group\n")
    L.append(het_income.to_markdown(index=False))
    L.append("\n### EKC curvature by region\n")
    L.append(het_region.to_markdown(index=False))

    L.append("\n## Why these estimator & SE choices? (diagnostics)\n")
    L.append("**FE vs RE:** the operative test is **Mundlak** (a joint Wald "
             "test that country means of the regressors are zero); it rejects, "
             "so fixed effects are required. The classic Hausman degenerates to "
             "a negative statistic here and is reported only for completeness.\n")
    L.append("**Standard errors:** the FE rows use **Driscoll-Kraay** SEs "
             "because the residual diagnostics below show both serial "
             "correlation and cross-sectional dependence — clustered SEs would "
             "handle the former but not the latter.\n")
    L.append(diagnostics.to_markdown(index=False))
    L.append("\nMulticollinearity (VIF) on the **centered** regressor set — the "
             "income terms are now well-conditioned (contrast the ~370 VIF on "
             "the raw, uncentered polynomial):\n")
    L.append(vif.to_markdown(index=False))
    if se_comparison is not None:
        L.append("\n**Clustered vs Driscoll-Kraay for the headline model "
                 "(rung 5).** Point estimates are identical; only the inference "
                 "changes:\n")
        L.append(se_comparison.to_markdown())

    L.append("\n## Robustness battery\n")
    L.append("Re-estimating the headline curvature under perturbations. If the "
             "EKC shape were real it should survive; instead the squared term "
             "flips/loses significance depending on outliers, DV, and period — "
             "reinforcing the headline.\n")
    L.append(robustness.to_markdown(index=False))

    L.append("\n## Out-of-sample prediction: levels are trivial, changes are "
             "the real test\n")
    L.append(f"Temporal holdout: train ≤ {C.SPLIT_YEAR}, test "
             f"{C.SPLIT_YEAR+1}–2024. Every model in a task is scored on the "
             "**same** held-out rows.\n")

    L.append("\n### Task 1 — levels (a persistence check, not a model win)\n")
    L.append("CO2 per capita is a near-random-walk (within-country "
             "autocorrelation ≈ 0.996), so a **naive baseline that carries each "
             "country's last training value forward** already nails it. Neither "
             "the structural model nor the tuned gradient-boosting benchmark "
             "beats that baseline — a high level-R² here reflects persistence, "
             "not skill.\n")
    L.append(val["level_metrics"].to_markdown(index=False))
    L.append("\n![oos](oos_predictions.png)\n")

    L.append("\n### Task 2 — changes (where models earn their keep)\n")
    L.append("Predicting the **within-country annual change** in CO2 per capita "
             "from first-differenced drivers. The naive 'predict zero change' "
             "baseline scores ≈ 0, so any lift is genuine signal. Here the "
             "**parsimonious structural (differenced-OLS) model beats the "
             "flexible black box** — structure, not flexibility, extracts the "
             "signal.\n")
    if val["change_metrics"] is not None:
        L.append(val["change_metrics"].to_markdown(index=False))
        L.append("\n![oos change](oos_change_predictions.png)\n")

    L.append("\n### What drives emissions? (held-out permutation importance)\n")
    L.append("Permutation importance on the held-out set (RMSE increase when a "
             "feature is shuffled) — more honest than impurity importance, "
             "which splits arbitrarily across collinear terms. The GBM is given "
             "raw features and its hyperparameters are tuned on an inner "
             "temporal fold (chosen params: "
             f"`{val['gbm_best']}`).\n")
    L.append(val["level_imp"].head(12).to_markdown(index=False))
    L.append("\n![importance](feature_importance.png)\n")

    L.append("\n## Limitations\n")
    L.append("- **Observational, not causal.** No instrument is used; reverse "
             "causality and omitted variables (energy prices, industrial "
             "structure) are not addressed.")
    L.append("- **EKC estimates are descriptive.** Turning points from a "
             "quadratic are sensitive to functional form and to the "
             "petrostate tail (see robustness).")
    L.append("- **Governance data gap.** WGI indicators are structurally "
             "missing for 1997/1999/2001 (biennial pre-2002).")
    L.append("- **Decoupling ≠ sufficiency.** A falling CO2/GDP ratio can "
             "coexist with rising absolute emissions if GDP grows faster than "
             "intensity falls.")
    L.append("- **Micro-states excluded** from all samples (per the data "
             "audit); results describe the ~150 larger economies.")

    out.write_text("\n".join(L), encoding="utf-8")
