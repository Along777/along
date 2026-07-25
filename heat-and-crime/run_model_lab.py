"""
Grok Model Lab v2 — baseline lock, tuning (G0–G7), falsification (F1–F7).

  py -3 run_model_lab.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from models import (
    bh_fdr,
    fit_dlag,
    fit_ols,
    fit_ols_cluster,
    fmt_pct,
    pct_ci,
    standardized_effect,
)
from panel import CTRL_M2, DATA, HAC_DEFAULT, OUTCOMES, ROOT, build_panel

warnings.filterwarnings("ignore", category=FutureWarning)

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

COL = {
    "violent": "#e34948",
    "property": "#2a78d6",
    "total": "#52514e",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def style_ax(ax):
    ax.grid(True, axis="y", color=COL["grid"], lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def nw_rule(T: int) -> int:
    return int(np.floor(4 * (T / 100.0) ** (2 / 9)))


# ----- G0–G7 (core) ---------------------------------------------------------

def g0(dfa, hac):
    print("\n=== G0 baseline ===")
    out = {}
    for y in OUTCOMES:
        res = fit_ols(y, "tmax10", dfa, hac=hac)
        out[y] = pct_ci(res, "tmax10")
        out[y]["aic"] = float(res.aic)
        print(f"  {y}: {fmt_pct(out[y])}")
    assert abs(out["violent"]["pct"] - 5.6) < 0.35
    print("  PASS G0")
    return out


def g1_hac(dfa):
    print("\n=== G1 HAC grid ===")
    T = len(dfa)
    nw = nw_rule(T)
    lags = sorted(set([3, 5, 7, 9, 10, 14, 21, 28, nw]))
    grid = []
    for y in ["violent", "property"]:
        for L in lags:
            res = fit_ols(y, "tmax10", dfa, hac=L)
            ci = pct_ci(res, "tmax10")
            grid.append(
                {
                    "y": y,
                    "maxlags": L,
                    "se_beta": float(res.bse["tmax10"]),
                    "ci_width_pct": ci["hi"] - ci["lo"],
                    **{k: ci[k] for k in ("pct", "lo", "hi")},
                }
            )
    preferred = max(nw, 9)
    headline = {y: pct_ci(fit_ols(y, "tmax10", dfa, hac=preferred), "tmax10") for y in OUTCOMES}
    print(f"  preferred maxlags={preferred} (NW thumb={nw})")
    fig, ax = plt.subplots(figsize=(8, 4))
    for y, c in [("violent", COL["violent"]), ("property", COL["property"])]:
        rows = sorted([r for r in grid if r["y"] == y], key=lambda r: r["maxlags"])
        ax.plot([r["maxlags"] for r in rows], [r["ci_width_pct"] for r in rows], "o-", color=c, label=y)
    ax.axvline(preferred, ls="--", color=COL["ink2"])
    ax.set_xlabel("HAC maxlags")
    ax.set_ylabel("CI width (pp)")
    ax.set_title("G1: HAC bandwidth")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "g1_hac_bandwidth.png", dpi=140)
    plt.close(fig)
    return {
        "nw_rule_of_thumb": nw,
        "preferred_maxlags": preferred,
        "grid": grid,
        "headline_preferred_hac": headline,
    }


def g2_treatment(dfa, hac):
    print("\n=== G2 treatment race (+ per-SD) ===")
    # within-ym SD of each treatment for standardized effect
    specs = [
        ("tmax10", "tmax10"),
        ("app_tmax10", "app_tmax10"),
        ("tmean10", "tmean10"),
        ("hi10 (heat index)", "hi10"),
        ("tmax10 + rh", "tmax10 + rh_c"),
        ("tmax10 + weekend×", "tmax10 + tmax10_x_weekend"),
        ("tmax10 + summer×", "tmax10 + tmax10_x_summer"),
    ]
    rows = []
    for name, xterms in specs:
        for y in ["violent", "property"]:
            res = fit_ols(y, xterms, dfa, hac=hac)
            main = next(
                t
                for t in ["tmax10", "app_tmax10", "tmean10", "hi10"]
                if t in res.params.index
            )
            # SD of the raw treatment column in analysis window
            col = main
            x_sd = float(dfa.groupby("ym")[col].transform(lambda s: s).std())  # crude
            # better: within-ym residual SD
            x_sd = float(
                (dfa[col] - dfa.groupby("ym")[col].transform("mean")).std()
            )
            d = pct_ci(res, main)
            d_sd = standardized_effect(res, main, x_sd)
            inter = None
            for it in ["tmax10_x_weekend", "tmax10_x_summer"]:
                if it in res.params.index:
                    inter = pct_ci(res, it)
            rows.append(
                {
                    "spec": name,
                    "y": y,
                    "main_term": main,
                    "effect_per_10F": d,
                    "effect_per_1sd": d_sd,
                    "x_within_ym_sd": x_sd,
                    "interaction": inter,
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                }
            )
            print(
                f"  {y:8s} {name:22s} /10F {fmt_pct(d)} | /1sd {fmt_pct(d_sd)} | AIC={res.aic:.1f}"
            )
    single = [
        r
        for r in rows
        if r["y"] == "violent"
        and r["spec"] in ("tmax10", "app_tmax10", "tmean10", "hi10 (heat index)")
    ]
    best = min(single, key=lambda r: r["aic"])
    # figure AIC
    fig, ax = plt.subplots(figsize=(8.5, 4))
    v = [r for r in rows if r["y"] == "violent"]
    ax.barh(range(len(v)), [r["aic"] for r in v], color=COL["violent"])
    ax.set_yticks(range(len(v)), [r["spec"] for r in v], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AIC")
    ax.set_title("G2: treatment AIC (violent)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "g2_treatment_aic.png", dpi=140)
    plt.close(fig)
    return {
        "rows": rows,
        "preferred_single_spec": best["spec"],
        "preferred_single_effect": best["effect_per_10F"],
        "note": "Public treatment remains tmax10; per-1sd column compares rulers fairly.",
    }


def g3_dlag(dfa, hac):
    print("\n=== G3 distributed lag ===")
    out = {}
    for y in ["violent", "property"]:
        recs = []
        for L in range(0, 8):
            res, terms, cum = fit_dlag(y, L, dfa, hac=hac)
            same = pct_ci(res, "tmax10")
            recs.append(
                {
                    "L": L,
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "same_day": same,
                    "cumulative": cum,
                    "lag_pct": {t: float(100 * np.expm1(res.params[t])) for t in terms},
                }
            )
            print(f"  {y} L={L}: AIC={res.aic:.1f} cum={fmt_pct(cum)}")
        star = min(recs, key=lambda r: r["aic"])
        out[y] = {
            "grid": recs,
            "L_star_aic": star["L"],
            "L_star_bic": min(recs, key=lambda r: r["bic"])["L"],
            "selected": star,
        }
        print(f"  → {y} L*={star['L']}")
    # plot violent profile
    sel = out["violent"]["selected"]
    terms = ["tmax10"] + [f"tmax10_L{k}" for k in range(1, sel["L"] + 1)]
    pcts = [sel["lag_pct"][t] for t in terms]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(pcts)), pcts, color=COL["violent"])
    ax.axhline(0, color=COL["axis"])
    ax.set_xlabel("lag k")
    ax.set_ylabel("% per +10°F")
    ax.set_title(f"G3 violent DLAG L*={sel['L']} cum {fmt_pct(sel['cumulative'])}")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "g3_dlag_violent.png", dpi=140)
    plt.close(fig)
    return out


def g4_shape(dfa, hac):
    print("\n=== G4 spline + kink ===")
    spline_rows = []
    for y in ["violent", "property"]:
        for df_ in range(3, 9):
            res = fit_ols(y, f"cr(tmax, df={df_})", dfa, hac=hac)
            spline_rows.append({"y": y, "df": df_, "aic": float(res.aic)})
    best_spline = {
        y: min([r for r in spline_rows if r["y"] == y], key=lambda r: r["aic"])
        for y in ["violent", "property"]
    }
    kink_rows = []
    for y in ["violent", "property"]:
        for tau in range(50, 95, 5):
            d = dfa.copy()
            d["tmax_low10"] = np.minimum(d["tmax"], tau) / 10.0
            d["tmax_high10"] = np.maximum(d["tmax"] - tau, 0.0) / 10.0
            res = fit_ols(y, "tmax_low10 + tmax_high10", d, hac=hac)
            kink_rows.append(
                {
                    "y": y,
                    "tau": tau,
                    "aic": float(res.aic),
                    "slope_below": pct_ci(res, "tmax_low10"),
                    "slope_above": pct_ci(res, "tmax_high10"),
                }
            )
    best_kink = {
        y: min([r for r in kink_rows if r["y"] == y], key=lambda r: r["aic"])
        for y in ["violent", "property"]
    }
    for y in ["violent", "property"]:
        print(
            f"  {y}: spline df*={best_spline[y]['df']} kink τ*={best_kink[y]['tau']} "
            f"below={fmt_pct(best_kink[y]['slope_below'])} above={fmt_pct(best_kink[y]['slope_above'])}"
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y in zip(axes, ["violent", "property"]):
        rows = [r for r in kink_rows if r["y"] == y]
        ax.plot([r["tau"] for r in rows], [r["aic"] for r in rows], "o-", color=COL[y])
        ax.axvline(best_kink[y]["tau"], ls="--", color=COL["ink2"])
        ax.set_title(f"{y} τ*={best_kink[y]['tau']}°F")
        ax.set_xlabel("τ °F")
        ax.set_ylabel("AIC")
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "g4_kink_aic.png", dpi=140)
    plt.close(fig)
    return {
        "spline_grid": spline_rows,
        "best_spline_df": best_spline,
        "kink_grid": kink_rows,
        "best_kink": best_kink,
    }


def g5_stability(dfa, hac):
    print("\n=== G5 LOYO + rolling ===")
    loyo = []
    for yr in sorted(dfa["year"].unique()):
        sub = dfa[dfa["year"] != yr]
        d = pct_ci(fit_ols("violent", "tmax10", sub, hac=hac), "tmax10")
        loyo.append({"drop_year": int(yr), "days": int(len(sub)), **d})
        print(f"  drop {yr}: {fmt_pct(d)}")
    rolling = []
    for start in range(2015, 2024):
        end = start + 2
        sub = dfa[(dfa["year"] >= start) & (dfa["year"] <= end)]
        d = pct_ci(fit_ols("violent", "tmax10", sub, hac=hac), "tmax10")
        rolling.append({"window": f"{start}-{end}", "start": start, "end": end, **d})
    base = pct_ci(fit_ols("violent", "tmax10", dfa, hac=hac), "tmax10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    est = [r["pct"] for r in loyo]
    lo = [r["lo"] for r in loyo]
    hi = [r["hi"] for r in loyo]
    ys = [r["drop_year"] for r in loyo]
    ax.errorbar(est, ys, xerr=[np.array(est) - lo, np.array(hi) - est], fmt="o", color=COL["violent"])
    ax.axvline(base["pct"], ls="--", color=COL["ink2"])
    ax.set_title("LOYO violent")
    style_ax(ax)
    ax = axes[1]
    ax.plot([r["start"] for r in rolling], [r["pct"] for r in rolling], "o-", color=COL["violent"])
    ax.fill_between(
        [r["start"] for r in rolling],
        [r["lo"] for r in rolling],
        [r["hi"] for r in rolling],
        alpha=0.15,
        color=COL["violent"],
    )
    ax.axhline(base["pct"], ls="--", color=COL["ink2"])
    ax.set_title("Rolling 3y")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "g5_stability.png", dpi=140)
    plt.close(fig)
    return {"baseline_violent": base, "loyo": loyo, "rolling_3y": rolling}


def g6_count(dfa, treatment="tmax10"):
    print("\n=== G6 Poisson + week cluster ===")
    out = {"treatment_used": treatment}
    for y in ["violent", "property"]:
        pois = smf.glm(
            f"{y} ~ {treatment} + {CTRL_M2}", data=dfa, family=sm.families.Poisson()
        ).fit(cov_type="HC1")
        out[f"poisson_{y}"] = pct_ci(pois, treatment)
        ols_w = fit_ols_cluster(y, treatment, dfa, "week_id")
        out[f"ols_week_cluster_{y}"] = pct_ci(ols_w, treatment)
        print(
            f"  {y}: Poisson {fmt_pct(out[f'poisson_{y}'])} | "
            f"week {fmt_pct(out[f'ols_week_cluster_{y}'])}"
        )
    return out


def g7_ridge(dfa):
    print("\n=== G7 residual ridge ===")
    d = dfa.copy()
    for col in ["violent", "property", "tmax10", "app_tmax10", "rh_c", "rain", "snow_day"]:
        d[f"{col}_dm"] = d[col] - d.groupby("ym")[col].transform("mean")
    results = {}
    for y in ["violent", "property"]:
        feats = ["tmax10_dm", "app_tmax10_dm", "rh_c_dm", "rain_dm", "snow_day_dm"]
        X, yy = d[feats].values, d[f"{y}_dm"].values
        best = {"alpha": 100.0, "mean_r2": -1e9, "folds": []}
        n = len(d)
        for a in [0.1, 1.0, 10.0, 50.0, 100.0]:
            r2s = []
            for cut in [int(n * f) for f in (0.5, 0.6, 0.7, 0.8)]:
                Xtr, Xte = X[:cut], X[cut : cut + int(n * 0.1)]
                ytr, yte = yy[:cut], yy[cut : cut + int(n * 0.1)]
                if len(Xte) < 50:
                    continue
                sc = StandardScaler()
                m = Ridge(alpha=a).fit(sc.fit_transform(Xtr), ytr)
                r2s.append(float(r2_score(yte, m.predict(sc.transform(Xte)))))
            mean_r2 = float(np.mean(r2s)) if r2s else -1e9
            if mean_r2 > best["mean_r2"]:
                best = {"alpha": a, "mean_r2": mean_r2, "folds": r2s}
        results[y] = {
            "best_alpha": best["alpha"],
            "cv_r2_mean": best["mean_r2"],
            "corr_tmax_residual": float(np.corrcoef(d["tmax10_dm"], yy)[0, 1]),
        }
        print(f"  {y}: corr={results[y]['corr_tmax_residual']:.3f} CV R²={best['mean_r2']:.4f}")
    return results


# ----- F1–F7 falsification ---------------------------------------------------

def f1_leads(dfa, hac):
    print("\n=== F1 leads placebo ===")
    # same-day + two leads
    res = fit_ols("violent", "tmax10 + tmax10_F1 + tmax10_F2", dfa.dropna(subset=["tmax10_F1", "tmax10_F2"]), hac=hac)
    out = {
        "same_day": pct_ci(res, "tmax10"),
        "lead1": pct_ci(res, "tmax10_F1"),
        "lead2": pct_ci(res, "tmax10_F2"),
        "aic": float(res.aic),
    }
    # pass if |lead| much smaller than same-day
    out["pass"] = abs(out["lead1"]["pct"]) < 0.5 * abs(out["same_day"]["pct"]) and abs(
        out["lead2"]["pct"]
    ) < 0.5 * abs(out["same_day"]["pct"])
    print(
        f"  same={fmt_pct(out['same_day'])} L1={fmt_pct(out['lead1'])} "
        f"L2={fmt_pct(out['lead2'])} PASS={out['pass']}"
    )
    return out


def f2_battery(dfa, hac):
    print("\n=== F2 battery / components ===")
    out = {}
    for y in ["battery", "assault", "robbery", "theft", "violent", "property"]:
        # skip zero days for log — battery always >0 in practice
        use = dfa[dfa[y] > 0].copy()
        res = fit_ols(y, "tmax10", use, hac=hac)
        out[y] = pct_ci(res, "tmax10")
        print(f"  {y}: {fmt_pct(out[y])}")
    out["pass"] = out["battery"]["pct"] > 3.0 and out["battery"]["p"] < 0.01
    print(f"  PASS battery large={out['pass']}")
    return out


def f3_heat_index(dfa, hac):
    print("\n=== F3 heat index race ===")
    rows = []
    for name, term in [("tmax10", "tmax10"), ("app_tmax10", "app_tmax10"), ("hi10", "hi10")]:
        res = fit_ols("violent", term, dfa, hac=hac)
        x_sd = float((dfa[term] - dfa.groupby("ym")[term].transform("mean")).std())
        rows.append(
            {
                "term": name,
                "per_10": pct_ci(res, term),
                "per_1sd": standardized_effect(res, term, x_sd),
                "aic": float(res.aic),
            }
        )
        print(f"  {name}: /10 {fmt_pct(rows[-1]['per_10'])} /sd {fmt_pct(rows[-1]['per_1sd'])} AIC={res.aic:.1f}")
    return {"rows": rows, "best_aic": min(rows, key=lambda r: r["aic"])["term"]}


def f4_inference(dfa):
    print("\n=== F4 inference alternatives ===")
    rows = []
    for label, fit in [
        ("HAC-9", lambda: fit_ols("violent", "tmax10", dfa, hac=9)),
        ("HAC-14", lambda: fit_ols("violent", "tmax10", dfa, hac=14)),
        ("week-cluster", lambda: fit_ols_cluster("violent", "tmax10", dfa, "week_id")),
        ("HC1", lambda: smf.ols(f"np.log(violent) ~ tmax10 + {CTRL_M2}", data=dfa).fit(cov_type="HC1")),
    ]:
        res = fit()
        d = pct_ci(res, "tmax10")
        rows.append({"spec": label, **d})
        print(f"  {label}: {fmt_pct(d)}")
    pcts = [r["pct"] for r in rows]
    return {"rows": rows, "pass": max(pcts) - min(pcts) < 0.5}


def f5_permutation(dfa, hac, B: int = 300, seed: int = 42):
    print(f"\n=== F5 permutation null (B={B}) ===")
    rng = np.random.default_rng(seed)
    obs = pct_ci(fit_ols("violent", "tmax10", dfa, hac=hac), "tmax10")
    # Within-ym demeaned residual regression; shuffle tmax within ym for null.
    d = dfa.copy()
    d["logv"] = np.log(d["violent"])
    d["logv_dm"] = d["logv"] - d.groupby("ym")["logv"].transform("mean")
    d["x_dm"] = d["tmax10"] - d.groupby("ym")["tmax10"].transform("mean")
    obs_b = float(np.polyfit(d["x_dm"], d["logv_dm"], 1)[0])
    null = []
    x = d["x_dm"].values.copy()
    y = d["logv_dm"].values
    ym = d["ym"].values
    for _ in range(B):
        # shuffle x within ym
        x_perm = x.copy()
        for g in np.unique(ym):
            idx = np.where(ym == g)[0]
            x_perm[idx] = rng.permutation(x_perm[idx])
        b = float(np.polyfit(x_perm, y, 1)[0])
        null.append(b)
    null = np.asarray(null)
    p = float(np.mean(np.abs(null) >= abs(obs_b)))
    print(f"  obs β(resid)={obs_b:.4f} null mean={null.mean():.4f} p_perm={p:.4f}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(null, bins=40, color=COL["muted"], edgecolor="white")
    ax.axvline(obs_b, color=COL["violent"], lw=2, label="observed")
    ax.set_title(f"F5: within-ym permutation null (p={p:.3f})")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "f5_permutation.png", dpi=140)
    plt.close(fig)
    return {
        "obs_residual_beta": obs_b,
        "obs_pct_hac": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p_permutation": p,
        "B": B,
        "pass": p < 0.01,
    }


def f6_influence(dfa, hac):
    print("\n=== F6 influence cuts ===")
    base = pct_ci(fit_ols("violent", "tmax10", dfa, hac=hac), "tmax10")
    cuts = {
        "baseline": dfa,
        "drop_unrest": dfa[dfa["unrest"] == 0],
        "drop_nyd": dfa[dfa["is_nyd"] == 0],
        "drop_top1pct_total": dfa[dfa["total"] < dfa["total"].quantile(0.99)],
        "drop_2020": dfa[dfa["year"] != 2020],
    }
    rows = []
    for name, sub in cuts.items():
        d = pct_ci(fit_ols("violent", "tmax10", sub, hac=hac), "tmax10")
        rows.append(
            {
                "cut": name,
                "days": int(len(sub)),
                **d,
                "delta_pp": d["pct"] - base["pct"],
            }
        )
        print(f"  {name}: {fmt_pct(d)} Δ={d['pct']-base['pct']:+.2f}pp")
    return {"baseline": base, "rows": rows, "pass": all(abs(r["delta_pp"]) < 1.0 for r in rows)}


def f7_fdr(dfa, hac):
    print("\n=== F7 multi-outcome FDR ===")
    ys = ["total", "violent", "property", "battery", "assault", "robbery", "theft"]
    effects = []
    pvals = []
    for y in ys:
        use = dfa[dfa[y] > 0]
        res = fit_ols(y, "tmax10", use, hac=hac)
        d = pct_ci(res, "tmax10")
        effects.append({"outcome": y, **d})
        pvals.append(d["p"])
    adj = bh_fdr(pvals)
    for e, a in zip(effects, adj):
        e["p_fdr_bh"] = a
        print(f"  {e['outcome']}: {fmt_pct(e)} p={e['p']:.2e} p_FDR={a:.2e}")
    return {"rows": effects, "pass": all(e["p_fdr_bh"] < 0.05 for e in effects if e["outcome"] in ("violent", "battery", "property", "total"))}


def main():
    print("Building panel...")
    dfa, _ = build_panel()
    print(f"  n={len(dfa)} mean violent={dfa['violent'].mean():.1f}")

    results = {
        "meta": {
            "window": ["2015-01-01", "2025-12-31"],
            "n_days": int(len(dfa)),
            "hac_default": HAC_DEFAULT,
            "ctrl": CTRL_M2,
            "version": 2,
        }
    }

    # use preferred hac from g1
    results["G1_hac"] = g1_hac(dfa)
    hac = int(results["G1_hac"]["preferred_maxlags"])
    results["G0_baseline"] = g0(dfa, hac=14)  # lock vs original Claude
    results["G0_baseline_hac_preferred"] = g0(dfa, hac=hac)
    results["G2_treatment"] = g2_treatment(dfa, hac=hac)
    results["G3_dlag"] = g3_dlag(dfa, hac=hac)
    results["G4_shape"] = g4_shape(dfa, hac=hac)
    results["G5_stability"] = g5_stability(dfa, hac=hac)
    results["G6_count"] = g6_count(dfa, treatment="tmax10")
    results["G7_residual_ml"] = g7_ridge(dfa)

    results["F1_leads"] = f1_leads(dfa, hac=hac)
    results["F2_battery"] = f2_battery(dfa, hac=hac)
    results["F3_heat_index"] = f3_heat_index(dfa, hac=hac)
    results["F4_inference"] = f4_inference(dfa)
    results["F5_permutation"] = f5_permutation(dfa, hac=hac, B=300)
    results["F6_influence"] = f6_influence(dfa, hac=hac)
    results["F7_fdr"] = f7_fdr(dfa, hac=hac)

    g0v = results["G0_baseline"]["violent"]
    g3v = results["G3_dlag"]["violent"]["selected"]
    g4k = results["G4_shape"]["best_kink"]["violent"]
    g4s = results["G4_shape"]["best_spline_df"]["violent"]
    loyo = results["G5_stability"]["loyo"]

    # weekend / summer from g2
    g2v = [r for r in results["G2_treatment"]["rows"] if r["y"] == "violent"]
    def _inter(substr):
        for r in g2v:
            if substr in r["spec"] and r.get("interaction"):
                return r["interaction"]
        return None

    results["summary"] = {
        "baseline_m2_violent_same_day": fmt_pct(g0v),
        "preferred_hac_maxlags": hac,
        "preferred_hac_violent_ci": fmt_pct(results["G1_hac"]["headline_preferred_hac"]["violent"]),
        "preferred_treatment": results["G2_treatment"]["preferred_single_spec"],
        "dlag_L_star_violent": g3v["L"],
        "dlag_same_day_violent": fmt_pct(g3v["same_day"]),
        "dlag_cumulative_violent": fmt_pct(g3v["cumulative"]),
        "kink_tau_violent": g4k["tau"],
        "kink_slope_below": fmt_pct(g4k["slope_below"]),
        "kink_slope_above": fmt_pct(g4k["slope_above"]),
        "best_spline_df_violent": g4s["df"],
        "loyo_min": min(r["pct"] for r in loyo),
        "loyo_max": max(r["pct"] for r in loyo),
        "weekend_interaction_violent": _inter("weekend"),
        "summer_interaction_violent": _inter("summer"),
        "falsification": {
            "F1_leads_pass": results["F1_leads"]["pass"],
            "F2_battery_pass": results["F2_battery"]["pass"],
            "F5_permutation_p": results["F5_permutation"]["p_permutation"],
            "F5_pass": results["F5_permutation"]["pass"],
            "F6_influence_pass": results["F6_influence"]["pass"],
            "F7_fdr_pass": results["F7_fdr"]["pass"],
        },
    }

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    out = DATA / "model_lab_results.json"
    out.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
    print(f"\nWrote {out}")
    print("SUMMARY:", json.dumps(results["summary"], indent=2, default=_default))
    return results


if __name__ == "__main__":
    main()
