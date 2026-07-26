"""Pre-registered Myth A / Myth B tests + secondary stats."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.panels import ANALYSIS_DIR, build_hmd_wide, build_owid_e0, save_panels
from src.paths import OUTPUTS, ensure_dirs

TABLES = OUTPUTS / "tables"
RNG = np.random.default_rng(42)


def bootstrap_median_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "median": None, "ci_low": None, "ci_high": None}
    med = float(np.median(x))
    boots = []
    for _ in range(n_boot):
        sample = RNG.choice(x, size=len(x), replace=True)
        boots.append(np.median(sample))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(len(x)),
        "median": med,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    m = float(np.mean(x))
    boots = [np.mean(RNG.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"n": int(len(x)), "mean": m, "ci_low": float(lo), "ci_high": float(hi)}


def myth_a(panel: pd.DataFrame) -> dict:
    """Among e0 < 40, distribution of expected age at death if survived to 65."""
    low = panel[panel["e0"] < 40].copy()
    exp = low["exp_death_65"].to_numpy()
    gap = low["adult_gap_65"].to_numpy()
    out = {
        "definition": "both-sex HMD summary country-years with e0 < 40",
        "n_country_years": int(len(low)),
        "n_countries": int(low["region_id"].nunique()),
        "countries": sorted(low["region_id"].unique().tolist()),
        "e0": {
            "median": float(np.median(low["e0"])),
            "p10": float(np.quantile(low["e0"], 0.1)),
            "p90": float(np.quantile(low["e0"], 0.9)),
        },
        "imr": {
            "median": float(np.median(low["imr"])),
            "p10": float(np.quantile(low["imr"], 0.1)),
            "p90": float(np.quantile(low["imr"], 0.9)),
        },
        "s_to_65": {
            "median": float(np.median(low["s_to_65"])),
            "p10": float(np.quantile(low["s_to_65"], 0.1)),
            "p90": float(np.quantile(low["s_to_65"], 0.9)),
        },
        "exp_death_65": bootstrap_median_ci(exp),
        "exp_death_65_p10_p90": [
            float(np.quantile(exp, 0.1)),
            float(np.quantile(exp, 0.9)),
        ],
        "share_exp_death_65_ge_70": float(np.mean(exp >= 70)),
        "share_exp_death_65_ge_60": float(np.mean(exp >= 60)),
        "adult_gap_65": bootstrap_median_ci(gap),
        "n_e0_in_30_35": int(((low["e0"] >= 30) & (low["e0"] <= 35)).sum()),
        "reject_myth_A": bool(np.median(exp) > 70 and np.mean(exp >= 70) > 0.95),
    }
    return out


def myth_b(panel: pd.DataFrame) -> dict:
    """Country-level change in e65: post-2000 mean minus pre-1900 mean."""
    pre = panel[panel["year"] < 1900].groupby("region_id")["e65"].mean()
    post = panel[panel["year"] >= 2000].groupby("region_id")["e65"].mean()
    both = pre.to_frame("e65_pre1900").join(post.to_frame("e65_post2000"), how="inner")
    both = both.dropna()
    both["delta_e65"] = both["e65_post2000"] - both["e65_pre1900"]
    # e80 if available
    pre80 = panel[panel["year"] < 1900].groupby("region_id")["e80"].mean()
    post80 = panel[panel["year"] >= 2000].groupby("region_id")["e80"].mean()
    b80 = pre80.to_frame("e80_pre").join(post80.to_frame("e80_post"), how="inner").dropna()
    b80["delta_e80"] = b80["e80_post"] - b80["e80_pre"]

    deltas = both["delta_e65"].to_numpy()
    out = {
        "definition": "countries with both pre-1900 and post-2000 both-sex e65 in HMD summary",
        "n_countries": int(len(both)),
        "country_deltas": both.reset_index().to_dict(orient="records"),
        "delta_e65": bootstrap_mean_ci(deltas),
        "delta_e65_median": float(np.median(deltas)),
        "delta_e65_min": float(np.min(deltas)),
        "delta_e65_max": float(np.max(deltas)),
        "reject_myth_B": bool(np.mean(deltas) > 3),  # substantial multi-year rise
    }
    if len(b80):
        out["delta_e80"] = bootstrap_mean_ci(b80["delta_e80"].to_numpy())
        out["n_countries_e80"] = int(len(b80))
    return out, both


def infant_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in panel.groupby("region_id"):
        g = g.dropna(subset=["e0", "imr"])
        if len(g) < 10:
            continue
        rows.append(
            {
                "region_id": rid,
                "n": len(g),
                "corr_pearson": float(g["e0"].corr(g["imr"])),
                "corr_spearman": float(g["e0"].corr(g["imr"], method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_pearson")


def fe_regression(panel: pd.DataFrame) -> dict:
    """Associational FE model e0 ~ imr + year + C(region)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not installed"}

    d = panel.dropna(subset=["e0", "imr", "year", "region_id"]).copy()
    d["year"] = d["year"].astype(float)
    # demean-style FE via dummies (countries with enough obs)
    counts = d["region_id"].value_counts()
    keep = counts[counts >= 20].index
    d = d[d["region_id"].isin(keep)]
    if d["region_id"].nunique() < 2:
        return {"error": "insufficient countries"}
    model = smf.ols("e0 ~ imr + year + C(region_id)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["region_id"]}
    )
    return {
        "n": int(model.nobs),
        "n_countries": int(d["region_id"].nunique()),
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "params": {
            "imr": float(model.params.get("imr", np.nan)),
            "year": float(model.params.get("year", np.nan)),
            "Intercept": float(model.params.get("Intercept", np.nan)),
        },
        "bse": {
            "imr": float(model.bse.get("imr", np.nan)),
            "year": float(model.bse.get("year", np.nan)),
        },
        "pvalue": {
            "imr": float(model.pvalues.get("imr", np.nan)),
            "year": float(model.pvalues.get("year", np.nan)),
        },
        "note": "Cluster-robust SE by region_id; associational not causal",
    }


def concordance(panel: pd.DataFrame) -> dict:
    """HMD summary e0 vs OWID e0 on overlapping region-year (best-effort region match)."""
    owid = build_owid_e0()
    # OWID often uses ISO3; HMD summary uses mixed IDs
    # Match on common short codes and a small alias map
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
        "PRT": "PRT",
        "AUT": "AUT",
        "FRANCE:_TOTAL_POPULATION": "FRA",
        "UK:_ENGLAND_&_WALES_TOTAL_POPULATION": "GBR",  # imperfect
        "USA": "USA",
    }
    h = panel.copy()
    h["iso_try"] = h["region_id"].map(lambda x: alias.get(x, x if len(str(x)) == 3 else None))
    h = h.dropna(subset=["iso_try"])
    o = owid[owid["source_id"] == "owid_le_hmd_unwpp"].copy()
    o["iso_try"] = o["region_id"].astype(str)
    m = h.merge(
        o[["iso_try", "year", "life_expectancy"]].rename(columns={"life_expectancy": "e0_owid"}),
        on=["iso_try", "year"],
        how="inner",
    )
    if m.empty:
        return {"n": 0, "note": "no overlap"}
    err = m["e0"] - m["e0_owid"]
    return {
        "n": int(len(m)),
        "n_countries": int(m["iso_try"].nunique()),
        "corr": float(m["e0"].corr(m["e0_owid"])),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_bias_hmd_minus_owid": float(np.mean(err)),
    }


def sweden_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    s = panel[panel["region_id"] == "SWE"].copy()
    years = [1751, 1800, 1850, 1900, 1950, 2000, 2020]
    rows = []
    for y in years:
        r = s[s["year"] == y]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(
            {
                "year": int(y),
                "e0": round(float(r["e0"]), 2),
                "imr": round(float(r["imr"]), 2),
                "e65": round(float(r["e65"]), 2),
                "exp_death_65": round(float(r["exp_death_65"]), 2),
                "s_to_65": round(float(r["s_to_65"]), 4),
                "adult_gap_65": round(float(r["adult_gap_65"]), 2),
            }
        )
    return pd.DataFrame(rows)


def run_all() -> None:
    ensure_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ensure panels exist
    panel_path = ANALYSIS_DIR / "hmd_summary_wide_both.parquet"
    if not panel_path.exists():
        save_panels()
    panel = pd.read_parquet(panel_path)

    # Myth A/B on all both-sex (full multi-country including subseries — document)
    # Prefer totals-ish for Myth B country list: use full panel but for deltas use all regions
    a = myth_a(panel)
    b, b_df = myth_b(panel)
    corr = infant_correlations(panel)
    reg = fe_regression(panel)
    conc = concordance(panel)
    snap = sweden_snapshot(panel)

    # also Myth A on primary allowlist only
    primary = panel[panel["in_primary_allowlist"]]
    a_primary = myth_a(primary) if len(primary[primary["e0"] < 40]) else {}

    results = {
        "myth_a_all": a,
        "myth_a_primary_allowlist": a_primary,
        "myth_b": {k: v for k, v in b.items() if k != "country_deltas"},
        "fe_regression": reg,
        "concordance_hmd_vs_owid": conc,
        "infant_corr_summary": {
            "n_countries": int(len(corr)),
            "median_pearson": float(corr["corr_pearson"].median()) if len(corr) else None,
            "mean_pearson": float(corr["corr_pearson"].mean()) if len(corr) else None,
        },
    }

    (TABLES / "myth_tests_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    b_df.reset_index().to_csv(TABLES / "myth_b_country_deltas.csv", index=False)
    corr.to_csv(TABLES / "infant_e0_correlations_by_country.csv", index=False)
    snap.to_csv(TABLES / "sweden_snapshot.csv", index=False)

    # Myth A low-e0 raw for tables
    low = panel[panel["e0"] < 40][
        ["region_id", "label", "year", "e0", "imr", "e65", "exp_death_65", "s_to_65", "adult_gap_65"]
    ]
    low.to_csv(TABLES / "myth_a_low_e0_country_years.csv", index=False)

    print(json.dumps(results, indent=2))
    print(f"Wrote tables under {TABLES}")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
