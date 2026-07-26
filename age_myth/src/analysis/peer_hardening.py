"""Peer-review remediations: de-dupe, cluster bootstrap, honest model stack, HLD mid-adult."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.panels import ANALYSIS_DIR, PRIMARY_ALLOWLIST, save_panels
from src.paths import OUTPUTS, PROCESSED, ensure_dirs

TABLES = OUTPUTS / "tables"
RNG = np.random.default_rng(123)


def load_panel() -> pd.DataFrame:
    path = ANALYSIS_DIR / "hmd_summary_wide_both.parquet"
    if not path.exists():
        save_panels()
    return pd.read_parquet(path)


def is_duplicate_subseries(region_id: str) -> bool:
    r = str(region_id).upper()
    if "CIVILIAN" in r:
        return True
    if "EAST" in r or "WEST" in r:
        return True
    if "MAORI" in r and "TOTAL" not in r:
        return True
    if "NON-MAORI" in r:
        return True
    return False


def dedupe_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop known subseries; keep national totals / short codes."""
    return panel[~panel["region_id"].map(is_duplicate_subseries)].copy()


def cluster_bootstrap_median(
    df: pd.DataFrame, col: str, cluster: str = "region_id", n_boot: int = 2000
) -> dict:
    """Resample clusters (countries), keep all years within cluster."""
    clusters = df[cluster].unique().tolist()
    x = df[col].to_numpy(dtype=float)
    point = float(np.median(x[np.isfinite(x)]))
    boots = []
    for _ in range(n_boot):
        drawn = RNG.choice(clusters, size=len(clusters), replace=True)
        parts = [df.loc[df[cluster] == c, col].to_numpy() for c in drawn]
        sample = np.concatenate(parts)
        sample = sample[np.isfinite(sample)]
        if len(sample):
            boots.append(np.median(sample))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n_rows": int(len(df)),
        "n_clusters": int(len(clusters)),
        "median": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "method": "cluster_bootstrap_by_country",
    }


def cluster_bootstrap_mean_of_cluster_stats(
    series_by_cluster: pd.Series, n_boot: int = 2000
) -> dict:
    """Bootstrap mean of country-level statistics (e.g. delta e65)."""
    vals = series_by_cluster.dropna().to_numpy(dtype=float)
    clusters = series_by_cluster.dropna().index.to_numpy()
    point = float(np.mean(vals))
    boots = []
    for _ in range(n_boot):
        idx = RNG.choice(len(vals), size=len(vals), replace=True)
        boots.append(np.mean(vals[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n_clusters": int(len(vals)),
        "mean": point,
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "method": "bootstrap_countries",
        "cluster_ids": clusters.tolist(),
    }


def myth_a_hardened(panel: pd.DataFrame) -> dict:
    d = dedupe_panel(panel)
    low = d[d["e0"] < 40].copy()
    out = {
        "definition": "both-sex HMD summary, e0<40, subseries de-duplicated",
        "n_country_years": int(len(low)),
        "n_countries": int(low["region_id"].nunique()),
        "countries": sorted(low["region_id"].unique().tolist()),
        "e0_median": float(low["e0"].median()),
        "imr_median": float(low["imr"].median()),
        "s_to_65_median": float(low["s_to_65"].median()),
        "exp_death_65": cluster_bootstrap_median(low, "exp_death_65"),
        "adult_gap_65": cluster_bootstrap_median(low, "adult_gap_65"),
        "share_exp_death_65_ge_70": float((low["exp_death_65"] >= 70).mean()),
        "share_exp_death_65_ge_60": float((low["exp_death_65"] >= 60).mean()),
        "exp_death_65_p10_p90": [
            float(low["exp_death_65"].quantile(0.1)),
            float(low["exp_death_65"].quantile(0.9)),
        ],
        "n_e0_in_30_35": int(((low["e0"] >= 30) & (low["e0"] <= 35)).sum()),
        "reject_myth_A": bool(
            low["exp_death_65"].median() > 70 and (low["exp_death_65"] >= 70).mean() > 0.95
        ),
        "caveat": "exp_death_65 is CONDITIONAL on surviving to age 65; median S(0->65) is low",
    }
    # primary allowlist sensitivity
    prim = low[low["region_id"].isin(PRIMARY_ALLOWLIST)]
    out["primary_allowlist"] = {
        "n_country_years": int(len(prim)),
        "n_countries": int(prim["region_id"].nunique()),
        "exp_death_65_median": float(prim["exp_death_65"].median()) if len(prim) else None,
        "share_ge_70": float((prim["exp_death_65"] >= 70).mean()) if len(prim) else None,
    }
    return out, low


def myth_b_hardened(panel: pd.DataFrame) -> dict:
    d = dedupe_panel(panel)
    pre = d[d["year"] < 1900].groupby("region_id")["e65"].mean()
    post = d[d["year"] >= 2000].groupby("region_id")["e65"].mean()
    both = pre.to_frame("pre").join(post.to_frame("post"), how="inner").dropna()
    both["delta"] = both["post"] - both["pre"]
    stats = cluster_bootstrap_mean_of_cluster_stats(both["delta"])
    pre80 = d[d["year"] < 1900].groupby("region_id")["e80"].mean()
    post80 = d[d["year"] >= 2000].groupby("region_id")["e80"].mean()
    b80 = pre80.to_frame("pre").join(post80.to_frame("post"), how="inner").dropna()
    b80["delta"] = b80["post"] - b80["pre"]
    out = {
        "definition": "de-duplicated countries with pre-1900 and post-2000 e65",
        "delta_e65": stats,
        "country_table": both.reset_index().to_dict(orient="records"),
        "reject_myth_B": bool(stats["mean"] is not None and stats["mean"] > 3),
    }
    if len(b80):
        out["delta_e80"] = cluster_bootstrap_mean_of_cluster_stats(b80["delta"])
    return out, both


def model_comparison(panel: pd.DataFrame) -> dict:
    """Honest association stack — no headline overfitting narrative."""
    import statsmodels.formula.api as smf

    d = dedupe_panel(panel).dropna(subset=["e0", "imr", "year", "region_id"]).copy()
    d["year"] = d["year"].astype(float)
    d["e0_dm"] = d["e0"] - d.groupby("region_id")["e0"].transform("mean")
    d["imr_dm"] = d["imr"] - d.groupby("region_id")["imr"].transform("mean")
    d["year_dm"] = d["year"] - d.groupby("region_id")["year"].transform("mean")
    # first differences within country
    d = d.sort_values(["region_id", "year"])
    d["de0"] = d.groupby("region_id")["e0"].diff()
    d["dimr"] = d.groupby("region_id")["imr"].diff()

    m0 = smf.ols("e0 ~ imr", data=d).fit()
    m1 = smf.ols("e0_dm ~ imr_dm", data=d).fit()
    m2 = smf.ols("e0_dm ~ imr_dm + year_dm", data=d).fit()
    d3 = d.dropna(subset=["de0", "dimr"])
    m3 = smf.ols("de0 ~ dimr", data=d3).fit()
    m_fe = smf.ols("e0 ~ imr + year + C(region_id)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["region_id"]}
    )

    # LOCO beta stability for within imr
    betas = []
    for rid in d["region_id"].unique():
        dd = d[d["region_id"] != rid]
        try:
            b = smf.ols("e0_dm ~ imr_dm + year_dm", data=dd).fit().params["imr_dm"]
            betas.append(float(b))
        except Exception:
            pass

    return {
        "note": (
            "High R2 with IMR is expected: e0 and IMR are functions of the same period "
            "mortality schedule. Not an ML overfit. Not causal."
        ),
        "corr_imr_year": float(d["imr"].corr(d["year"])),
        "models": {
            "M0_pooled_imr_only": {
                "r2": float(m0.rsquared),
                "beta_imr": float(m0.params["imr"]),
                "se": float(m0.bse["imr"]),
            },
            "M1_within_imr_only": {
                "r2": float(m1.rsquared),
                "beta_imr": float(m1.params["imr_dm"]),
                "se": float(m1.bse["imr_dm"]),
            },
            "M2_within_imr_plus_year": {
                "r2": float(m2.rsquared),
                "beta_imr": float(m2.params["imr_dm"]),
                "se": float(m2.bse["imr_dm"]),
                "beta_year": float(m2.params["year_dm"]),
            },
            "M3_first_difference": {
                "r2": float(m3.rsquared),
                "beta_dimr": float(m3.params["dimr"]),
                "se": float(m3.bse["dimr"]),
                "n": int(m3.nobs),
                "corr_de0_dimr": float(d3["de0"].corr(d3["dimr"])),
            },
            "M_fe_full_sensitivity": {
                "r2": float(m_fe.rsquared),
                "beta_imr": float(m_fe.params["imr"]),
                "se_cluster": float(m_fe.bse["imr"]),
                "do_not_headline_r2": True,
            },
        },
        "loco_within_imr_year_beta": {
            "mean": float(np.mean(betas)),
            "std": float(np.std(betas)),
            "min": float(np.min(betas)),
            "max": float(np.max(betas)),
        },
        "n_rows": int(len(d)),
        "n_countries": int(d["region_id"].nunique()),
    }


def hld_mid_adult() -> dict:
    """Myth A support at ages 15 and 30 using HLD median panel."""
    path = PROCESSED / "life_expectancy_modeling_hld_median.parquet"
    if not path.exists():
        return {"error": "HLD median file missing"}
    h = pd.read_parquet(path)
    # average sexes for national-ish view
    w = h.pivot_table(
        index=["region_id", "year"], columns="age", values="life_expectancy", aggfunc="mean"
    )
    need = [c for c in [0, 15, 30, 65] if c in w.columns]
    w = w.dropna(subset=[0]).copy()
    out = {"n_rows_e0": int(len(w))}
    if 15 in w.columns:
        ww = w.dropna(subset=[15])
        ww = ww.copy()
        ww["exp15"] = 15 + ww[15]
        low = ww[ww[0] < 40]
        out["age15"] = {
            "n_low_e0": int(len(low)),
            "n_countries": int(low.reset_index()["region_id"].nunique()) if len(low) else 0,
            "median_e0": float(low[0].median()) if len(low) else None,
            "median_e15_remaining": float(low[15].median()) if len(low) else None,
            "median_exp_age_if_15": float(low["exp15"].median()) if len(low) else None,
            "share_exp15_ge_50": float((low["exp15"] >= 50).mean()) if len(low) else None,
            "share_exp15_ge_45": float((low["exp15"] >= 45).mean()) if len(low) else None,
            "caveat": "HLD median-collapsed multi-table; male/female averaged; methods heterogeneous",
        }
        # save long for figures
        low.reset_index().to_csv(TABLES / "hld_low_e0_age15.csv", index=False)
    if 30 in w.columns:
        ww = w.dropna(subset=[30]).copy()
        ww["exp30"] = 30 + ww[30]
        low = ww[ww[0] < 40]
        out["age30"] = {
            "n_low_e0": int(len(low)),
            "median_exp_age_if_30": float(low["exp30"].median()) if len(low) else None,
            "share_exp30_ge_55": float((low["exp30"] >= 55).mean()) if len(low) else None,
        }
    if 65 in w.columns:
        ww = w.dropna(subset=[65]).copy()
        ww["exp65"] = 65 + ww[65]
        low = ww[ww[0] < 40]
        out["age65_hld"] = {
            "n_low_e0": int(len(low)),
            "median_exp_age_if_65": float(low["exp65"].median()) if len(low) else None,
            "share_ge_70": float((low["exp65"] >= 70).mean()) if len(low) else None,
        }
    # Sweden series for e0 and exp15
    if "SWE" in w.index.get_level_values(0) and 15 in w.columns:
        s = w.loc["SWE"].dropna(subset=[0, 15]).copy()
        s["exp15"] = 15 + s[15]
        s.reset_index().to_csv(TABLES / "hld_sweden_e0_e15.csv", index=False)
    return out


def within_country_corrs(panel: pd.DataFrame) -> pd.DataFrame:
    d = dedupe_panel(panel)
    rows = []
    for rid, g in d.groupby("region_id"):
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
    return pd.DataFrame(rows)


def run() -> dict:
    ensure_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    a, low = myth_a_hardened(panel)
    b, btab = myth_b_hardened(panel)
    models = model_comparison(panel)
    hld = hld_mid_adult()
    corrs = within_country_corrs(panel)
    corrs.to_csv(TABLES / "peer_within_country_corrs.csv", index=False)
    btab.reset_index().to_csv(TABLES / "peer_myth_b_deltas_deduped.csv", index=False)
    low.to_csv(TABLES / "peer_myth_a_low_e0_deduped.csv", index=False)

    results = {
        "peer_review_version": "1.0",
        "myth_a_hardened": a,
        "myth_b_hardened": b,
        "model_comparison": models,
        "hld_mid_adult": hld,
        "within_corr_summary": {
            "n_countries": int(len(corrs)),
            "median_pearson": float(corrs["corr_pearson"].median()) if len(corrs) else None,
            "mean_pearson": float(corrs["corr_pearson"].mean()) if len(corrs) else None,
        },
    }
    path = TABLES / "peer_hardened_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2)[:4000])
    print(f"Wrote {path}")
    return results


def main() -> None:
    run()


if __name__ == "__main__":
    main()
