"""Recompute all primary claims with cluster CIs and hard gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.claim_registry import CLAIMS, SCOPE
from src.analysis.ladder import (
    BULLET,
    aggregate_ladder_when_low_e0,
    build_all_ladders,
    hld_ladder_long,
    load_hmd_wide,
)
from src.analysis.populations import PRIMARY_ALLOWLIST, dedupe_hmd, filter_primary
from src.paths import ensure_dirs

RNG = np.random.default_rng(7)


def cluster_boot_median(df: pd.DataFrame, col: str, cluster: str = "region_id", n: int = 2000) -> dict:
    clusters = df[cluster].dropna().unique()
    x = df[col].to_numpy(dtype=float)
    point = float(np.median(x[np.isfinite(x)]))
    boots = []
    for _ in range(n):
        drawn = RNG.choice(clusters, size=len(clusters), replace=True)
        sample = np.concatenate([df.loc[df[cluster] == c, col].to_numpy() for c in drawn])
        sample = sample[np.isfinite(sample)]
        if len(sample):
            boots.append(np.median(sample))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "median": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_rows": int(len(df)),
        "n_clusters": int(len(clusters)),
        "method": "cluster_bootstrap_country",
    }


def cluster_boot_mean(values: np.ndarray, n: int = 2000) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    point = float(np.mean(values))
    boots = [float(np.mean(RNG.choice(values, size=len(values), replace=True))) for _ in range(n)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean": point,
        "median": float(np.median(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
        "method": "bootstrap_countries",
    }


def claim_a(panel: pd.DataFrame) -> dict:
    low = panel[panel["e0"] < 40].copy()
    exp = cluster_boot_median(low, "exp_death_65")
    gap = cluster_boot_median(low, "adult_gap_65")
    return {
        "n_country_years": int(len(low)),
        "n_countries": int(low["region_id"].nunique()),
        "countries": sorted(low["region_id"].unique().tolist()),
        "median_e0": float(low["e0"].median()),
        "median_imr": float(low["imr"].median()),
        "median_s_to_65": float(low["s_to_65"].median()),
        "exp_death_65": exp,
        "adult_gap_65": gap,
        "share_exp65_ge_70": float((low["exp_death_65"] >= 70).mean()),
        "share_exp65_ge_60": float((low["exp_death_65"] >= 60).mean()),
        "p10_p90_exp65": [
            float(low["exp_death_65"].quantile(0.1)),
            float(low["exp_death_65"].quantile(0.9)),
        ],
        "n_e0_30_35": int(((low["e0"] >= 30) & (low["e0"] <= 35)).sum()),
    }


def claim_a2() -> dict:
    """Mid-adult from HLD gold, fallback median."""
    result = {"quality_used": None, "ages": {}}
    for mode in ("gold", "median"):
        try:
            long = hld_ladder_long(mode=mode)
        except Exception as e:  # noqa: BLE001
            result[f"error_{mode}"] = str(e)
            continue
        if long.empty:
            continue
        avg = (
            long.groupby(["region_id", "year", "age_x"], as_index=False)
            .agg(e_x=("e_x", "mean"), expected_age=("expected_age", "mean"), e0=("e0", "mean"))
        )
        low = avg[(avg["e0"] < 40)]
        if low.empty:
            continue
        # need enough age 15
        a15 = low[low["age_x"] == 15]
        a30 = low[low["age_x"] == 30]
        if len(a15) < 20 and mode == "gold":
            result["gold_too_thin_n15"] = int(len(a15))
            continue
        result["quality_used"] = mode
        for age, sub in [(15, a15), (30, a30), (65, low[low["age_x"] == 65])]:
            if sub.empty:
                continue
            exp = sub["expected_age"].to_numpy(dtype=float)
            # cluster boot by region
            tmp = sub.copy()
            result["ages"][str(age)] = {
                **cluster_boot_median(tmp, "expected_age"),
                "median_e_x": float(sub["e_x"].median()),
                "share_ge_45": float(np.mean(exp >= 45)),
                "share_ge_50": float(np.mean(exp >= 50)),
                "share_ge_55": float(np.mean(exp >= 55)),
                "share_ge_60": float(np.mean(exp >= 60)),
                "share_ge_70": float(np.mean(exp >= 70)),
                "n_countries": int(sub["region_id"].nunique()),
            }
        # full ladder agg
        agg = aggregate_ladder_when_low_e0(avg, 40.0)
        result["ladder_agg"] = agg.to_dict(orient="records")
        break
    return result


def claim_b(panel: pd.DataFrame) -> dict:
    pre = panel[panel["year"] < 1900].groupby("region_id")["e65"].mean()
    post = panel[panel["year"] >= 2000].groupby("region_id")["e65"].mean()
    both = pre.to_frame("pre").join(post.to_frame("post"), how="inner").dropna()
    both["delta"] = both["post"] - both["pre"]
    stats = cluster_boot_mean(both["delta"].to_numpy())
    pre80 = panel[panel["year"] < 1900].groupby("region_id")["e80"].mean()
    post80 = panel[panel["year"] >= 2000].groupby("region_id")["e80"].mean()
    b80 = pre80.to_frame("pre").join(post80.to_frame("post"), how="inner").dropna()
    out = {
        "delta_e65": stats,
        "countries": both.reset_index().to_dict(orient="records"),
    }
    if len(b80):
        b80["delta"] = b80["post"] - b80["pre"]
        out["delta_e80"] = cluster_boot_mean(b80["delta"].to_numpy())
    return out


def claim_c(panel: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    corrs = []
    for rid, g in panel.groupby("region_id"):
        g = g.dropna(subset=["e0", "imr"])
        if len(g) < 10:
            continue
        corrs.append(float(g["e0"].corr(g["imr"])))
    d = panel.sort_values(["region_id", "year"]).copy()
    d["de0"] = d.groupby("region_id")["e0"].diff()
    d["dimr"] = d.groupby("region_id")["imr"].diff()
    d3 = d.dropna(subset=["de0", "dimr"])
    m3 = smf.ols("de0 ~ dimr", data=d3).fit()
    # within + year
    d["e0_dm"] = d["e0"] - d.groupby("region_id")["e0"].transform("mean")
    d["imr_dm"] = d["imr"] - d.groupby("region_id")["imr"].transform("mean")
    d["year_dm"] = d["year"].astype(float) - d.groupby("region_id")["year"].transform(
        lambda s: s.astype(float).mean()
    )
    m2 = smf.ols("e0_dm ~ imr_dm + year_dm", data=d.dropna(subset=["e0_dm", "imr_dm", "year_dm"])).fit()
    m0 = smf.ols("e0 ~ imr", data=panel.dropna(subset=["e0", "imr"])).fit()
    m1 = smf.ols("e0_dm ~ imr_dm", data=d.dropna(subset=["e0_dm", "imr_dm"])).fit()
    # out-of-time sign stability
    early = d[d["year"] < 1950].dropna(subset=["e0_dm", "imr_dm", "year_dm"])
    late = d[d["year"] >= 1950].dropna(subset=["e0_dm", "imr_dm", "year_dm"])
    oot = {}
    if len(early) > 100 and len(late) > 100:
        fit = smf.ols("e0_dm ~ imr_dm + year_dm", data=early).fit()
        oot = {
            "beta_imr_pre1950": float(fit.params["imr_dm"]),
            "beta_imr_post1950_refit": float(
                smf.ols("e0_dm ~ imr_dm + year_dm", data=late).fit().params["imr_dm"]
            ),
            "same_sign": bool(fit.params["imr_dm"] < 0 and smf.ols("e0_dm ~ imr_dm + year_dm", data=late).fit().params["imr_dm"] < 0),
        }
    return {
        "median_within_corr_e0_imr": float(np.median(corrs)) if corrs else None,
        "mean_within_corr_e0_imr": float(np.mean(corrs)) if corrs else None,
        "n_countries_corr": len(corrs),
        "models": {
            "M0_pooled": {"r2": float(m0.rsquared), "beta_imr": float(m0.params["imr"])},
            "M1_within": {"r2": float(m1.rsquared), "beta_imr": float(m1.params["imr_dm"])},
            "M2_within_year": {
                "r2": float(m2.rsquared),
                "beta_imr": float(m2.params["imr_dm"]),
                "beta_year": float(m2.params["year_dm"]),
            },
            "M3_first_diff": {
                "r2": float(m3.rsquared),
                "beta_dimr": float(m3.params["dimr"]),
                "corr_de0_dimr": float(d3["de0"].corr(d3["dimr"])),
            },
        },
        "out_of_time": oot,
        "note": "Associational only; high R2 expected from shared mortality schedule",
    }


def sweden_snapshot(panel: pd.DataFrame) -> dict:
    s = panel[(panel["region_id"] == "SWE") & (panel["year"] == 1800)]
    if s.empty:
        return {"error": "missing SWE 1800"}
    r = s.iloc[0]
    return {
        "year": 1800,
        "e0": float(r["e0"]),
        "imr": float(r["imr"]),
        "e65": float(r["e65"]),
        "exp_death_65": float(r["exp_death_65"]),
        "s_to_65": float(r["s_to_65"]),
    }


def sensitivity_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = {
        "deduped_all": panel,
        "primary_allowlist": filter_primary(panel),
        "female": load_sex("female"),
        "male": load_sex("male"),
    }
    for name, p in variants.items():
        if p is None or p.empty:
            continue
        for thr in (40, 35):
            low = p[p["e0"] < thr]
            if low.empty:
                continue
            rows.append(
                {
                    "variant": name,
                    "e0_threshold": thr,
                    "n": len(low),
                    "n_countries": low["region_id"].nunique(),
                    "median_exp65": float(low["exp_death_65"].median()),
                    "share_ge_70": float((low["exp_death_65"] >= 70).mean()),
                    "median_s65": float(low["s_to_65"].median()),
                    "median_imr": float(low["imr"].median()),
                }
            )
    # pre-1850
    pre = panel[panel["year"] < 1850]
    low = pre[pre["e0"] < 40]
    if len(low):
        rows.append(
            {
                "variant": "pre_1850_only",
                "e0_threshold": 40,
                "n": len(low),
                "n_countries": low["region_id"].nunique(),
                "median_exp65": float(low["exp_death_65"].median()),
                "share_ge_70": float((low["exp_death_65"] >= 70).mean()),
                "median_s65": float(low["s_to_65"].median()),
                "median_imr": float(low["imr"].median()),
            }
        )
    return pd.DataFrame(rows)


def load_sex(sex: str) -> pd.DataFrame:
    from src.analysis.panels import ANALYSIS_DIR

    path = ANALYSIS_DIR / f"hmd_summary_wide_{sex}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return dedupe_hmd(pd.read_parquet(path))


def evaluate_gates(claims: dict) -> dict:
    gates = {}
    a = claims["claim_A"]
    g = CLAIMS["A_birth_e0_not_adult_death_age"]["gates"]
    gates["A"] = {
        "median_exp65_ge_70": a["exp_death_65"]["median"] >= g["median_exp65_ge"],
        "share_ge_70": a["share_exp65_ge_70"] >= g["share_exp65_ge_70_ge"],
        "s_to_65_low": a["median_s_to_65"] < g["median_s_to_65_lt"],
        "imr_high": a["median_imr"] > g["median_imr_gt"],
        "n_ok": a["n_country_years"] >= g["min_n_country_years"],
    }
    gates["A"]["pass"] = all(gates["A"].values())

    a2 = claims["claim_A2"]
    g2 = CLAIMS["A2_mid_adult"]["gates"]
    ages = a2.get("ages", {})
    a15 = ages.get("15", {})
    a30 = ages.get("30", {})
    gates["A2"] = {
        "has_age15": bool(a15),
        "median_exp15": (a15.get("median") or 0) >= g2["median_exp15_ge"] if a15 else False,
        "share_exp15_ge_45": (a15.get("share_ge_45") or 0) >= g2["share_exp15_ge_45_ge"] if a15 else False,
        "median_exp30": (a30.get("median") or 0) >= g2["median_exp30_ge"] if a30 else False,
        "n_ok": (a15.get("n_rows") or 0) >= g2["min_n_low_e0"] if a15 else False,
    }
    gates["A2"]["pass"] = all(gates["A2"].values())

    b = claims["claim_B"]["delta_e65"]
    gB = CLAIMS["B_adults_improved"]["gates"]
    gates["B"] = {
        "mean_gt": b["mean"] > gB["mean_delta_e65_gt"],
        "ci_low_gt_0": b["ci_low"] > gB["ci_low_gt"],
        "n_countries": b["n"] >= gB["min_countries"],
    }
    gates["B"]["pass"] = all(gates["B"].values())

    c = claims["claim_C"]
    gC = CLAIMS["C_infant_mechanism"]["gates"]
    gates["C"] = {
        "corr": (c.get("median_within_corr_e0_imr") or 0) < gC["median_within_corr_lt"],
        "fd_sign": c["models"]["M3_first_diff"]["beta_dimr"] < gC["first_diff_beta_imr_lt"],
    }
    gates["C"]["pass"] = all(gates["C"].values())

    snap = claims["sweden_1800"]
    gS = CLAIMS["S_sweden_1800_snapshot"]["gates"]
    if "error" not in snap:
        gates["S"] = {
            "e0": abs(snap["e0"] - gS["e0_target"]) <= gS["e0_tol"],
            "exp65": abs(snap["exp_death_65"] - gS["exp65_target"]) <= gS["exp65_tol"],
            "imr": abs(snap["imr"] - gS["imr_target"]) <= gS["imr_tol"],
        }
        gates["S"]["pass"] = all(gates["S"].values())
    else:
        gates["S"] = {"pass": False, "error": snap["error"]}

    gates["all_pass"] = all(v.get("pass") for v in gates.values())
    return gates


def run() -> dict:
    ensure_dirs()
    BULLET.mkdir(parents=True, exist_ok=True)
    build_all_ladders()

    panel = load_hmd_wide("both")
    claims = {
        "scope": SCOPE,
        "claim_A": claim_a(panel),
        "claim_A_primary": claim_a(filter_primary(panel)),
        "claim_A2": claim_a2(),
        "claim_B": claim_b(panel),
        "claim_C": claim_c(panel),
        "sweden_1800": sweden_snapshot(panel),
    }
    sens = sensitivity_matrix(panel)
    sens.to_csv(BULLET / "sensitivity_matrix.csv", index=False)
    gates = evaluate_gates(claims)
    claims["gates"] = gates

    (BULLET / "claims.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")
    (BULLET / "claim_gate_results.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    print(json.dumps(gates, indent=2))
    print(f"all_pass={gates['all_pass']}")
    if not gates["all_pass"]:
        raise SystemExit(2)
    return claims


def main() -> None:
    run()


if __name__ == "__main__":
    main()
