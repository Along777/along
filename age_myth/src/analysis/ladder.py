"""Multi-age expected-age-at-death ladders (HMD summary + HLD)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.panels import ANALYSIS_DIR, save_panels
from src.analysis.populations import dedupe_hmd, hld_quality_slice, label_region
from src.paths import OUTPUTS, PROCESSED, ensure_dirs

BULLET = OUTPUTS / "bulletproof"
LADDER_AGES_HLD = [0, 1, 5, 15, 20, 30, 50, 65]


def load_hmd_wide(sex: str = "both") -> pd.DataFrame:
    path = ANALYSIS_DIR / f"hmd_summary_wide_{sex}.parquet"
    if not path.exists():
        save_panels()
    return dedupe_hmd(pd.read_parquet(path))


def load_hld_long() -> pd.DataFrame:
    path = PROCESSED / "life_expectancy_modeling_hld_median.parquet"
    return pd.read_parquet(path)


def hmd_ladder_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Long rows: region, year, sex, age_x, e_x, expected_age, imr, s_to_65."""
    rows = []
    for age, ecol in [(0, "e0"), (65, "e65"), (80, "e80")]:
        if ecol not in panel.columns:
            continue
        part = panel.dropna(subset=[ecol]).copy()
        part["age_x"] = age
        part["e_x"] = part[ecol]
        part["expected_age"] = age + part["e_x"]
        cols = [
            "region_id",
            "year",
            "sex",
            "age_x",
            "e_x",
            "expected_age",
            "e0",
            "imr",
            "s_to_65",
            "label",
        ]
        if "label" not in part.columns:
            part["label"] = part["region_id"].map(label_region)
        rows.append(part[[c for c in cols if c in part.columns]])
    return pd.concat(rows, ignore_index=True)


def hld_wide(mode: str = "gold") -> pd.DataFrame:
    """Wide e(x) by region-year-sex; gold = n_tables==1 only."""
    h = hld_quality_slice(load_hld_long(), mode=mode)
    # keep sex-specific; also build sex-averaged later
    w = h.pivot_table(
        index=["region_id", "year", "sex"],
        columns="age",
        values="life_expectancy",
        aggfunc="mean",
    )
    w = w.reset_index()
    w["quality"] = "hld_n_tables_1" if mode == "gold" else "hld_median"
    return w


def hld_ladder_long(mode: str = "gold") -> pd.DataFrame:
    w = hld_wide(mode=mode)
    ages = [a for a in LADDER_AGES_HLD if a in w.columns]
    rows = []
    for age in ages:
        part = w.dropna(subset=[age, 0] if age != 0 and 0 in w.columns else [age]).copy()
        part["age_x"] = age
        part["e_x"] = part[age]
        part["expected_age"] = age + part["e_x"]
        part["e0"] = part[0] if 0 in part.columns else np.nan
        part["label"] = part["region_id"].map(label_region)
        rows.append(
            part[
                [
                    "region_id",
                    "year",
                    "sex",
                    "age_x",
                    "e_x",
                    "expected_age",
                    "e0",
                    "quality",
                    "label",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_ladder_when_low_e0(
    ladder: pd.DataFrame, e0_threshold: float = 40.0, e0_col: str = "e0"
) -> pd.DataFrame:
    """Median expected age by age_x among rows with e0 < threshold."""
    d = ladder.dropna(subset=[e0_col, "expected_age", "age_x"]).copy()
    d = d[d[e0_col] < e0_threshold]
    rows = []
    for age, g in d.groupby("age_x"):
        exp = g["expected_age"].to_numpy(dtype=float)
        rows.append(
            {
                "age_x": int(age),
                "n": int(len(g)),
                "n_countries": int(g["region_id"].nunique()),
                "median_e0": float(g[e0_col].median()),
                "median_e_x": float(g["e_x"].median()),
                "median_expected_age": float(np.median(exp)),
                "p10_expected_age": float(np.quantile(exp, 0.1)),
                "p90_expected_age": float(np.quantile(exp, 0.9)),
                "share_ge_45": float(np.mean(exp >= 45)),
                "share_ge_50": float(np.mean(exp >= 50)),
                "share_ge_60": float(np.mean(exp >= 60)),
                "share_ge_70": float(np.mean(exp >= 70)),
                "e0_threshold": e0_threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("age_x")


def build_all_ladders() -> dict[str, Path]:
    ensure_dirs()
    BULLET.mkdir(parents=True, exist_ok=True)
    paths = {}

    hmd = load_hmd_wide("both")
    hmd_l = hmd_ladder_rows(hmd)
    p = BULLET / "ladder_hmd_long.csv"
    hmd_l.to_csv(p, index=False)
    paths["hmd_long"] = p

    for thr in (40.0, 35.0):
        agg = aggregate_ladder_when_low_e0(hmd_l, e0_threshold=thr)
        ap = BULLET / f"ladder_hmd_agg_e0lt{int(thr)}.csv"
        agg.to_csv(ap, index=False)
        paths[f"hmd_agg_{int(thr)}"] = ap

    for mode in ("gold", "median"):
        try:
            hl = hld_ladder_long(mode=mode)
            if hl.empty:
                continue
            # sex-averaged view for main ladder
            avg = (
                hl.groupby(["region_id", "year", "age_x", "quality"], as_index=False)
                .agg(
                    e_x=("e_x", "mean"),
                    expected_age=("expected_age", "mean"),
                    e0=("e0", "mean"),
                    label=("label", "first"),
                )
            )
            avg["sex"] = "sex_avg"
            lp = BULLET / f"ladder_hld_{mode}_long.csv"
            avg.to_csv(lp, index=False)
            paths[f"hld_{mode}_long"] = lp
            for thr in (40.0, 35.0):
                agg = aggregate_ladder_when_low_e0(avg, e0_threshold=thr)
                ap = BULLET / f"ladder_hld_{mode}_agg_e0lt{int(thr)}.csv"
                agg.to_csv(ap, index=False)
                paths[f"hld_{mode}_agg_{int(thr)}"] = ap
        except Exception as e:  # noqa: BLE001
            print(f"HLD ladder {mode} failed: {e}")

    print(f"Ladders written under {BULLET}")
    return paths


def main() -> None:
    build_all_ladders()


if __name__ == "__main__":
    main()
