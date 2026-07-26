"""Build analysis-ready wide panels from the modeling table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paths import PROCESSED, ensure_dirs

ANALYSIS_DIR = PROCESSED / "analysis"

# Preferred national / total-population series for multi-country charts
PRIMARY_ALLOWLIST = [
    "SWE",
    "FRANCE:_TOTAL_POPULATION",
    "UK:_ENGLAND_&_WALES_TOTAL_POPULATION",
    "DNK",
    "NOR",
    "NLD",
    "BEL",
    "ITA",
    "CHE",
    "FIN",
    "ISL",
    "USA",
    "AUS",
    "CAN",
    "JPN",
]

# Pretty labels for plots
LABELS = {
    "SWE": "Sweden",
    "FRANCE:_TOTAL_POPULATION": "France",
    "UK:_ENGLAND_&_WALES_TOTAL_POPULATION": "England & Wales",
    "DNK": "Denmark",
    "NOR": "Norway",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "ITA": "Italy",
    "CHE": "Switzerland",
    "FIN": "Finland",
    "ISL": "Iceland",
    "USA": "United States",
    "AUS": "Australia",
    "CAN": "Canada",
    "JPN": "Japan",
}


def load_modeling() -> pd.DataFrame:
    pq = PROCESSED / "life_expectancy_modeling.parquet"
    csv = PROCESSED / "life_expectancy_modeling.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(csv, low_memory=False)


def is_total_population(region_id: str) -> bool:
    r = str(region_id)
    if r in {"SWE", "USA", "NLD", "DNK", "NOR", "BEL", "ITA", "CHE", "FIN", "ISL", "AUS", "CAN", "JPN", "ESP", "PRT", "AUT", "HUN", "POL", "CZE", "SVK", "SVN", "HRV", "BGR", "GRC", "IRL", "LUX", "EST", "LVA", "LTU", "RUS", "UKR", "BLR", "CHL", "ISR", "KOR", "TWN", "HKG"}:
        return True
    if "CIVILIAN" in r or "MAORI" in r or "EAST" in r or "WEST" in r:
        return False
    if "TOTAL" in r:
        return True
    # short codes only
    return len(r) <= 4 and r.isalpha()


def era_label(year: float | int) -> str:
    y = int(year)
    if y < 1850:
        return "pre_1850"
    if y < 1900:
        return "1850_1899"
    if y < 1950:
        return "1900_1949"
    if y < 2000:
        return "1950_1999"
    return "2000_plus"


def build_hmd_wide(sex: str = "both") -> pd.DataFrame:
    df = load_modeling()
    h = df[(df["source_id"] == "hmd_summary_public") & (df["sex"] == sex)].copy()
    e0 = h[h["age"] == 0][
        ["region_id", "country_region", "year", "life_expectancy", "infant_mortality_rate"]
    ].rename(columns={"life_expectancy": "e0", "infant_mortality_rate": "imr"})
    e65 = h[h["age"] == 65][
        ["region_id", "year", "life_expectancy", "survival_probability"]
    ].rename(columns={"life_expectancy": "e65", "survival_probability": "s_to_65"})
    e80 = h[h["age"] == 80][["region_id", "year", "life_expectancy"]].rename(
        columns={"life_expectancy": "e80"}
    )
    p = e0.merge(e65, on=["region_id", "year"], how="inner").merge(
        e80, on=["region_id", "year"], how="left"
    )
    p["year"] = pd.to_numeric(p["year"], errors="coerce").astype("Int64")
    p["exp_death_0"] = p["e0"]
    p["exp_death_65"] = 65 + p["e65"]
    p["exp_death_80"] = 80 + p["e80"]
    p["adult_gap_65"] = p["exp_death_65"] - p["e0"]
    p["era"] = p["year"].map(era_label)
    p["is_total_pop"] = p["region_id"].map(is_total_population)
    p["in_primary_allowlist"] = p["region_id"].isin(PRIMARY_ALLOWLIST)
    p["label"] = p["region_id"].map(lambda x: LABELS.get(x, str(x).replace("_", " ")[:40]))
    p["sex"] = sex
    p["low_e0"] = p["e0"] < 40
    return p.sort_values(["region_id", "year"]).reset_index(drop=True)


def build_eurostat_long() -> pd.DataFrame:
    df = load_modeling()
    eu = df[df["source_id"] == "eurostat_demo_mlexpec"].copy()
    eu["year"] = pd.to_numeric(eu["year"], errors="coerce").astype("Int64")
    return eu


def build_owid_e0() -> pd.DataFrame:
    df = load_modeling()
    o = df[
        (df["source_id"].isin(["owid_le_longrun", "owid_le_hmd_unwpp"])) & (df["age"] == 0)
    ].copy()
    o["year"] = pd.to_numeric(o["year"], errors="coerce").astype("Int64")
    return o


def save_panels() -> dict[str, Path]:
    ensure_dirs()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sex in ("both", "female", "male"):
        wide = build_hmd_wide(sex=sex)
        path = ANALYSIS_DIR / f"hmd_summary_wide_{sex}.parquet"
        wide.to_parquet(path, index=False)
        wide.to_csv(ANALYSIS_DIR / f"hmd_summary_wide_{sex}.csv", index=False)
        paths[f"hmd_{sex}"] = path
        print(f"Wrote {path} ({len(wide):,} rows, {wide.region_id.nunique()} regions)")

    # primary totals only convenience extract
    both = build_hmd_wide("both")
    primary = both[both["in_primary_allowlist"] & both["is_total_pop"]].copy()
    # allowlist already total for short codes
    primary = both[both["in_primary_allowlist"]].copy()
    ppath = ANALYSIS_DIR / "hmd_summary_wide_both_primary.parquet"
    primary.to_parquet(ppath, index=False)
    primary.to_csv(ANALYSIS_DIR / "hmd_summary_wide_both_primary.csv", index=False)
    paths["hmd_primary"] = ppath
    print(f"Wrote {ppath} ({len(primary):,} rows, {primary.region_id.nunique()} regions)")

    eu = build_eurostat_long()
    epath = ANALYSIS_DIR / "eurostat_long.parquet"
    eu.to_parquet(epath, index=False)
    paths["eurostat"] = epath
    print(f"Wrote {epath} ({len(eu):,} rows)")
    return paths


def main() -> None:
    save_panels()


if __name__ == "__main__":
    main()
