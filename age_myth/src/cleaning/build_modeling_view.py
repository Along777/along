"""Build analysis-ready modeling tables that exclude known footguns.

Outputs:
  data/processed/life_expectancy_modeling.parquet|csv
      - no fixture / literature
      - year <= MAX_HISTORICAL_YEAR (default 2023)
      - adds is_projection (should be False after filter)
      - adds modeling_tier for source priority
      - EXCLUDES raw HLD by default (multi-table fan-out); optional --include-hld

  data/processed/life_expectancy_modeling_hld_median.parquet|csv (optional)
      - HLD only, median-collapsed to grain (region, year, sex, age)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.paths import OUTPUTS, PROCESSED, ensure_dirs

MAX_HISTORICAL_YEAR = 2023
EXCLUDE_SOURCES = {"hmd_fixture", "gurven_kaplan_2007"}

# Preferred order for modelers (documentation / ranking only)
TIER = {
    "hmd_summary_public": 1,
    "eurostat_demo_mlexpec": 1,
    "owid_le_hmd_unwpp": 2,
    "owid_le_longrun": 2,
    "owid_le_age15": 2,
    "clio_zijdeman_2015": 2,
    "hld": 3,
}


def load_fact() -> pd.DataFrame:
    pq = PROCESSED / "life_expectancy_long.parquet"
    csv = PROCESSED / "life_expectancy_long.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError("No life_expectancy_long table found")


def build_main(df: pd.DataFrame, *, include_hld: bool) -> pd.DataFrame:
    out = df.copy()
    out = out[~out["source_id"].astype(str).isin(EXCLUDE_SOURCES)]
    if not include_hld:
        out = out[out["source_id"].astype(str) != "hld"]

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["is_projection"] = out["year"].notna() & (out["year"] > MAX_HISTORICAL_YEAR)
    # historical modeling default: drop projections
    out = out[~out["is_projection"] | out["year"].isna()]
    # drop rows still beyond cutoff if year present
    out = out[out["year"].isna() | (out["year"] <= MAX_HISTORICAL_YEAR)]

    # Eurostat: keep 2-letter country codes only (drop EA20, EU27_2020, DE_TOT, ...)
    eu_mask = out["source_id"].astype(str) == "eurostat_demo_mlexpec"
    if eu_mask.any():
        iso2_ok = out["region_id"].astype(str).str.fullmatch(r"[A-Z]{2}").fillna(False)
        out = out.loc[~eu_mask | iso2_ok]

    out["modeling_tier"] = out["source_id"].map(TIER).fillna(9).astype(int)

    # cast year where possible
    out["year"] = out["year"].astype("Int64")
    return out.reset_index(drop=True)


def build_hld_median(df: pd.DataFrame) -> pd.DataFrame:
    hld = df[df["source_id"].astype(str) == "hld"].copy()
    if hld.empty:
        return hld
    hld["year"] = pd.to_numeric(hld["year"], errors="coerce")
    hld = hld[hld["year"].notna() & (hld["year"] <= MAX_HISTORICAL_YEAR)]
    grain = ["region_id", "year", "sex", "age"]
    agg = (
        hld.groupby(grain, as_index=False)
        .agg(
            country_region=("country_region", "first"),
            life_expectancy=("life_expectancy", "median"),
            survival_probability=("survival_probability", "median"),
            n_tables=("life_expectancy", "size"),
            period_start=("period_start", "first"),
            period_end=("period_end", "first"),
        )
    )
    agg["infant_mortality_rate"] = pd.NA
    agg["measure_type"] = "period"
    agg["population_type"] = "national"
    agg["table_type"] = "hld_median_collapse"
    agg["data_quality_flag"] = "hld_published_table"
    agg["source_id"] = "hld_median"
    agg["notes"] = "Median across HLD tables at grain; n_tables retained for uncertainty"
    agg["retrieved_at"] = pd.Timestamp.today().date().isoformat()
    agg["is_projection"] = False
    agg["modeling_tier"] = 3
    agg["year"] = agg["year"].astype("Int64")
    return agg


def write_summary(main: pd.DataFrame, hld_med: pd.DataFrame | None) -> None:
    lines = [
        "# Modeling view summary",
        "",
        f"- Built with `year <= {MAX_HISTORICAL_YEAR}`, excluded sources: `{sorted(EXCLUDE_SOURCES)}`.",
        f"- Main modeling rows: **{len(main):,}**",
        f"- Sources: {sorted(main['source_id'].dropna().unique())}",
        f"- Ages: {sorted(int(a) for a in main['age'].dropna().unique())}",
        "",
        "## Rows by source",
        "",
        "| source_id | rows |",
        "|-----------|------|",
    ]
    for sid, n in main.groupby("source_id").size().sort_values(ascending=False).items():
        lines.append(f"| {sid} | {n:,} |")
    if hld_med is not None and len(hld_med):
        lines += [
            "",
            f"- Optional HLD median-collapsed rows: **{len(hld_med):,}** "
            f"(max n_tables={int(hld_med['n_tables'].max())})",
        ]
    lines += [
        "",
        "## Safe defaults",
        "",
        "- Prefer `source_id == 'hmd_summary_public'` for long-run e0/e65/e80 + IMR.",
        "- Prefer `source_id == 'eurostat_demo_mlexpec'` for EU multi-age (ISO2 geos only).",
        "- Use `hld_median` only as sensitivity; inspect `n_tables`.",
        "",
    ]
    path = OUTPUTS / "modeling_view_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-hld",
        action="store_true",
        help="Keep raw multi-table HLD rows in the main modeling file (not recommended)",
    )
    parser.add_argument(
        "--hld-median",
        action="store_true",
        default=True,
        help="Also write HLD median-collapsed table (default on)",
    )
    parser.add_argument("--no-hld-median", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    raw = load_fact()
    main_df = build_main(raw, include_hld=args.include_hld)
    out_csv = PROCESSED / "life_expectancy_modeling.csv"
    out_pq = PROCESSED / "life_expectancy_modeling.parquet"
    main_df.to_csv(out_csv, index=False)
    try:
        main_df.to_parquet(out_pq, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {out_csv} ({len(main_df):,} rows)")

    hld_med = None
    if args.hld_median and not args.no_hld_median:
        hld_med = build_hld_median(raw)
        if len(hld_med):
            pcsv = PROCESSED / "life_expectancy_modeling_hld_median.csv"
            ppq = PROCESSED / "life_expectancy_modeling_hld_median.parquet"
            hld_med.to_csv(pcsv, index=False)
            try:
                hld_med.to_parquet(ppq, index=False)
            except Exception as e:  # noqa: BLE001
                print(f"Parquet skip: {e}")
            print(f"Wrote {pcsv} ({len(hld_med):,} rows)")

    write_summary(main_df, hld_med)


if __name__ == "__main__":
    main()
