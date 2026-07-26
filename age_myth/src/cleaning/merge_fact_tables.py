"""Optional: rebuild life_expectancy_long from interim parts + literature seed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS


def main() -> None:
    ensure_dirs()
    parts = []
    for name in [
        "owid_life_expectancy_long.csv",
        "hmd_life_expectancy_long.csv",
        "hmd_summary_life_expectancy_long.csv",
        "hld_inventory_parse.csv",
        "clio_life_expectancy_long.csv",
        "eurostat_life_expectancy_long.csv",
    ]:
        p = INTERIM / name
        if p.exists() and p.stat().st_size > 50:
            df = pd.read_csv(p)
            if len(df):
                parts.append(df)
                print(f"Include {p.name}: {len(df):,} rows")

    lit = PROCESSED / "literature_benchmarks.csv"
    if lit.exists():
        ldf = pd.read_csv(lit)
        # map literature into fact columns only
        cols = [c for c in FACT_COLUMNS if c in ldf.columns]
        if cols and len(ldf):
            # only rows with life_expectancy
            ldf = ldf.copy()
            if "life_expectancy" in ldf.columns:
                ldf = ldf[pd.to_numeric(ldf["life_expectancy"], errors="coerce").notna()]
            if len(ldf):
                parts.append(ldf[FACT_COLUMNS] if set(FACT_COLUMNS).issubset(ldf.columns) else ldf[cols])
                print(f"Include literature_benchmarks: {len(ldf):,} rows")

    if not parts:
        print("No interim parts found")
        raise SystemExit(1)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    for c in FACT_COLUMNS:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[FACT_COLUMNS]
    out = PROCESSED / "life_expectancy_long.csv"
    combined.to_csv(out, index=False)
    try:
        combined.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {out} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
