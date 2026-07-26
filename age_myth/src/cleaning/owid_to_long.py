"""Convert OWID life expectancy CSVs into life_expectancy_long rows."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS

# Map download file -> age, source_id, quality flag
FILE_SPECS = [
    {
        "file": "life-expectancy.csv",
        "age": 0,
        "source_id": "owid_le_longrun",
        "flag": "owid_stitched",
        "table_type": "owid_compiled",
    },
    {
        "file": "life-expectancy-at-age-15.csv",
        "age": 15,
        "source_id": "owid_le_age15",
        "flag": "owid_stitched",
        "table_type": "owid_hmd_unwpp",
    },
    {
        "file": "life-expectancy-hmd-unwpp.csv",
        "age": 0,
        "source_id": "owid_le_hmd_unwpp",
        "flag": "owid_stitched",
        "table_type": "owid_hmd_unwpp",
    },
]


def _slug_region(entity: str, code: str | float | None) -> str:
    if code is not None and not (isinstance(code, float) and pd.isna(code)):
        c = str(code).strip()
        if c and c.lower() not in {"nan", "none"}:
            return c
    return (
        str(entity)
        .strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace(".", "")
    )


def _value_column(df: pd.DataFrame) -> str:
    # OWID full CSV: Entity, Code, Year, <indicator name>
    skip = {"entity", "code", "year", "entities", "countries"}
    for c in df.columns:
        if c.lower() in skip:
            continue
        return c
    raise ValueError(f"No value column found in columns={list(df.columns)}")


def load_one(path: Path, age: int, source_id: str, flag: str, table_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns
    colmap = {c: c.strip() for c in df.columns}
    df = df.rename(columns=colmap)
    lower = {c.lower(): c for c in df.columns}
    entity_col = lower.get("entity") or lower.get("country") or list(df.columns)[0]
    year_col = lower.get("year")
    code_col = lower.get("code")
    if year_col is None:
        raise ValueError(f"No Year column in {path}")
    val_col = _value_column(df)

    out = pd.DataFrame()
    out["country_region"] = df[entity_col].astype(str)
    out["region_id"] = [
        _slug_region(e, df[code_col].iloc[i] if code_col else None)
        for i, e in enumerate(df[entity_col])
    ]
    out["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    out["period_start"] = pd.NA
    out["period_end"] = pd.NA
    out["sex"] = "both"
    out["age"] = age
    out["life_expectancy"] = pd.to_numeric(df[val_col], errors="coerce")
    out["survival_probability"] = pd.NA
    out["infant_mortality_rate"] = pd.NA
    out["measure_type"] = "period"
    out["population_type"] = "national"
    # aggregates
    out.loc[
        out["country_region"].str.contains("World|income|region|continent", case=False, na=False),
        "population_type",
    ] = "national"  # keep; region_type handled elsewhere
    out["table_type"] = table_type
    out["data_quality_flag"] = flag
    out["source_id"] = source_id
    out["notes"] = f"OWID series from {path.name}"
    out["retrieved_at"] = date.today().isoformat()
    out = out.dropna(subset=["life_expectancy", "year"])
    # drop projections far future if desired — keep all for research transparency
    return out[FACT_COLUMNS]


def main() -> None:
    ensure_dirs()
    raw = RAW / "owid"
    frames: list[pd.DataFrame] = []
    for spec in FILE_SPECS:
        path = raw / spec["file"]
        if not path.exists():
            print(f"Skip missing {path}")
            continue
        part = load_one(path, spec["age"], spec["source_id"], spec["flag"], spec["table_type"])
        print(f"Loaded {len(part):,} rows from {path.name} (age={spec['age']})")
        frames.append(part)

    if not frames:
        print("No OWID files found. Run: python -m src.acquisition.download_owid")
        raise SystemExit(1)

    combined = pd.concat(frames, ignore_index=True)
    interim_path = INTERIM / "owid_life_expectancy_long.csv"
    combined.to_csv(interim_path, index=False)
    print(f"Wrote {interim_path} ({len(combined):,} rows)")

    # merge into processed fact table (replace OWID rows, keep others)
    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists() and fact_path.stat().st_size > 50:
        existing = pd.read_csv(fact_path)
        if len(existing):
            existing = existing[~existing["source_id"].astype(str).str.startswith("owid_")]
            combined = pd.concat([existing, combined], ignore_index=True)
    combined.to_csv(fact_path, index=False)
    try:
        combined.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {fact_path} ({len(combined):,} total rows)")


if __name__ == "__main__":
    main()
