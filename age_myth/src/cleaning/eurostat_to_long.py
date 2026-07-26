"""Parse Eurostat demo_mlexpec SDMX-CSV into life_expectancy_long."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS

TARGET_AGES = {0, 1, 5, 15, 20, 30, 50, 65, 80}


def _map_sex(v: str) -> str:
    v = str(v).upper()
    if v in {"M", "MALE"}:
        return "male"
    if v in {"F", "FEMALE"}:
        return "female"
    if v in {"T", "TOTAL"}:
        return "both"
    return "both"


def _parse_age(v) -> int | None:
    s = str(v).strip().upper()
    # Eurostat: Y0, Y1, Y15, Y_LT1, Y65, etc.
    if s in {"Y_LT1", "Y0", "0"}:
        return 0
    if s.startswith("Y") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    return None


def main() -> None:
    ensure_dirs()
    path = RAW / "eurostat" / "demo_mlexpec.csv"
    if not path.exists():
        print("Missing demo_mlexpec.csv. Run: python -m src.acquisition.download_eurostat")
        raise SystemExit(1)

    # SDMX-CSV can be large; read in chunks
    usecols_candidates = None
    peek = pd.read_csv(path, nrows=5)
    cols = list(peek.columns)
    print(f"Eurostat columns: {cols[:20]}...")

    # Common SDMX-CSV: DATAFLOW, LAST UPDATE, freq, unit, sex, age, geo, TIME_PERIOD, OBS_VALUE
    lower = {c.lower(): c for c in cols}

    def pick(*names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        for c in cols:
            cl = c.lower()
            for n in names:
                if n.lower() in cl:
                    return c
        return None

    sex_c = pick("sex")
    age_c = pick("age")
    geo_c = pick("geo", "geo_code")
    time_c = pick("time_period", "time", "year")
    val_c = pick("obs_value", "value")
    if not all([age_c, geo_c, time_c, val_c]):
        print(f"Could not map required columns from {cols}")
        raise SystemExit(1)

    today = date.today().isoformat()
    parts = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        chunk["_age"] = chunk[age_c].map(_parse_age)
        chunk = chunk[chunk["_age"].isin(TARGET_AGES)]
        chunk["_year"] = pd.to_numeric(chunk[time_c], errors="coerce")
        chunk["_val"] = pd.to_numeric(chunk[val_c], errors="coerce")
        chunk = chunk.dropna(subset=["_year", "_val", "_age"])
        chunk = chunk[(chunk["_val"] > 0) & (chunk["_val"] < 120)]
        chunk = chunk[(chunk["_year"] >= 1950) & (chunk["_year"] <= 2100)]
        if chunk.empty:
            continue
        sex = chunk[sex_c].map(_map_sex) if sex_c else "both"
        out = pd.DataFrame(
            {
                "region_id": chunk[geo_c].astype(str),
                "country_region": chunk[geo_c].astype(str),
                "year": chunk["_year"].astype(int),
                "period_start": pd.NA,
                "period_end": pd.NA,
                "sex": sex.values if not isinstance(sex, str) else sex,
                "age": chunk["_age"].astype(int),
                "life_expectancy": chunk["_val"].astype(float),
                "survival_probability": pd.NA,
                "infant_mortality_rate": pd.NA,
                "measure_type": "period",
                "population_type": "national",
                "table_type": "eurostat_demo_mlexpec",
                "data_quality_flag": "official_stats",
                "source_id": "eurostat_demo_mlexpec",
                "notes": "Eurostat demo_mlexpec life expectancy by age and sex",
                "retrieved_at": today,
            }
        )
        parts.append(out)

    if not parts:
        print("No Eurostat rows parsed")
        raise SystemExit(1)

    fact = pd.concat(parts, ignore_index=True)
    # drop EU aggregates? keep them with region_type later; filter common aggregates optional
    interim = INTERIM / "eurostat_life_expectancy_long.csv"
    fact.to_csv(interim, index=False)
    print(f"Wrote {interim} ({len(fact):,} rows); ages={sorted(fact['age'].unique())}")

    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists() and fact_path.stat().st_size > 50:
        existing = pd.read_csv(fact_path, low_memory=False)
        if len(existing):
            existing = existing[existing["source_id"].astype(str) != "eurostat_demo_mlexpec"]
            fact = pd.concat([existing, fact], ignore_index=True)
    fact.to_csv(fact_path, index=False)
    try:
        fact.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {fact_path} ({len(fact):,} total rows)")


if __name__ == "__main__":
    main()
