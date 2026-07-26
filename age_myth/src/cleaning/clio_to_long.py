"""Parse Clio-Infra / Zijdeman life expectancy at birth spreadsheet if present."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS


def find_clio_file() -> Path | None:
    d = RAW / "clio_infra"
    if not d.exists():
        return None
    for p in d.iterdir():
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"} and p.stat().st_size > 1000:
            return p
    return None


def parse_clio(path: Path) -> pd.DataFrame:
    today = date.today().isoformat()

    if path.suffix.lower() in {".xlsx", ".xls"}:
        # Zijdeman Clio layout: title rows, header row with "country name", year columns
        raw = pd.read_excel(path, sheet_name=0, header=None)
        header_idx = None
        for i in range(min(15, len(raw))):
            row_vals = [str(v).strip().lower() for v in raw.iloc[i].tolist() if pd.notna(v)]
            if "country name" in row_vals or (
                "country" in row_vals and any(str(x).replace(".0", "").isdigit() for x in row_vals)
            ):
                header_idx = i
                break
        if header_idx is None:
            print("Could not find Clio header row with 'country name'")
            return pd.DataFrame(columns=FACT_COLUMNS)
        headers = []
        for j, v in enumerate(raw.iloc[header_idx].tolist()):
            if pd.isna(v):
                headers.append(f"col_{j}")
            else:
                headers.append(str(v).strip())
        df = raw.iloc[header_idx + 1 :].copy()
        df.columns = headers
        df = df.dropna(how="all")
    else:
        df = pd.read_csv(path)

    # locate country column
    country_c = None
    for c in df.columns:
        if str(c).strip().lower() in {"country name", "country", "nation", "entity"}:
            country_c = c
            break
    if country_c is None:
        print(f"No country column. Columns sample: {list(df.columns)[:12]}")
        return pd.DataFrame(columns=FACT_COLUMNS)

    year_cols = []
    for c in df.columns:
        try:
            y = int(float(str(c).strip()))
            if 1000 <= y <= 2100:
                year_cols.append(c)
        except (TypeError, ValueError):
            continue

    if not year_cols:
        print("No year columns found in Clio file")
        return pd.DataFrame(columns=FACT_COLUMNS)

    long = df.melt(
        id_vars=[country_c],
        value_vars=year_cols,
        var_name="year",
        value_name="life_expectancy",
    )
    long = long.dropna(subset=["life_expectancy"])
    long = long[long[country_c].notna()]
    # drop non-numeric values
    long["life_expectancy"] = pd.to_numeric(long["life_expectancy"], errors="coerce")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long = long.dropna(subset=["life_expectancy", "year"])
    long = long[(long["life_expectancy"] > 0) & (long["life_expectancy"] < 120)]

    out = pd.DataFrame(
        {
            "region_id": long[country_c]
            .astype(str)
            .str.upper()
            .str.replace(r"[^A-Z0-9]+", "_", regex=True)
            .str.strip("_"),
            "country_region": long[country_c].astype(str),
            "year": long["year"].astype(int),
            "period_start": pd.NA,
            "period_end": pd.NA,
            "sex": "both",
            "age": 0,
            "life_expectancy": long["life_expectancy"].astype(float),
            "survival_probability": pd.NA,
            "infant_mortality_rate": pd.NA,
            "measure_type": "period",
            "population_type": "national",
            "table_type": "clio_e0_wide",
            "data_quality_flag": "clio_historical",
            "source_id": "clio_zijdeman_2015",
            "notes": f"Clio-Infra Zijdeman wide melt from {path.name}",
            "retrieved_at": today,
        }
    )
    return out[FACT_COLUMNS]


def main() -> None:
    ensure_dirs()
    path = find_clio_file()
    if path is None:
        print("No Clio file in data/raw/clio_infra. Run download_clio.")
        return
    print(f"Parsing {path}")
    # openpyxl may be required
    try:
        combined = parse_clio(path)
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
        combined = parse_clio(path)

    interim = INTERIM / "clio_life_expectancy_long.csv"
    combined.to_csv(interim, index=False)
    print(f"Wrote {interim} ({len(combined):,} rows)")

    if combined.empty:
        return

    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists() and fact_path.stat().st_size > 50:
        existing = pd.read_csv(fact_path)
        if len(existing):
            existing = existing[existing["source_id"].astype(str) != "clio_zijdeman_2015"]
            combined = pd.concat([existing, combined], ignore_index=True)
    combined.to_csv(fact_path, index=False)
    try:
        combined.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {fact_path} ({len(combined):,} total rows)")


if __name__ == "__main__":
    main()
