"""Parse HLD bulk `res` table into life_expectancy_long."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS

TARGET_AGES = {0, 1, 5, 15, 20, 30, 50, 65}
# HLD Sex codes in bulk file: 1 and 2 (male/female). Confirmed counts ~equal.
SEX_MAP = {1: "male", 2: "female", 3: "both", 0: "both"}


def find_res_file() -> Path | None:
    candidates = [
        RAW / "hld" / "extracted" / "res",
        RAW / "hld" / "extracted" / "res.csv",
        RAW / "hld" / "res",
        RAW / "hld" / "res.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            return p
    # search
    root = RAW / "hld"
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.stat().st_size > 1_000_000:
                # peek header
                try:
                    head = p.open("rb").read(80).decode("utf-8", errors="ignore")
                    if "Country" in head and "e(x)" in head:
                        return p
                except OSError:
                    continue
    return None


def parse_hld_res(path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    usecols = ["Country", "Year1", "Year2", "Sex", "Age", "e(x)", "l(x)"]
    today = date.today().isoformat()
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["Age"].isin(TARGET_AGES)]
        # drop absurd years (data-entry noise)
        chunk = chunk[(chunk["Year1"] >= 1500) & (chunk["Year1"] <= 2100)]
        chunk = chunk.dropna(subset=["e(x)", "Country", "Age"])
        # drop non-physical e(x)
        chunk = chunk[(chunk["e(x)"] > 0) & (chunk["e(x)"] < 120)]
        if chunk.empty:
            continue

        year = chunk["Year1"].where(chunk["Year1"] == chunk["Year2"], other=pd.NA)
        period_start = chunk["Year1"].where(chunk["Year1"] != chunk["Year2"], other=pd.NA)
        period_end = chunk["Year2"].where(chunk["Year1"] != chunk["Year2"], other=pd.NA)
        # also store Year1 as year when single-year
        year = chunk["Year1"].where(chunk["Year1"] == chunk["Year2"], other=chunk["Year1"])

        out = pd.DataFrame(
            {
                "region_id": chunk["Country"].astype(str).str.upper(),
                "country_region": chunk["Country"].astype(str),
                "year": year.values,
                "period_start": period_start.values,
                "period_end": period_end.values,
                "sex": chunk["Sex"].map(SEX_MAP).fillna("both"),
                "age": chunk["Age"].astype(int),
                "life_expectancy": chunk["e(x)"].astype(float),
                "survival_probability": pd.NA,
                "infant_mortality_rate": pd.NA,
                "measure_type": "period",
                "population_type": "national",
                "table_type": "hld_bulk_res",
                "data_quality_flag": "hld_published_table",
                "source_id": "hld",
                "notes": "HLD bulk res table; Sex 1=male, 2=female",
                "retrieved_at": today,
            }
        )
        # survival: compute within (country, year1, year2, sex) vs age 0 lx if present in chunk
        # better second pass: store lx temporarily
        out["_lx"] = chunk["l(x)"].values
        out["_y1"] = chunk["Year1"].values
        out["_y2"] = chunk["Year2"].values
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=FACT_COLUMNS)

    df = pd.concat(parts, ignore_index=True)

    # survival probability l(x)/l(0) within table key
    keys = ["region_id", "_y1", "_y2", "sex"]
    base = (
        df.loc[df["age"] == 0, keys + ["_lx"]]
        .drop_duplicates(keys)
        .rename(columns={"_lx": "_l0"})
    )
    df = df.merge(base, on=keys, how="left")
    df["survival_probability"] = df["_lx"] / df["_l0"]
    df.loc[df["_l0"].isna() | (df["_l0"] == 0), "survival_probability"] = pd.NA
    # clamp / null invalid survival ratios (HLD tables can have nonstandard radices)
    sp = pd.to_numeric(df["survival_probability"], errors="coerce")
    df.loc[sp.notna() & ((sp < 0) | (sp > 1.0)), "survival_probability"] = pd.NA
    df = df.drop(columns=["_lx", "_y1", "_y2", "_l0"])
    return df[FACT_COLUMNS]


def main() -> None:
    ensure_dirs()
    path = find_res_file()
    if path is None:
        print(
            "No HLD res file found. Run: python -m src.acquisition.download_hld\n"
            "Expected data/raw/hld/extracted/res (CSV bulk export)."
        )
        INTERIM.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=FACT_COLUMNS).to_csv(INTERIM / "hld_inventory_parse.csv", index=False)
        return

    print(f"Parsing HLD bulk file: {path} ({path.stat().st_size:,} bytes)")
    combined = parse_hld_res(path)
    interim = INTERIM / "hld_inventory_parse.csv"
    combined.to_csv(interim, index=False)
    print(f"Wrote {interim} ({len(combined):,} rows)")

    if combined.empty:
        print("HLD parse produced 0 rows after filters.")
        return

    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists() and fact_path.stat().st_size > 50:
        existing = pd.read_csv(fact_path)
        if len(existing):
            existing = existing[existing["source_id"].astype(str) != "hld"]
            combined = pd.concat([existing, combined], ignore_index=True)
    combined.to_csv(fact_path, index=False)
    try:
        combined.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {fact_path} ({len(combined):,} total rows)")
    print(
        f"HLD ages: {sorted(combined.loc[combined['source_id']=='hld','age'].unique())}; "
        f"countries: {combined.loc[combined['source_id']=='hld','region_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
