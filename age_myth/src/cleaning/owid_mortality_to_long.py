"""Join OWID infant/child mortality onto OWID e0 rows; also write IMR-enriched interim."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS


def _slug_region(entity: str, code) -> str:
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


def load_owid_metric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    entity = lower.get("entity") or list(df.columns)[0]
    year = lower.get("year")
    code = lower.get("code")
    skip = {"entity", "code", "year"}
    val_col = next(c for c in df.columns if c.lower() not in skip)
    out = pd.DataFrame(
        {
            "region_id": [
                _slug_region(e, df[code].iloc[i] if code else None)
                for i, e in enumerate(df[entity])
            ],
            "country_region": df[entity].astype(str),
            "year": pd.to_numeric(df[year], errors="coerce"),
            "value": pd.to_numeric(df[val_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["year", "value"])


def main() -> None:
    ensure_dirs()
    raw = RAW / "owid"
    imr_path = raw / "infant-mortality.csv"
    u5_path = raw / "child-mortality.csv"
    if not imr_path.exists():
        print("Missing infant-mortality.csv. Run: python -m src.acquisition.download_owid")
        raise SystemExit(1)

    imr = load_owid_metric(imr_path)
    imr = imr.rename(columns={"value": "infant_mortality_rate"})
    print(f"OWID IMR rows: {len(imr):,}")

    u5 = None
    if u5_path.exists():
        u5 = load_owid_metric(u5_path)
        u5 = u5.rename(columns={"value": "child_mortality_rate"})
        print(f"OWID U5MR rows: {len(u5):,}")

    # Enrich OWID e0 interim if present
    owid_e0 = INTERIM / "owid_life_expectancy_long.csv"
    if owid_e0.exists():
        fact = pd.read_csv(owid_e0, low_memory=False)
        # only attach to age 0 e0 series
        e0_mask = (fact["age"] == 0) & (
            fact["source_id"].isin(["owid_le_longrun", "owid_le_hmd_unwpp"])
        )
        key = ["region_id", "year"]
        imr_k = imr[key + ["infant_mortality_rate"]].drop_duplicates(key)
        # merge onto e0 subset
        left = fact.loc[e0_mask].drop(columns=["infant_mortality_rate"], errors="ignore")
        left = left.merge(imr_k, on=key, how="left")
        fact.loc[e0_mask, "infant_mortality_rate"] = left["infant_mortality_rate"].values
        # store U5 in notes column would be lossy; keep separate interim
        fact.to_csv(owid_e0, index=False)
        print(
            f"Enriched OWID e0 IMR coverage: "
            f"{fact.loc[e0_mask, 'infant_mortality_rate'].notna().mean()*100:.1f}%"
        )
    else:
        print("No owid_life_expectancy_long.csv yet; writing mortality-only interim")

    # mortality-only interim for modeling joins
    mort = imr.copy()
    mort["sex"] = "both"
    mort["age"] = 0
    mort["source_id"] = "owid_imr"
    mort["measure"] = "infant_mortality_rate"
    if u5 is not None:
        u5b = u5.copy()
        u5b["sex"] = "both"
        u5b["age"] = 0
        u5b["source_id"] = "owid_u5mr"
        u5b["measure"] = "child_mortality_rate"
        u5b = u5b.rename(columns={"child_mortality_rate": "value"})
        mort2 = mort.rename(columns={"infant_mortality_rate": "value"})
        mort_out = pd.concat(
            [
                mort2[["region_id", "country_region", "year", "sex", "age", "source_id", "measure", "value"]],
                u5b[["region_id", "country_region", "year", "sex", "age", "source_id", "measure", "value"]],
            ],
            ignore_index=True,
        )
    else:
        mort_out = mort.rename(columns={"infant_mortality_rate": "value"})
        mort_out["measure"] = "infant_mortality_rate"
        mort_out = mort_out[
            ["region_id", "country_region", "year", "sex", "age", "source_id", "measure", "value"]
        ]

    mort_path = INTERIM / "owid_mortality_long.csv"
    mort_out.to_csv(mort_path, index=False)
    print(f"Wrote {mort_path} ({len(mort_out):,} rows)")

    # Update processed fact: re-apply IMR onto owid e0 rows in full fact table
    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists():
        full = pd.read_csv(fact_path, low_memory=False)
        e0_mask = (full["age"] == 0) & (
            full["source_id"].isin(["owid_le_longrun", "owid_le_hmd_unwpp"])
        )
        imr_k = imr[["region_id", "year", "infant_mortality_rate"]].drop_duplicates(
            ["region_id", "year"]
        )
        left = full.loc[e0_mask, ["region_id", "year"]].reset_index()
        merged = left.merge(imr_k, on=["region_id", "year"], how="left")
        full.loc[merged["index"].values, "infant_mortality_rate"] = merged[
            "infant_mortality_rate"
        ].values
        full.to_csv(fact_path, index=False)
        try:
            full.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            print(f"Parquet skip: {e}")
        print(
            f"Processed OWID e0 IMR coverage: "
            f"{full.loc[e0_mask, 'infant_mortality_rate'].notna().mean()*100:.1f}%"
        )


if __name__ == "__main__":
    main()
