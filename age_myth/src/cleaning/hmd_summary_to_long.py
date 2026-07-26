"""Parse HMD public summary workbooks into life_expectancy_long rows."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS

NAME_TO_CODE = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Chile": "CHL",
    "Croatia": "HRV",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRATNP",
    "Germany": "DEUTNP",
    "Greece": "GRC",
    "Hong Kong": "HKG",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Netherlands": "NLD",
    "New Zealand": "NZL_NP",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Republic of Korea": "KOR",
    "Korea": "KOR",
    "Russia": "RUS",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Taiwan": "TWN",
    "U.K.": "GBR_NP",
    "UK": "GBR_NP",
    "United Kingdom": "GBR_NP",
    "U.S.A.": "USA",
    "USA": "USA",
    "United States": "USA",
    "Ukraine": "UKR",
}


def _region_id(name: str) -> str:
    name = str(name).strip()
    if name in NAME_TO_CODE:
        return NAME_TO_CODE[name]
    for k, v in NAME_TO_CODE.items():
        if k.lower() == name.lower():
            return v
    return name.upper().replace(" ", "_").replace(".", "").replace("'", "")


def _sex_from_sheet(sheet: str) -> str:
    s = sheet.lower()
    if "female" in s:
        return "female"
    if "male" in s:
        return "male"
    return "both"


def melt_wide_sheet(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_idx = None
    for i in range(min(10, len(raw))):
        if str(raw.iloc[i, 0]).strip().lower() == "year":
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    headers = [
        str(c).strip() if pd.notna(c) else f"col_{j}" for j, c in enumerate(raw.iloc[header_idx])
    ]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = headers
    df = df.rename(columns={headers[0]: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    country_cols = [c for c in df.columns if c != "year"]
    long = df.melt(
        id_vars=["year"], value_vars=country_cols, var_name="country_region", value_name="value"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["year"] = long["year"].astype(int)
    long["sex"] = _sex_from_sheet(sheet)
    long["region_id"] = long["country_region"].map(_region_id)
    return long


def parse_ex(path: Path) -> pd.DataFrame:
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        if sheet == "Introduction":
            continue
        sl = sheet.lower().replace(" ", "")
        if "e65" in sl:
            age = 65
        elif "e80" in sl:
            age = 80
        elif "e0" in sl:
            age = 0
        else:
            continue
        part = melt_wide_sheet(path, sheet)
        if part.empty:
            continue
        part["age"] = age
        part["life_expectancy"] = part["value"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_imr(path: Path) -> pd.DataFrame:
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        if "imr" not in sheet.lower():
            continue
        part = melt_wide_sheet(path, sheet)
        if part.empty:
            continue
        part["infant_mortality_rate"] = part["value"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_surv(path: Path) -> pd.DataFrame:
    frames = []
    for sheet in pd.ExcelFile(path).sheet_names:
        if "surv" not in sheet.lower() and "0 to 65" not in sheet.lower():
            continue
        part = melt_wide_sheet(path, sheet)
        if part.empty:
            continue
        # percent -> probability
        part["survival_probability"] = part["value"] / 100.0
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_fact(ex: pd.DataFrame, imr: pd.DataFrame, surv: pd.DataFrame) -> pd.DataFrame:
    today = date.today().isoformat()
    if ex.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)

    fact = pd.DataFrame(
        {
            "region_id": ex["region_id"],
            "country_region": ex["country_region"],
            "year": ex["year"],
            "period_start": pd.NA,
            "period_end": pd.NA,
            "sex": ex["sex"],
            "age": ex["age"].astype(int),
            "life_expectancy": ex["life_expectancy"].astype(float),
            "survival_probability": pd.NA,
            "infant_mortality_rate": pd.NA,
            "measure_type": "period",
            "population_type": "national",
            "table_type": "hmd_summary_public",
            "data_quality_flag": "hmd_complete",
            "source_id": "hmd_summary_public",
            "notes": "HMD public summary indicators (no registration)",
            "retrieved_at": today,
        }
    )

    key = ["region_id", "year", "sex"]
    if not imr.empty:
        imr_k = imr[key + ["infant_mortality_rate"]].drop_duplicates(key)
        fact = fact.merge(imr_k, on=key, how="left", suffixes=("", "_y"))
        if "infant_mortality_rate_y" in fact.columns:
            fact["infant_mortality_rate"] = fact["infant_mortality_rate_y"]
            fact = fact.drop(columns=["infant_mortality_rate_y"])

    if not surv.empty:
        # S(0→65) attaches to age-65 rows
        surv_k = surv[key + ["survival_probability"]].drop_duplicates(key)
        age65 = fact["age"] == 65
        left = fact.loc[age65, key].reset_index()
        merged = left.merge(surv_k, on=key, how="left")
        fact.loc[merged["index"].values, "survival_probability"] = merged["survival_probability"].values

    # physical ranges
    fact = fact[
        (fact["life_expectancy"] > 0)
        & (fact["life_expectancy"] < 120)
        & (fact["year"] >= 1500)
        & (fact["year"] <= 2100)
    ]
    return fact[FACT_COLUMNS]


def main() -> None:
    ensure_dirs()
    raw = RAW / "hmd_summary"
    ex_path = raw / "hmd_summary_ex_0_65_80.xlsx"
    imr_path = raw / "hmd_summary_IMR.xlsx"
    surv_path = raw / "hmd_summary_px_0_to_65.xlsx"
    for p in (ex_path, imr_path, surv_path):
        if not p.exists():
            print(f"Missing {p}. Run: python -m src.acquisition.download_hmd_summary")
            raise SystemExit(1)

    ex = parse_ex(ex_path)
    imr = parse_imr(imr_path)
    surv = parse_surv(surv_path)
    print(f"ex cells={len(ex):,} imr={len(imr):,} surv={len(surv):,}")
    fact = build_fact(ex, imr, surv)
    interim = INTERIM / "hmd_summary_life_expectancy_long.csv"
    fact.to_csv(interim, index=False)
    imr_cov = fact["infant_mortality_rate"].notna().mean() * 100
    surv_cov = fact.loc[fact["age"] == 65, "survival_probability"].notna().mean() * 100
    print(f"Wrote {interim} ({len(fact):,} rows); IMR cov={imr_cov:.1f}%; age65 surv cov={surv_cov:.1f}%")

    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists() and fact_path.stat().st_size > 50:
        existing = pd.read_csv(fact_path, low_memory=False)
        if len(existing):
            existing = existing[existing["source_id"].astype(str) != "hmd_summary_public"]
            fact = pd.concat([existing, fact], ignore_index=True)
    fact.to_csv(fact_path, index=False)
    try:
        fact.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {fact_path} ({len(fact):,} total rows)")


if __name__ == "__main__":
    main()
