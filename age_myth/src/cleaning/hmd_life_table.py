"""Parse HMD-format period life tables into long e(x) + survival rows."""
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.paths import FIXTURES, INTERIM, PROCESSED, RAW, ensure_dirs
from src.schema.create_templates import FACT_COLUMNS

TARGET_AGES = {0, 1, 5, 15, 20, 30, 50, 65, 80}


def write_fixture() -> Path:
    """Write a small HMD-like sample (illustrative rates, not real HMD values)."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / "hmd_sweden_sample.txt"
    # Minimal 1x1 both-sex style: Year, Age, mx, qx, ax, lx, dx, Lx, Tx, ex
    # Header layout matches common HMD text exports.
    lines = [
        "Sweden, Life tables (period 1x1), Total  [fixture demo — NOT official HMD data]",
        "Year          Age         mx         qx         ax         lx         dx         Lx         Tx         ex",
    ]
    # Two years, ages 0,1,5,15,20,30,50,65
    demo = {
        1800: {
            0: (0.20, 100000, 35.0),
            1: (0.05, 80000, 42.0),
            5: (0.01, 70000, 45.0),
            15: (0.008, 62000, 40.0),
            20: (0.009, 60000, 36.0),
            30: (0.012, 55000, 30.0),
            50: (0.025, 40000, 18.0),
            65: (0.05, 25000, 10.0),
        },
        1900: {
            0: (0.10, 100000, 50.0),
            1: (0.02, 90000, 54.0),
            5: (0.005, 85000, 55.0),
            15: (0.004, 82000, 48.0),
            20: (0.005, 80000, 44.0),
            30: (0.007, 76000, 36.0),
            50: (0.015, 60000, 22.0),
            65: (0.04, 40000, 12.0),
        },
    }
    for year, ages in demo.items():
        for age, (mx, lx, ex) in ages.items():
            qx = min(mx * 0.9, 0.99)
            ax = 0.5
            dx = int(lx * qx)
            Lx = int(lx - dx / 2)
            Tx = int(ex * lx)
            lines.append(
                f"{year:>4} {age:>12} {mx:10.5f} {qx:10.5f} {ax:10.2f} "
                f"{lx:10.0f} {dx:10.0f} {Lx:10.0f} {Tx:10.0f} {ex:10.2f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_hmd_life_table(
    path: Path,
    *,
    region_id: str,
    country_region: str,
    sex: str,
    source_id: str,
    quality_flag: str,
) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    # find header line with Year and Age
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bYear\b", line) and re.search(r"\bAge\b", line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No Year/Age header in {path}")

    header = re.split(r"\s+", lines[header_idx].strip())
    # HMD uses fixed-ish whitespace; use pandas read_fwf-like via whitespace
    data_lines = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        if line.strip().startswith("#"):
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 3:
            continue
        data_lines.append(parts)

    if not data_lines:
        raise ValueError(f"No data rows in {path}")

    # Map columns by header names when lengths match; else positional HMD order
    # Year Age mx qx ax lx dx Lx Tx ex
    rows = []
    for parts in data_lines:
        try:
            year = int(float(parts[0]))
            age_raw = parts[1]
            if age_raw.endswith("+"):
                age = int(age_raw[:-1])
            else:
                age = int(float(age_raw))
        except ValueError:
            continue
        if age not in TARGET_AGES:
            continue
        # find ex: last column typically
        try:
            ex = float(parts[-1])
        except ValueError:
            continue
        lx = None
        if len(parts) >= 6:
            try:
                lx = float(parts[5])
            except ValueError:
                lx = None
        rows.append({"year": year, "age": age, "ex": ex, "lx": lx})

    if not rows:
        raise ValueError(f"No target-age rows parsed from {path}")

    df = pd.DataFrame(rows)
    # survival relative to lx at age 0 within year
    surv_parts: list[pd.Series] = []
    for _, g in df.groupby("year"):
        base = g.loc[g["age"] == 0, "lx"]
        if base.empty or pd.isna(base.iloc[0]) or float(base.iloc[0]) == 0:
            surv_parts.append(pd.Series([pd.NA] * len(g), index=g.index, dtype="object"))
        else:
            surv_parts.append(g["lx"] / float(base.iloc[0]))
    df["survival_probability"] = pd.concat(surv_parts).sort_index()

    today = date.today().isoformat()
    out = pd.DataFrame(
        {
            "region_id": region_id,
            "country_region": country_region,
            "year": df["year"],
            "period_start": pd.NA,
            "period_end": pd.NA,
            "sex": sex,
            "age": df["age"],
            "life_expectancy": df["ex"],
            "survival_probability": df["survival_probability"],
            "infant_mortality_rate": pd.NA,
            "measure_type": "period",
            "population_type": "national",
            "table_type": "hmd_1x1",
            "data_quality_flag": quality_flag,
            "source_id": source_id,
            "notes": f"Parsed from {path.name}",
            "retrieved_at": today,
        }
    )
    return out[FACT_COLUMNS]


def sex_from_filename(name: str) -> str:
    n = name.lower()
    if n.startswith("flt") or "female" in n:
        return "female"
    if n.startswith("mlt") or "male" in n:
        return "male"
    return "both"


def discover_hmd_files() -> list[tuple[Path, str, str, str]]:
    """Return list of (path, region_id, country_region, sex)."""
    found: list[tuple[Path, str, str, str]] = []
    root = RAW / "hmd"
    if not root.exists():
        return found
    for path in root.rglob("*ltper*.txt"):
        # parent dir often country code
        code = path.parent.name if path.parent.name != "hmd" else path.stem
        region_id = code
        country = code
        sex = sex_from_filename(path.name)
        found.append((path, region_id, country, sex))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse HMD life tables to long format")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Parse demo fixture instead of real HMD downloads",
    )
    args = parser.parse_args()
    ensure_dirs()

    frames: list[pd.DataFrame] = []
    if args.fixture:
        fpath = write_fixture()
        part = parse_hmd_life_table(
            fpath,
            region_id="SWE",
            country_region="Sweden",
            sex="both",
            source_id="hmd_fixture",
            quality_flag="fixture_demo",
        )
        frames.append(part)
        print(f"Fixture rows: {len(part)}")
    else:
        files = discover_hmd_files()
        if not files:
            print("No HMD files under data/raw/hmd. Falling back to --fixture behavior.")
            fpath = write_fixture()
            part = parse_hmd_life_table(
                fpath,
                region_id="SWE",
                country_region="Sweden",
                sex="both",
                source_id="hmd_fixture",
                quality_flag="fixture_demo",
            )
            frames.append(part)
        else:
            for path, region_id, country, sex in files:
                try:
                    part = parse_hmd_life_table(
                        path,
                        region_id=region_id,
                        country_region=country,
                        sex=sex,
                        source_id="hmd_v6",
                        quality_flag="hmd_complete",
                    )
                    frames.append(part)
                    print(f"Parsed {path}: {len(part)} rows")
                except Exception as e:  # noqa: BLE001
                    print(f"Skip {path}: {e}")

    if not frames:
        raise SystemExit("No HMD data parsed")

    combined = pd.concat(frames, ignore_index=True)
    interim = INTERIM / "hmd_life_expectancy_long.csv"
    combined.to_csv(interim, index=False)

    fact_path = PROCESSED / "life_expectancy_long.csv"
    if fact_path.exists() and fact_path.stat().st_size > 50:
        existing = pd.read_csv(fact_path)
        if len(existing):
            existing = existing[~existing["source_id"].astype(str).isin(["hmd_v6", "hmd_fixture"])]
            combined = pd.concat([existing, combined], ignore_index=True)
    combined.to_csv(fact_path, index=False)
    try:
        combined.to_parquet(PROCESSED / "life_expectancy_long.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"Parquet skip: {e}")
    print(f"Wrote {fact_path} ({len(combined):,} total rows)")


if __name__ == "__main__":
    main()
