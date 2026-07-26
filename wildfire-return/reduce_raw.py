from __future__ import annotations

"""Stream-reduce the raw FPA FOD-Attributes CSVs into small committed caches.

Reads the 29 annual CSVs in data/raw/ (~5 GB, 308 columns) in 250k-row chunks,
keeps the ~50 curated columns defined in wildfire.RAW_COLUMNS, and writes:

    data/national_annual.csv    state x year x size-class x cause-class: n, acres
    data/national_monthly.csv   state x year x month x cause-class: n
    data/conus_grid.parquet     0.1-degree cells x era: n, acres, n_human, n_natural
    data/ca_fires.parquet       fire-level California, curated columns
    data/fl_fires.parquet       fire-level Florida, curated columns
    data/tubbs_record.json      the full 308-column row of the 2017 Tubbs Fire
    data/geo/us_states_20m.json Census state boundaries as plain polylines
    data/reduce_report.json     per-year QA + corpus checks + column mapping
    data/MANIFEST.json          sha256 of every committed cache

Fail-loud philosophy: any missing curated column, any date-parse degeneracy
(the 2020 notebook's every-fire-is-Jan-1-1970 bug), or an ambiguous Tubbs match
stops the run with a diagnostic instead of producing a quietly wrong cache.

Usage:  python reduce_raw.py            (needs data/raw/ from fetch_raw.py)
"""

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import wildfire as wf

CHUNK_ROWS = 250_000

RAW_TO_CANON = {raw: canon for canon, raw in wf.RAW_COLUMNS.items()}

# Read EVERYTHING as string except the six columns that must be numerically clean
# (those fail loudly on junk). All other numerics go through to_numeric(coerce)
# in clean_chunk. This also sidesteps a pandas-3.0 C-parser crash when a
# mixed-dtype DtypeWarning fires under usecols.
READ_DTYPES = {raw: "str" for raw in wf.RAW_COLUMNS.values()}
READ_DTYPES.update({
    "FOD_ID": "int64", "FIRE_YEAR": "int32",
    "FIRE_SIZE": "float64", "LATITUDE": "float64", "LONGITUDE": "float64",
})


def annual_files() -> list[Path]:
    files = sorted(wf.RAW.glob("*_FPA_FOD_cons.csv"))
    if not files:
        raise SystemExit(f"No raw CSVs in {wf.RAW} -- run fetch_raw.py --yes first")
    return files


def validate_headers(files: list[Path]) -> None:
    """Every curated raw column must exist in every year file. No silent guessing."""
    problems = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        missing = [raw for raw in wf.RAW_COLUMNS.values() if raw not in header]
        if missing:
            problems.append(f"{path.name}: missing {missing}")
    if problems:
        print("COLUMN DISCOVERY FAILED -- curated columns absent from raw files:")
        print("\n".join(problems))
        print("\nFull header of first failing file for inspection:")
        print(list(pd.read_csv(files[0], nrows=0).columns))
        raise SystemExit(1)
    print(f"[headers] all {len(wf.RAW_COLUMNS)} curated columns present in {len(files)} year files")


def parse_discovery_dates(raw: pd.Series, context: str) -> pd.Series:
    """Parse DISCOVERY_DATE, with the fix 2020 needed if it is ever Julian-numeric.

    The 2020 notebook ran pd.to_datetime() on Julian day numbers and got
    1970-01-01 for every fire (nanoseconds since epoch), silently killing the
    MONTH and DAY_OF_WEEK features. Here the equivalent failure raises.
    """
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.9 and numeric.dropna().median() > 2_400_000:
        # Julian day numbers, the 2015-era storage format -- the one-argument fix:
        dates = pd.to_datetime(numeric, unit="D", origin="julian")
    else:
        dates = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    ok = dates.notna()
    if ok.mean() < 0.999:
        raise SystemExit(f"[dates] {context}: {(~ok).sum()} of {len(dates)} DISCOVERY_DATE "
                         f"values failed to parse -- format drifted, refusing to continue")
    in_range = dates.dt.year.between(1992, 2020)
    if not in_range[ok].all():
        bad = dates[ok & ~in_range]
        raise SystemExit(f"[dates] {context}: {len(bad)} parsed dates outside 1992-2020 "
                         f"(e.g. {bad.iloc[0]}) -- this is the 1970 bug class, refusing")
    return dates


def clean_chunk(chunk: pd.DataFrame, year: int, context: str) -> pd.DataFrame:
    df = chunk.rename(columns=RAW_TO_CANON)
    df["discovery_date"] = parse_discovery_dates(df["discovery_date"], context)
    df["discovery_doy_raw"] = pd.to_numeric(df["discovery_doy_raw"], errors="coerce").astype("float32")
    df["firestations_10km"] = pd.to_numeric(df["firestations_10km"], errors="coerce").astype("float32")
    # sentinels -> NaN on numeric columns.
    # Two kinds: the enumerated codes (-9999, 32767, ...) and OVERFLOW FILLS --
    # float32-max (3.4e38) used as nodata in ghm/sdi. The Round-2 audit scanned
    # for sentinel *strings* and never saw the second kind, so it rode into the
    # Round-1 models as a real "human modification" value on 367 CA rows.
    # No physical quantity in this dataset is ~1e30, so the rule is safe.
    for col in wf.FLOAT32_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.mask(s.isin(wf.SENTINELS))
        s = s.mask(s.abs() >= wf.OVERFLOW_FILL)
        df[col] = s.astype("float32")
    # containment (m/d/Y H:M:S, frequently missing) -> burn duration in days
    cont = pd.to_datetime(df["cont_date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    burn = (cont - df["discovery_date"]).dt.days
    df["burn_days"] = burn.where((burn >= 0) & (burn < 400)).astype("float32")
    df = df.drop(columns=["cont_date"])
    # derived time features -- the ones the 2020 notebook thought it had
    df["month"] = df["discovery_date"].dt.month.astype("int8")
    df["dow"] = df["discovery_date"].dt.dayofweek.astype("int8")
    df["doy_std"] = wf.doy_std(df["discovery_date"])
    df["year_match"] = (df["discovery_date"].dt.year == df["fire_year"])
    df["cause_group"] = df["cause_general"].map(wf.CAUSE_GROUPS)
    df["old13"] = df["cause_general"].map(wf.NWCG_TO_OLD13)
    return df


def main() -> None:
    files = annual_files()
    years = [int(p.name[:4]) for p in files]
    print(f"[reduce] {len(files)} year files: {min(years)}-{max(years)}")
    validate_headers(files)

    ann_parts, mon_parts, grid_parts = [], [], []
    ca_parts, fl_parts = [], []
    report_files = []
    month_seen, dow_seen = set(), set()
    rows_total = 0
    year_mismatch_total = 0
    missing_counts: dict[str, int] = {}
    cause_class_counts: dict[str, int] = {}
    cause_general_counts: dict[str, int] = {}

    for path in files:
        year = int(path.name[:4])
        n_rows = n_ca = n_fl = 0
        for i, chunk in enumerate(pd.read_csv(path, usecols=list(wf.RAW_COLUMNS.values()),
                                              dtype=READ_DTYPES, chunksize=CHUNK_ROWS)):
            df = clean_chunk(chunk, year, f"{path.name} chunk {i}")
            n_rows += len(df)
            month_seen.update(df["month"].unique().tolist())
            dow_seen.update(df["dow"].unique().tolist())
            year_mismatch_total += int((~df["year_match"]).sum())
            for col in ("cause_general", "fire_size", "lat", "lon", "erc", "vpd", "ndvi_1day", "svi"):
                missing_counts[col] = missing_counts.get(col, 0) + int(df[col].isna().sum())
            for val, cnt in df["cause_class"].value_counts(dropna=False).items():
                cause_class_counts[str(val)] = cause_class_counts.get(str(val), 0) + int(cnt)
            for val, cnt in df["cause_general"].value_counts(dropna=False).items():
                cause_general_counts[str(val)] = cause_general_counts.get(str(val), 0) + int(cnt)

            ann_parts.append(df.groupby(["state", "fire_year", "size_class", "cause_class"], dropna=False)
                             .agg(n=("fod_id", "size"), acres=("fire_size", "sum")).reset_index())
            mon_parts.append(df.groupby(["state", "fire_year", "month", "cause_class"], dropna=False)
                             .agg(n=("fod_id", "size")).reset_index())

            g = df[df["lat"].notna() & df["lon"].notna()].copy()
            g["lat_bin"] = (np.floor(g["lat"] * 10) / 10).astype("float32")
            g["lon_bin"] = (np.floor(g["lon"] * 10) / 10).astype("float32")
            g["era"] = np.where(g["fire_year"] < wf.ERA_SPLIT, "1992-2005", "2006-2020")
            g["is_human"] = (g["cause_class"] == "Human").astype("int32")
            g["is_natural"] = (g["cause_class"] == "Natural").astype("int32")
            grid_parts.append(g.groupby(["lat_bin", "lon_bin", "era"])
                              .agg(n=("fod_id", "size"), acres=("fire_size", "sum"),
                                   n_human=("is_human", "sum"), n_natural=("is_natural", "sum"))
                              .reset_index())

            ca = df[df["state"] == "CA"]
            fl = df[df["state"] == "FL"]
            n_ca += len(ca)
            n_fl += len(fl)
            keep = [c for c in df.columns if c not in ("year_match",)]
            if len(ca):
                ca_parts.append(ca[keep])
            if len(fl):
                fl_parts.append(fl[keep])
        rows_total += n_rows
        report_files.append({"year": year, "rows": n_rows, "ca_rows": n_ca, "fl_rows": n_fl})
        print(f"  [year] {year}: {n_rows:,} fires ({n_ca:,} CA, {n_fl:,} FL)", flush=True)

    # ------------------------------------------------------------------ corpus checks
    if len(month_seen) < 12 or len(dow_seen) != 7:
        raise SystemExit(f"[degeneracy] months seen: {sorted(month_seen)}; weekdays seen: "
                         f"{sorted(dow_seen)} -- the 2020 constant-feature bug would look "
                         f"exactly like this. Refusing to write caches.")
    mismatch_share = year_mismatch_total / rows_total
    if mismatch_share > 0.001:
        raise SystemExit(f"[dates] discovery year != FIRE_YEAR for {mismatch_share:.2%} of rows")
    print(f"[checks] months {len(month_seen)}/12, weekdays {len(dow_seen)}/7, "
          f"year-mismatch {mismatch_share:.4%} -- the 2020 degeneracy cannot recur silently")

    # ------------------------------------------------------------------ write caches
    ann = (pd.concat(ann_parts).groupby(["state", "fire_year", "size_class", "cause_class"], dropna=False)
           .sum().reset_index())
    ann.to_csv(wf.DATA / "national_annual.csv", index=False)
    mon = (pd.concat(mon_parts).groupby(["state", "fire_year", "month", "cause_class"], dropna=False)
           .sum().reset_index())
    mon.to_csv(wf.DATA / "national_monthly.csv", index=False)
    grid = (pd.concat(grid_parts).groupby(["lat_bin", "lon_bin", "era"]).sum().reset_index())
    grid.to_parquet(wf.DATA / "conus_grid.parquet", index=False)

    for name, parts in (("ca_fires.parquet", ca_parts), ("fl_fires.parquet", fl_parts)):
        state_df = pd.concat(parts, ignore_index=True)
        for col in wf.CATEGORY_COLS:
            state_df[col] = state_df[col].astype("category")
        state_df.to_parquet(wf.DATA / name, index=False)
        print(f"  [cache] {name}: {len(state_df):,} rows, "
              f"{(wf.DATA / name).stat().st_size / (1 << 20):.1f} MB")

    # ------------------------------------------------------------------ Tubbs record
    ca_all = pd.concat(ca_parts, ignore_index=True)
    named = ca_all[(ca_all["fire_year"] == 2017)
                   & ca_all["fire_name"].astype("string").str.upper().str.contains("TUBBS", na=False)]
    if len(named) != 1:
        print(named.to_string())
        raise SystemExit(f"[tubbs] expected exactly 1 CA-2017 TUBBS match, got {len(named)}")
    fod_id = int(named["fod_id"].iloc[0])
    if fod_id != wf.TUBBS_FOD_ID:
        raise SystemExit(f"[tubbs] FOD_ID {fod_id} != expected {wf.TUBBS_FOD_ID} -- dataset "
                         f"version changed; update wildfire.TUBBS_FOD_ID after verifying")
    full_row = None
    src_2017 = wf.RAW / "2017_FPA_FOD_cons.csv"
    for chunk in pd.read_csv(src_2017, chunksize=CHUNK_ROWS, low_memory=False):
        hit = chunk[chunk["FOD_ID"] == fod_id]
        if len(hit):
            full_row = hit.iloc[0]
            break
    if full_row is None:
        raise SystemExit(f"[tubbs] FOD_ID {fod_id} vanished on full-width re-read")
    record = {
        "provenance": {
            "source_file": src_2017.name,
            "zenodo_record": "8381129",
            "predicate": "STATE=='CA' & FIRE_YEAR==2017 & FIRE_NAME contains 'TUBBS'",
            "fod_id": fod_id,
            "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "curated": {k: (None if pd.isna(v) else (v.isoformat() if isinstance(v, pd.Timestamp) else
                    (int(v) if isinstance(v, (np.integer,)) else
                     (float(v) if isinstance(v, (np.floating,)) else v))))
                    for k, v in named.iloc[0].items()},
        "full_raw_row": {k: (None if (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
                             else (int(v) if isinstance(v, (np.integer,))
                                   else (float(v) if isinstance(v, (np.floating,)) else str(v))))
                         for k, v in full_row.items() if k != "geometry"},
    }
    (wf.DATA / "tubbs_record.json").write_text(json.dumps(record, indent=2))
    print(f"[tubbs] TUBBS FOUND: FOD_ID {fod_id}, {record['curated']['fire_size']:.0f} acres, "
          f"discovered {record['curated']['discovery_date'][:10]} -- full row saved")

    # ------------------------------------------------------------------ geo conversion
    convert_geo()

    # ------------------------------------------------------------------ report + manifest
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_total": rows_total,
        "files": report_files,
        "column_mapping": wf.RAW_COLUMNS,
        "corpus": {
            "months_seen": sorted(int(m) for m in month_seen),
            "weekdays_seen": sorted(int(d) for d in dow_seen),
            "year_mismatch_share": mismatch_share,
            "missing_shares": {k: v / rows_total for k, v in missing_counts.items()},
            "cause_class_counts": cause_class_counts,
            "cause_general_counts": cause_general_counts,
        },
    }
    (wf.DATA / "reduce_report.json").write_text(json.dumps(report, indent=2))

    committed = ["national_annual.csv", "national_monthly.csv", "conus_grid.parquet",
                 "ca_fires.parquet", "fl_fires.parquet", "tubbs_record.json",
                 "reduce_report.json", "geo/us_states_20m.json"]
    for extra in ("recent_annual.csv", "recent_annual_meta.json", "dup_flags.parquet"):
        if (wf.DATA / extra).exists():
            committed.append(extra)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "wildfire-return",
        "source": {
            "zenodo_doi": "10.5281/zenodo.8381129",
            "dataset": "FPA FOD-Attributes (FPA-FOD v6 + ~270 attributes), 1992-2020, CONUS",
            "license": "CC-BY-4.0",
            "boundaries": "US Census cb_2023_us_state_20m (public domain)",
            "recent": "NIFC WFIGS Interagency Fire Perimeters aggregates, 2021-2025 (public domain)",
        },
        "files": {name: {"bytes": (wf.DATA / name).stat().st_size,
                         "sha256": wf.sha256_of(wf.DATA / name)} for name in committed},
    }
    wf.MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[done] {len(committed)} caches manifested in {wf.MANIFEST}")


def convert_geo() -> None:
    """Census state shapefile zip -> plain-JSON polylines (CONUS only, 4-dp coords)."""
    import shapefile  # pyshp

    out = wf.GEO / "us_states_20m.json"
    zip_path = wf.RAW / "geo" / "cb_2023_us_state_20m.zip"
    if not zip_path.exists():
        raise SystemExit(f"{zip_path} missing -- run fetch_raw.py")
    extract_dir = wf.RAW / "geo" / "extracted"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    shp = next(extract_dir.glob("*.shp"))
    reader = shapefile.Reader(str(shp))
    skip = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}
    states = []
    for rec, shape in zip(reader.records(), reader.shapes()):
        stusps = rec["STUSPS"]
        if stusps in skip:
            continue
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        rings = []
        for a, b in zip(parts[:-1], parts[1:]):
            rings.append([[round(x, 4), round(y, 4)] for x, y in pts[a:b]])
        states.append({"stusps": stusps, "name": rec["NAME"], "rings": rings})
    wf.GEO.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"crs": "NAD83 lon/lat (EPSG:4269)", "states": states}))
    print(f"  [geo ] {out.name}: {len(states)} CONUS states+DC, "
          f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
