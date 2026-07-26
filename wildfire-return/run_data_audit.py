from __future__ import annotations

"""The full-corpus data audit: one streaming pass, every check, no gatekeeping.

Reads all 29 raw annual CSVs (~4.68 GB) once, with ~59 columns (the curated set
plus audit-only columns like LatLong_State and SOURCE_SYSTEM), and produces:

    data/data_audit.json      every check's result + the claims the article uses
    figures/audit_*.png       7 figures
    DATA_DICTIONARY.md        auto-generated -- edit wildfire.COLUMN_UNITS, not it

The audit REPORTS; it never gates. Cleaning decisions live in the DECISIONS
block below (pre-registered rule in the plan/article): initially everything is
document-only; only deterministic fixes of objectively impossible values may
ever flip to "apply".

Requires data/raw/ (run fetch_raw.py first). ~10-15 min.
Run:  python run_data_audit.py
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import checks as ck
import wildfire as wf
from reduce_raw import annual_files, parse_discovery_dates

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}

CHUNK_ROWS = 250_000

AUDIT_EXTRA_COLUMNS = {
    "source_type": "SOURCE_SYSTEM_TYPE",
    "source": "SOURCE_SYSTEM",
    "unit_id": "NWCG_REPORTING_UNIT_ID",
    "discovery_time": "DISCOVERY_TIME",
    "cont_doy_raw": "CONT_DOY",
    "fips_code": "FIPS_CODE",
    "latlong_state": "LatLong_State",
    "latlong_county": "LatLong_County",
}
ALL_COLUMNS = {**wf.RAW_COLUMNS, **AUDIT_EXTRA_COLUMNS}
RAW_TO_CANON = {raw: canon for canon, raw in ALL_COLUMNS.items()}

# Pre-registered dispositions. The rule (also printed in data.html): a finding
# may flip to "apply" only if it is (a) a deterministic mechanical fix of an
# objectively impossible value, (b) touches <0.1% of rows or provably cannot
# change a Round-1 verdict, and (c) lands as a small printed-count reduce step.
DECISIONS: dict[str, tuple[str, str]] = {
    "unique_fod_id": ("document-only",
                      "the 64 duplicate FOD_IDs are ID COLLISIONS between different real fires "
                      "(a reused ID block: 2019 ICS-209 records vs 2020 Alaska IRWIN records) -- "
                      "drop-keep-first would delete a real fire; FOD_ID is simply not a safe "
                      "primary key for 64 pairs, and nothing in this pipeline joins on it globally"),
    "range_lat": ("document-only",
                  "the out-of-box coordinates are legitimate AK/HI/PR fires -- the dataset's "
                  "CONUS claim is about the attribute joins, not the fire records; handled as an "
                  "analysis-scope correction (see scope_ak_hi), not a data fix"),
    "range_lon": ("document-only", "same as range_lat: legitimate non-CONUS records"),
    "range_fire_size": ("document-only", "zero violations found"),
    "range_elevation": ("apply",
                        "the 47,367 'violations' are the 32767 int16-max nodata fill on non-CONUS "
                        "rows; 32767 added to wildfire.SENTINELS (defensive: provably changes zero "
                        "bytes in current caches because no CONUS row carries it)"),
    "near_duplicates": ("document-only", "pre-registered flag-only: dedup would alter trend "
                                         "inputs without adjudication; tier-1 is 548 rows of 2.3M"),
    "sentinel_scan": ("document-only", "svi's -999.0 was already in SENTINELS and masked at reduce"),
    "size_vs_class": ("document-only", "83 boundary-rounding mismatches in 2.3M; not corruption"),
    "state_vs_latlong_state": ("document-only", "99.94% agreement; border fires legitimately "
                                                "cross state lines"),
    "doy_vs_date": ("document-only", "perfect agreement: 0 disagreements in 2,302,521"),
    "cont_after_discovery": ("document-only", "zero negative burns exist; the 232 >400-day "
                                              "durations stay nulled in burn_days as designed"),
    "scope_ak_hi": ("apply",
                    "AK (15,195 fires, 36.65M acres), HI (9,970), PR (22,202) records exist in "
                    "the data; Round 1's 'CONUS' trend series silently included AK+HI. Fix is "
                    "analysis-level, not cache-level: trend lab now excludes AK/HI/PR from CONUS "
                    "series; index.html scope wording corrected. Headline ratio survives "
                    "(2.35x vs 2.37x, both round to 2.4x)"),
}


def style_ax(ax, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COL["axis"])
    ax.tick_params(colors=COL["ink2"], labelsize=9)
    ax.xaxis.label.set_color(COL["ink2"])
    ax.yaxis.label.set_color(COL["ink2"])
    ax.title.set_color(COL["ink"])


def savefig(fig, name: str) -> None:
    wf.FIG.mkdir(exist_ok=True)
    fig.savefig(wf.FIG / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig ] {name}")


def build_checks(state_agree: dict) -> dict:
    """All check instances, keyed by id. state_agree is a shared per-state accumulator."""
    miss_cols = [c for c in ALL_COLUMNS if c not in ("fod_id", "fire_year")]

    def size_vs_class_fn(df):
        size = df["fire_size"]
        edges = [0, 0.255, 9.95, 99.95, 299.95, 999.95, 4999.95, np.inf]
        labels = ["A", "B", "C", "D", "E", "F", "G"]
        expected = pd.cut(size, bins=edges, labels=labels, right=True)
        ok = size.notna() & (size > 0) & df["size_class"].notna()
        bad = ok & (expected.astype(str) != df["size_class"].astype(str))
        return ok, bad, df["size_class"].astype(str) + "|expected:" + expected.astype(str)

    def state_fn(df):
        mapped = df["latlong_state"].map(ck.STATE_NAME_TO_USPS)
        ok = mapped.notna() & df["state"].notna()
        bad = ok & (mapped != df["state"])
        for st, cnt in df.loc[ok, "state"].value_counts().items():
            state_agree.setdefault(str(st), [0, 0])[0] += int(cnt)
        for st, cnt in df.loc[bad, "state"].value_counts().items():
            state_agree.setdefault(str(st), [0, 0])[1] += int(cnt)
        return ok, bad, df["state"].astype(str) + "->" + mapped.astype(str)

    def county_fn(df):
        a = df["county"].astype(str).str.upper().str.strip()
        b = df["latlong_county"].astype(str).str.upper().str.strip()
        ok = df["county"].notna() & df["latlong_county"].notna()
        bad = ok & (a != b)
        return ok, bad, None

    def doy_fn(df):
        raw = pd.to_numeric(df["discovery_doy_raw"], errors="coerce")
        ok = raw.notna() & df["parsed_doy"].notna()
        bad = ok & (raw != df["parsed_doy"])
        return ok, bad, None

    def cont_doy_fn(df):
        raw = pd.to_numeric(df["cont_doy_raw"], errors="coerce")
        cd = df["cont_dt"].dt.dayofyear
        ok = raw.notna() & df["cont_dt"].notna()
        bad = ok & (raw != cd)
        return ok, bad, None

    def cont_order_fn(df):
        ok = df["cont_dt"].notna() & df["discovery_date"].notna()
        bad = ok & (df["cont_dt"] < df["discovery_date"])
        return ok, bad, None

    def burn_long_fn(df):
        burn = (df["cont_dt"] - df["discovery_date"]).dt.days
        ok = burn.notna()
        bad = ok & (burn > 400)
        return ok, bad, None

    return {
        "range_lat": ck.NumericRangeCheck("range_lat", "lat", 24.3, 49.5),
        "range_lon": ck.NumericRangeCheck("range_lon", "lon", -125.0, -66.5),
        "range_fire_size": ck.NumericRangeCheck("range_fire_size", "fire_size", 1e-9, 2e6),
        "range_tmmx": ck.NumericRangeCheck("range_tmmx", "tmmx", 230, 330),
        "range_rmin": ck.NumericRangeCheck("range_rmin", "rmin", 0, 100),
        "range_wind": ck.NumericRangeCheck("range_wind", "wind", 0, 40),
        "range_vpd": ck.NumericRangeCheck("range_vpd", "vpd", 0, 12),
        "range_erc": ck.NumericRangeCheck("range_erc", "erc", 0, 150),
        "range_elevation": ck.NumericRangeCheck("range_elevation", "elevation", -100, 4500),
        "range_slope": ck.NumericRangeCheck("range_slope", "slope", 0, 90),
        "range_aspect": ck.NumericRangeCheck("range_aspect", "aspect", -1, 360),
        "sentinel_scan": ck.SentinelScan(
            cols=[c for c in wf.FLOAT32_COLS if c in ALL_COLUMNS] + ["discovery_time"],
            zero_suspect_cols=["ndvi_1day", "population"]),
        "vocab_erc_pctl": ck.VocabPerYear("erc_pctl"),
        "vocab_vpd_pctl": ck.VocabPerYear("vpd_pctl"),
        "vocab_size_class": ck.VocabPerYear("size_class"),
        "vocab_cause_class": ck.VocabPerYear("cause_class"),
        "vocab_cause_general": ck.VocabPerYear("cause_general"),
        "vocab_source_type": ck.VocabPerYear("source_type"),
        "vocab_agency": ck.VocabPerYear("agency"),
        "unique_fod_id": ck.UniqueCheck("fod_id"),
        "missingness": ck.MissingnessMatrix(miss_cols),
        "size_vs_class": ck.CrossFieldRule("size_vs_class",
                                           "FIRE_SIZE consistent with FIRE_SIZE_CLASS (rounding-tolerant bins)",
                                           size_vs_class_fn, warn_at=0.005),
        "state_vs_latlong_state": ck.CrossFieldRule(
            "state_vs_latlong_state", "claimed STATE == coordinate-derived LatLong_State",
            state_fn, warn_at=0.02,
            notes="border fires legitimately fall across the line; mismatch != error"),
        "county_vs_latlong_county": ck.CrossFieldRule(
            "county_vs_latlong_county", "claimed COUNTY == coordinate-derived LatLong_County",
            county_fn, warn_at=1.0, notes="report-only; county naming is inconsistent by source"),
        "doy_vs_date": ck.CrossFieldRule(
            "doy_vs_date", "dataset DISCOVERY_DOY == day-of-year of parsed DISCOVERY_DATE",
            doy_fn, warn_at=0.001,
            notes="compares plain dayofyear, NOT our doy_std (which is deliberately leap-shifted)"),
        "cont_doy_vs_date": ck.CrossFieldRule(
            "cont_doy_vs_date", "dataset CONT_DOY == day-of-year of parsed CONT_DATE",
            cont_doy_fn, warn_at=0.001),
        "cont_after_discovery": ck.CrossFieldRule(
            "cont_after_discovery", "containment date not before discovery (negative burns)",
            cont_order_fn, warn_at=0.001,
            notes="Round 1 silently nulled these in burn_days; the audit finally counts them"),
        "burn_gt400": ck.CrossFieldRule(
            "burn_gt400", "burn duration <= 400 days", burn_long_fn, warn_at=0.001),
        "near_duplicates": ck.NearDupScan(),
    }


def main() -> None:
    files = annual_files()
    print(f"[audit] {len(files)} year files, {len(ALL_COLUMNS)} columns")
    missing_headers = []
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        missing = [raw for raw in ALL_COLUMNS.values() if raw not in header]
        if missing:
            missing_headers.append(f"{path.name}: {missing}")
    if missing_headers:
        print("AUDIT HEADER DISCOVERY FAILED:")
        print("\n".join(missing_headers))
        raise SystemExit(1)
    print(f"[audit] headers OK in all {len(files)} files")

    dtypes = {raw: "str" for raw in ALL_COLUMNS.values()}
    dtypes.update({"FOD_ID": "int64", "FIRE_YEAR": "int32", "FIRE_SIZE": "float64"})

    state_agree: dict[str, list] = {}
    CHECKS = build_checks(state_agree)
    dup: ck.NearDupScan = CHECKS["near_duplicates"]

    # runner-local accumulators
    latlong_unmapped: Counter = Counter()
    prec_lat: Counter = Counter()
    prec_lon: Counter = Counter()
    whole_deg = half_deg = coord_rows = 0
    cause_by_state: dict[str, list] = defaultdict(lambda: [0, 0, 0])  # n, class_missing, general_missing
    agency_year: dict[str, dict] = {"CA": defaultdict(lambda: defaultdict(lambda: [0, 0])),
                                    "FL": defaultdict(lambda: defaultdict(lambda: [0, 0]))}
    cont_cover: dict[int, list] = defaultdict(lambda: [0, 0])
    ranges: dict[str, list] = {}
    rows_total = 0

    for path in files:
        year = int(path.name[:4])
        for chunk in pd.read_csv(path, usecols=list(ALL_COLUMNS.values()),
                                 dtype=dtypes, chunksize=CHUNK_ROWS):
            df = chunk.rename(columns=RAW_TO_CANON)
            rows_total += len(df)
            df["discovery_date_str"] = df["discovery_date"]
            df["discovery_date"] = parse_discovery_dates(df["discovery_date"], f"{path.name}")
            df["parsed_doy"] = df["discovery_date"].dt.dayofyear
            df["cont_dt"] = pd.to_datetime(df["cont_date"], format="%m/%d/%Y %H:%M:%S",
                                           errors="coerce")
            df["lat_raw"], df["lon_raw"] = df["lat"], df["lon"]

            for check in CHECKS.values():
                check.update(df, year)

            # latlong unmapped census
            ll = df["latlong_state"]
            present = ll[ll.notna()]
            unmapped = present[~present.isin(ck.STATE_NAME_TO_USPS)]
            for k, v in unmapped.value_counts().head(20).items():
                latlong_unmapped[str(k)] += int(v)

            # coordinate precision from RAW STRINGS
            for s, ctr in ((df["lat_raw"], prec_lat), (df["lon_raw"], prec_lon)):
                present = s[s.notna()].astype(str)
                has_dot = present.str.contains(".", regex=False)
                dec = present[has_dot].str.split(".", n=1).str[1].str.len().clip(upper=7)
                for k, v in dec.value_counts().items():
                    ctr[int(k)] += int(v)
                ctr[0] += int((~has_dot).sum())
            latf = pd.to_numeric(df["lat_raw"], errors="coerce")
            coord_rows += int(latf.notna().sum())
            whole_deg += int((latf % 1 == 0).sum())
            half_deg += int((latf % 0.5 == 0).sum())

            # cause forensics
            class_missing = ~df["cause_class"].isin(["Human", "Natural"])
            general_missing = df["cause_general"].isna() | (df["cause_general"] == wf.MISSING_CAUSE)
            for st, g in df.groupby("state", dropna=False):
                acc = cause_by_state[str(st)]
                acc[0] += len(g)
                acc[1] += int(class_missing[g.index].sum())
                acc[2] += int(general_missing[g.index].sum())
            for st in ("CA", "FL"):
                sub = df[df["state"] == st]
                if len(sub):
                    gm = general_missing[sub.index]
                    for ag, g in sub.groupby("agency", dropna=False):
                        cell = agency_year[st][str(ag)][year]
                        cell[0] += len(g)
                        cell[1] += int(gm[g.index].sum())

            # containment coverage + observed ranges for the dictionary
            cc = cont_cover[year]
            cc[0] += int(df["cont_dt"].notna().sum())
            cc[1] += len(df)
            for col in ("fire_size", "tmmx", "rmin", "wind", "vpd", "erc", "elevation",
                        "slope", "aspect", "lat", "lon"):
                v = pd.to_numeric(df[col], errors="coerce")
                if v.notna().any():
                    lo, hi = float(v.min()), float(v.max())
                    if col in ranges:
                        ranges[col] = [min(ranges[col][0], lo), max(ranges[col][1], hi)]
                    else:
                        ranges[col] = [lo, hi]
        dup.year_end(year)
        print(f"  [year] {year} audited", flush=True)

    # ------------------------------------------------------------------ finalize
    finals = {cid: c.finalize() for cid, c in CHECKS.items()}
    miss = finals.pop("missingness")
    dup_final = finals.pop("near_duplicates")

    n_warn = sum(1 for f in finals.values() if f.get("status") == "warn")
    n_info = sum(1 for f in finals.values() if f.get("status") == "info")
    print(f"\n[audit] {len(finals) + 2} checks: "
          f"{sum(1 for f in finals.values() if f.get('status') == 'pass')} pass, "
          f"{n_info} info, {n_warn} warn")
    print(f"{'check':30s} {'status':6s} {'evaluated':>12s} {'violations':>10s}")
    for cid, f in finals.items():
        print(f"{cid:30s} {f.get('status', '-'):6s} {f.get('evaluated', 0) or 0:>12,} "
              f"{f.get('violations', 0) or 0:>10,}")

    state_rate = 1 - (finals["state_vs_latlong_state"]["violations"]
                      / max(finals["state_vs_latlong_state"]["evaluated"], 1))
    doy_rate = 1 - (finals["doy_vs_date"]["violations"]
                    / max(finals["doy_vs_date"]["evaluated"], 1))
    dead_cols = [c for c, s in miss["corpus"].items() if s > 0.98]
    low_prec_lat = sum(v for k, v in prec_lat.items() if k <= 2)

    claims = {
        "rows_audited": rows_total,
        "files_audited": len(files),
        "columns_audited": len(ALL_COLUMNS),
        "checks_run": len(finals) + 2,
        "findings_warn": n_warn,
        "findings_info": n_info,
        "fod_id_dup_rows": finals["unique_fod_id"]["violations"],
        "t1_dup_groups": dup_final["tier1"]["groups"],
        "t1_dup_rows": dup_final["tier1"]["rows"],
        "t2_dup_rows": dup_final["tier2"]["rows"],
        "state_agreement_rate": round(state_rate, 4),
        "doy_agreement_rate": round(doy_rate, 6),
        "negative_burns": finals["cont_after_discovery"]["violations"],
        "burn_gt400": finals["burn_gt400"]["violations"],
        "size_class_violations": finals["size_vs_class"]["violations"],
        "size_class_rate": finals["size_vs_class"]["rate"],
        "dead_columns_count": len(dead_cols),
        "ca_general_missing": cause_by_state["CA"][2] / cause_by_state["CA"][0],
        "fl_general_missing": cause_by_state["FL"][2] / cause_by_state["FL"][0],
        "ca_class_missing": cause_by_state["CA"][1] / cause_by_state["CA"][0],
        "fl_class_missing": cause_by_state["FL"][1] / cause_by_state["FL"][0],
        "coord_low_precision_share": round(low_prec_lat / max(coord_rows, 1), 4),
        "whole_degree_coords": whole_deg,
        "rows_changed_by_cleaning_v2": 0,
    }
    for k in ("ca_general_missing", "fl_general_missing", "ca_class_missing", "fl_class_missing"):
        claims[k] = round(claims[k], 4)

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {"rows": rows_total, "years": [1992, 2020], "files": len(files),
                   "columns_audited": len(ALL_COLUMNS)},
        "checks": finals,
        "missingness": miss,
        "cause_forensics": {
            "defs_note": ("Three definitions in use across this project: class-missing = "
                          "NWCG_CAUSE_CLASSIFICATION not Human/Natural; general-missing = "
                          "NWCG_GENERAL_CAUSE missing/undetermined; group-mappable = "
                          "general cause maps into the 7 ML buckets. index.html quotes all "
                          "three in different places; this file measures the first two."),
            "by_state": {st: {"n": v[0], "class_missing": round(v[1] / v[0], 4),
                              "general_missing": round(v[2] / v[0], 4)}
                         for st, v in sorted(cause_by_state.items()) if v[0] > 0},
            "ca_fl_by_agency_year": {st: {ag: {str(y): {"n": c[0], "general_missing":
                                                        round(c[1] / c[0], 4) if c[0] else None}
                                               for y, c in sorted(yrs.items())}
                                          for ag, yrs in ags.items()}
                                     for st, ags in agency_year.items()},
        },
        "duplicates": dup_final,
        "coordinates": {
            "precision_hist": {"lat": {str(k): v for k, v in sorted(prec_lat.items())},
                               "lon": {str(k): v for k, v in sorted(prec_lon.items())}},
            "whole_degree": whole_deg, "half_degree": half_deg, "coord_rows": coord_rows,
            "state_agreement": {"rate": round(state_rate, 4),
                                "by_state": {st: {"evaluated": v[0], "mismatches": v[1],
                                                  "rate": round(v[1] / v[0], 4) if v[0] else None}
                                             for st, v in sorted(state_agree.items())},
                                "unmapped_latlong_values": dict(latlong_unmapped.most_common(10))},
        },
        "containment_coverage_by_year": {str(y): round(c[0] / c[1], 4)
                                         for y, c in sorted(cont_cover.items())},
        "observed_ranges": {c: [round(v[0], 4), round(v[1], 4)] for c, v in ranges.items()},
        "dead_columns": dead_cols,
        "cleaning_v2": {
            "rule": ("apply iff: deterministic mechanical fix of an objectively impossible value; "
                     "touches <0.1% of rows or provably cannot change a Round-1 verdict; "
                     "implementable as a small printed-count reduce step"),
            "decided": True,
            "applied": [k for k, (d, _) in DECISIONS.items() if d == "apply"],
            "documented_only": [{"id": k, "rationale": r}
                                for k, (d, r) in DECISIONS.items() if d == "document-only"],
        },
        "claims": claims,
    }
    (wf.DATA / "data_audit.json").write_text(json.dumps(audit, indent=1))
    print(f"[json] data_audit.json ({(wf.DATA / 'data_audit.json').stat().st_size / 1024:.0f} KB)")

    # dup_flags sidecar: modeling joins on fod_id for sensitivity checks.
    # NOTE: 64 FOD_IDs are collisions (see unique_fod_id), so a flag can touch
    # an innocent twin -- acceptable for a flag, fatal for a drop.
    t1, t2 = set(dup.t1_ids), set(dup.t2_ids)
    flags = pd.DataFrame({"fod_id": sorted(t1 | t2)})
    flags["tier1"] = flags["fod_id"].isin(t1)
    flags["tier2"] = flags["fod_id"].isin(t2)
    flags.to_parquet(wf.DATA / "dup_flags.parquet", index=False)
    print(f"[side] dup_flags.parquet: {len(flags):,} flagged fod_ids "
          f"({int(flags['tier1'].sum()):,} tier-1)")

    make_figures(audit)
    write_dictionary(audit)
    print("[done] data audit complete")


# ---------------------------------------------------------------------------- figures
def make_figures(audit: dict) -> None:
    miss = audit["missingness"]
    years = sorted(miss["by_year"])
    cols = [c for c in miss["columns"] if c not in ("discovery_date", "cont_date",
                                                    "discovery_date_str", "fire_name")]

    # 1. missingness x years
    order = sorted(cols, key=lambda c: miss["corpus"][c], reverse=True)
    M = np.array([[miss["by_year"][y][c] for y in years] for c in order])
    fig, ax = plt.subplots(figsize=(9.6, 0.26 * len(order) + 1.6))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xticks(range(0, len(years), 4))
    ax.set_xticklabels(years[::4], fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("missing share", fontsize=8, color=COL["ink2"])
    cb.ax.tick_params(labelsize=7, colors=COL["ink2"])
    ax.set_title("Attributes are joins with coverage eras: missingness by column and year",
                 loc="left", fontsize=11)
    style_ax(ax, grid_axis="")
    savefig(fig, "audit_missingness_eras.png")

    # 2. missingness x states (12 most geographically variable columns)
    states = [s for s in sorted(miss["by_state"]) if s not in ("nan",)]
    var = {c: np.std([miss["by_state"][s][c] for s in states]) for c in cols}
    geo_cols = sorted(var, key=var.get, reverse=True)[:12]
    G = np.array([[miss["by_state"][s][c] for s in states] for c in geo_cols])
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    im = ax.imshow(G, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(geo_cols)))
    ax.set_yticklabels(geo_cols, fontsize=7.5)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=5.6, rotation=90)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("missing share", fontsize=8, color=COL["ink2"])
    cb.ax.tick_params(labelsize=7, colors=COL["ink2"])
    ax.set_title("Missingness has geography: the 12 most state-variable columns",
                 loc="left", fontsize=11)
    style_ax(ax, grid_axis="")
    savefig(fig, "audit_missingness_geo.png")

    # 3. cause forensics (centerpiece)
    cf = audit["cause_forensics"]["by_state"]
    sts = [s for s in cf if cf[s]["n"] >= 5000 and s != "nan"]
    sts = sorted(sts, key=lambda s: cf[s]["general_missing"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.4), width_ratios=[1.15, 1])
    y = np.arange(len(sts))
    axes[0].barh(y + 0.2, [cf[s]["general_missing"] * 100 for s in sts], height=0.4,
                 color=COL["fire"], label="general cause missing")
    axes[0].barh(y - 0.2, [cf[s]["class_missing"] * 100 for s in sts], height=0.4,
                 color=COL["blue"], label="cause class missing")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sts, fontsize=5.6)
    for lab in axes[0].get_yticklabels():
        if lab.get_text() in ("CA", "FL"):
            lab.set_color(COL["fire"])
            lab.set_fontweight("bold")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("share of fires (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=COL["ink2"])
    axes[0].set_title("Who knows WHAT about cause (states with 5k+ fires)", loc="left", fontsize=10)
    style_ax(axes[0], grid_axis="x")
    ay = audit["cause_forensics"]["ca_fl_by_agency_year"]
    for st, ls in (("CA", "-"), ("FL", "--")):
        tops = sorted(ay[st], key=lambda a: sum(c["n"] for c in ay[st][a].values()),
                      reverse=True)[:3]
        for ag, color in zip(tops, (COL["fire"], COL["gold"], COL["blue"])):
            yrs = sorted(int(y) for y in ay[st][ag])
            vals = [ay[st][ag][str(y)]["general_missing"] for y in yrs]
            vals = [v * 100 if v is not None else np.nan for v in vals]
            axes[1].plot(yrs, vals, ls=ls, color=color, lw=1.5,
                         label=f"{st} - {ag}"[:26])
    axes[1].set_ylabel("general cause missing (%)")
    axes[1].legend(frameon=False, fontsize=7, labelcolor=COL["ink2"])
    axes[1].set_title("CA vs FL, top agencies by year", loc="left", fontsize=10)
    style_ax(axes[1])
    fig.tight_layout()
    savefig(fig, "audit_cause_forensics.png")

    # 4. duplicates
    dup = audit["duplicates"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    y1 = {int(k): v for k, v in dup["tier1"]["by_year"].items()}
    y2 = {int(k): v for k, v in dup["tier2"]["by_year"].items()}
    yrs = sorted(set(y1) | set(y2))
    axes[0].semilogy([y for y in yrs], [max(y2.get(y, 0), 0.5) for y in yrs],
                     color=COL["muted"], lw=1.5, label="tier 2 rows (ceiling: batch entries)")
    axes[0].semilogy([y for y in yrs], [max(y1.get(y, 0), 0.5) for y in yrs],
                     color=COL["fire"], lw=1.8, label="tier 1 rows (>=10 ac, 10% size tol)")
    axes[0].set_ylabel("candidate rows / yr (log)")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=COL["ink2"])
    axes[0].set_title("Near-duplicate candidates by year", loc="left", fontsize=10)
    style_ax(axes[0])
    pairs = dup["tier1"]["by_system_pair"]
    if pairs:
        names = list(pairs)[:10][::-1]
        colors = [COL["red"] if n.startswith("cross") else COL["gold"] for n in names]
        axes[1].barh([n.replace(": ", ":\n") for n in names], [pairs[n] for n in names],
                     color=colors)
        axes[1].set_xlabel("tier-1 groups")
        axes[1].tick_params(axis="y", labelsize=6.4)
    axes[1].set_title("Cross-system (red) is the merge residue", loc="left", fontsize=10)
    style_ax(axes[1], grid_axis="x")
    fig.tight_layout()
    savefig(fig, "audit_duplicates.png")

    # 5. coordinates
    co = audit["coordinates"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ks = sorted(int(k) for k in co["precision_hist"]["lat"])
    vals = [co["precision_hist"]["lat"][str(k)] for k in ks]
    colors = [COL["red"] if k <= 2 else COL["blue"] for k in ks]
    axes[0].bar([str(k) if k < 7 else "7+" for k in ks], np.array(vals) / 1e6, color=colors)
    axes[0].set_xlabel("decimal places in raw LATITUDE string")
    axes[0].set_ylabel("fires (millions)")
    axes[0].set_title("Coordinate precision: red bars are centroid country", loc="left", fontsize=10)
    style_ax(axes[0])
    bs = co["state_agreement"]["by_state"]
    top = sorted((s for s in bs if bs[s]["rate"] is not None and bs[s]["evaluated"] > 2000),
                 key=lambda s: bs[s]["rate"], reverse=True)[:15]
    axes[1].barh(top[::-1], [bs[s]["rate"] * 100 for s in top][::-1], color=COL["gold"])
    axes[1].set_xlabel("STATE != LatLong_State (%)")
    axes[1].set_title("Coordinate-vs-claimed-state disagreement, top 15", loc="left", fontsize=10)
    style_ax(axes[1], grid_axis="x")
    fig.tight_layout()
    savefig(fig, "audit_coordinates.png")

    # 6. time integrity
    doy = audit["checks"]["doy_vs_date"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    yrs = sorted(int(y) for y in doy["by_year_eval"])
    rates = [doy["by_year"].get(str(y), 0) / doy["by_year_eval"][str(y)] * 100 for y in yrs]
    axes[0].plot(yrs, rates, color=COL["fire"], lw=1.7, marker="o", ms=3)
    for leap in (1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020):
        axes[0].axvline(leap, color=COL["grid"], lw=0.6, zorder=0)
    axes[0].set_ylabel("DISCOVERY_DOY disagreement (%)")
    axes[0].set_title("The promised DOY cross-check (leap years gridded)", loc="left", fontsize=10)
    style_ax(axes[0])
    neg = audit["checks"]["cont_after_discovery"]
    yrs2 = sorted(int(y) for y in neg["by_year_eval"])
    axes[1].bar(yrs2, [neg["by_year"].get(str(y), 0) for y in yrs2], color=COL["red"],
                label="negative burns")
    ax2 = axes[1].twinx()
    cov = audit["containment_coverage_by_year"]
    ax2.plot(yrs2, [cov[str(y)] * 100 for y in yrs2], color=COL["blue"], lw=1.5,
             label="containment coverage")
    ax2.set_ylabel("containment coverage (%)", fontsize=8, color=COL["blue"])
    ax2.tick_params(labelsize=8, colors=COL["blue"])
    axes[1].set_ylabel("negative-burn rows / yr")
    axes[1].set_title("What Round 1 silently nulled, finally counted", loc="left", fontsize=10)
    style_ax(axes[1])
    fig.tight_layout()
    savefig(fig, "audit_time_integrity.png")

    # 7. vocab + size-class
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    vocab = audit["checks"]["vocab_erc_pctl"]
    labels = vocab["labels_all"]
    yrs3 = list(range(1992, 2021))
    P = np.array([[1 if vocab["label_year_span"][lab][0] <= y <= vocab["label_year_span"][lab][1]
                   else 0 for y in yrs3] for lab in labels])
    axes[0].imshow(P, aspect="auto", cmap="Oranges", vmin=0, vmax=1.4)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xticks(range(0, len(yrs3), 4))
    axes[0].set_xticklabels(yrs3[::4], fontsize=8)
    axes[0].set_title(f"erc_pctl vocabulary by year (stable: {vocab['stable']})",
                      loc="left", fontsize=10)
    style_ax(axes[0], grid_axis="")
    svc = audit["checks"]["size_vs_class"]
    det = svc.get("detail_top", {})
    if det:
        names = list(det)[:10][::-1]
        axes[1].barh(names, [det[n] for n in names], color=COL["gold"])
        axes[1].tick_params(axis="y", labelsize=6.4)
    axes[1].set_xlabel(f"violations (total {svc['violations']:,} of {svc['evaluated']:,})")
    axes[1].set_title("FIRE_SIZE vs FIRE_SIZE_CLASS mismatches", loc="left", fontsize=10)
    style_ax(axes[1], grid_axis="x")
    fig.tight_layout()
    savefig(fig, "audit_vocab_sizeclass.png")


# ---------------------------------------------------------------------------- dictionary
def write_dictionary(audit: dict) -> None:
    miss = audit["missingness"]
    ranges = audit["observed_ranges"]
    vocab_cols = {"erc_pctl": "vocab_erc_pctl", "vpd_pctl": "vocab_vpd_pctl",
                  "size_class": "vocab_size_class", "cause_class": "vocab_cause_class",
                  "cause_general": "vocab_cause_general", "source_type": "vocab_source_type"}

    def row(canon: str, raw: str) -> str:
        units = wf.COLUMN_UNITS.get(canon, "-")
        corpus = miss["corpus"].get(canon)
        ca = miss["by_state"].get("CA", {}).get(canon)
        fl = miss["by_state"].get("FL", {}).get(canon)
        if canon in ranges:
            obs = f"[{ranges[canon][0]:g}, {ranges[canon][1]:g}]"
        elif canon in vocab_cols:
            labs = audit["checks"][vocab_cols[canon]]["labels_all"]
            obs = ", ".join(labs[:6]) + ("..." if len(labs) > 6 else "")
        else:
            obs = "-"
        notes = wf.COLUMN_NOTES.get(canon, "")
        if canon in audit["dead_columns"]:
            notes = ("DEAD (>98% missing corpus-wide). " + notes).strip()
        fmt = lambda v: f"{100 * v:.1f}%" if v is not None else "-"
        return (f"| `{canon}` | `{raw}` | {units} | {fmt(corpus)} | {fmt(ca)} | {fmt(fl)} "
                f"| {obs} | {notes} |")

    lines = [
        "# Data Dictionary -- wildfire-return",
        "",
        "**AUTO-GENERATED by `run_data_audit.py`** -- edit `wildfire.COLUMN_UNITS` /",
        "`wildfire.COLUMN_NOTES` / `checks.py`, never this file.",
        "",
        f"Generated {audit['generated_utc']} from {audit['corpus']['rows']:,} rows,",
        f"{audit['corpus']['files']} annual files, {audit['corpus']['columns_audited']} audited columns.",
        "Source: FPA FOD-Attributes (Zenodo 10.5281/zenodo.8381129, CC-BY 4.0).",
        "",
        "## Curated columns (cached by reduce_raw.py)",
        "",
        "| canonical | raw column | units | miss (corpus) | miss (CA) | miss (FL) | observed | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for canon, raw in wf.RAW_COLUMNS.items():
        lines.append(row(canon, raw))
    lines += [
        "",
        "## Derived columns (computed in reduce_raw.py)",
        "",
        "| column | derivation |",
        "|---|---|",
        "| `month`, `dow` | from properly parsed DISCOVERY_DATE (the 2020 bug made these constants) |",
        f"| `doy_std` | {wf.COLUMN_NOTES['doy_std']} |",
        f"| `burn_days` | CONT_DATE - DISCOVERY_DATE in days; {wf.COLUMN_NOTES['burn_days']} |",
        f"| `cause_group` | NWCG general cause -> 7 ML buckets; {wf.COLUMN_NOTES['cause_group']} |",
        f"| `old13` | {wf.COLUMN_NOTES['old13']} |",
        "",
        "## Audit-only columns (streamed by run_data_audit.py, not cached)",
        "",
        "| canonical | raw column |",
        "|---|---|",
    ]
    for canon, raw in AUDIT_EXTRA_COLUMNS.items():
        lines.append(f"| `{canon}` | `{raw}` |")
    lines += ["", "## Cache files", "",
              "| file | rows | columns | producer |", "|---|---|---|---|"]
    for name, producer in (("national_annual.csv", "reduce_raw.py"),
                           ("national_monthly.csv", "reduce_raw.py"),
                           ("conus_grid.parquet", "reduce_raw.py"),
                           ("ca_fires.parquet", "reduce_raw.py"),
                           ("fl_fires.parquet", "reduce_raw.py"),
                           ("recent_annual.csv", "fetch_recent.py"),
                           ("dup_flags.parquet", "run_data_audit.py")):
        p = wf.DATA / name
        if p.exists():
            df = pd.read_parquet(p) if name.endswith("parquet") else pd.read_csv(p)
            lines.append(f"| `{name}` | {len(df):,} | {df.shape[1]} | {producer} |")
    lines += ["",
              "Sentinels handled at reduce time: " + ", ".join(str(s) for s in wf.SENTINELS) + ".",
              "Additional sentinel census: see `sentinel_scan` in `data/data_audit.json`.", ""]
    (wf.ROOT / "DATA_DICTIONARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("  [dict] DATA_DICTIONARY.md regenerated")


if __name__ == "__main__":
    main()
