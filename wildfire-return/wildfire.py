from __future__ import annotations

"""wildfire.py -- the single source of truth for Return to Fire.

Owns: paths, the curated column map (canonical name -> exact raw CSV column),
cause taxonomies (NWCG buckets for the honest models, NWCG -> old-13 crosswalk
for the 2020 museum replica), region/size constants, leap-safe day-of-year, and
the cache loaders (which verify the SHA-256 manifest so silent drift fails loud).

Every other script imports from here. Nothing here touches the network.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
GEO = DATA / "geo"
FIG = ROOT / "figures"
MANIFEST = DATA / "MANIFEST.json"

# ---------------------------------------------------------------------------
# Raw column map: canonical name -> exact column in YYYY_FPA_FOD_cons.csv
# (verified against the real 2017 header, 308 columns; reduce_raw.py asserts
# every one of these exists in every year file and fails loudly otherwise).
# ---------------------------------------------------------------------------
RAW_COLUMNS: dict[str, str] = {
    # identity + label targets
    "fod_id": "FOD_ID",
    "fpa_id": "FPA_ID",
    "fire_name": "FIRE_NAME",
    "fire_year": "FIRE_YEAR",
    "discovery_date": "DISCOVERY_DATE",
    "discovery_doy_raw": "DISCOVERY_DOY",
    "cause_class": "NWCG_CAUSE_CLASSIFICATION",      # Human / Natural / Missing...
    "cause_general": "NWCG_GENERAL_CAUSE",           # 13 NWCG general causes
    "cont_date": "CONT_DATE",
    "fire_size": "FIRE_SIZE",
    "size_class": "FIRE_SIZE_CLASS",
    "lat": "LATITUDE",
    "lon": "LONGITUDE",
    "state": "STATE",
    "county": "COUNTY",
    "agency": "NWCG_REPORTING_AGENCY",               # for reporting-coverage EDA
    "owner": "OWNER_DESCR",
    # gridMET weather at ignition (day-of), Kelvin/m s-1/kPa/unitless
    "tmmx": "tmmx",
    "rmin": "rmin",
    "wind": "vs",
    "precip": "pr",
    "vpd": "vpd",
    "erc": "erc",
    "bi": "bi",
    "fm100": "fm100",
    "fm1000": "fm1000",
    # short antecedent window
    "precip_5d_mean": "pr_5D_mean",
    "vpd_5d_max": "vpd_5D_max",
    "erc_5d_max": "erc_5D_max",
    "wind_5d_max": "vs_5D_max",
    # local climatological context
    "erc_pctl": "erc_Percentile",
    "vpd_pctl": "vpd_Percentile",
    "vpd_normal": "vpd_Normal",
    "erc_normal": "erc_Normal",
    # vegetation / fuels
    "ndvi_1day": "NDVI-1day",
    "ndvi_12m": "MOD_NDVI_12m",
    "land_cover": "Land_Cover",
    "frg": "FRG",                                     # fire regime group
    # topography
    "elevation": "Elevation",
    "slope": "Slope",
    "aspect": "Aspect",
    "tpi": "TPI",
    # human / social
    "population": "Population",
    "ghm": "GHM",                                     # global human modification
    "road_county_dis": "road_county_dis",
    "road_interstate_dis": "road_interstate_dis",
    "firestations_10km": "No_FireStation_10.0km",
    "svi": "RPL_THEMES",                              # overall SVI percentile
    # ecological context
    "ecoregion_l3": "Ecoregion_US_L3CODE",
    "aridity": "Aridity_index",
    "gacc": "GACCAbbrev",
    # ---------------- cache v2 (Round 3, the modeling stage) ----------------
    # Wind as a VECTOR: direction unlocks wind-terrain alignment, the physical
    # mechanism behind the fires that reach towns (Diablo/Santa Ana downslope).
    "wind_dir": "th",
    "wind_dir_5d_max": "th_5D_max",
    # Climatological normals -> place-relative anomaly features ("how unusual
    # is today HERE"), which is what makes one model work across regions.
    "fm100_normal": "fm100_Normal",
    "fm1000_normal": "fm1000_Normal",
    "bi_normal": "bi_Normal",
    "tmmx_normal": "tmmx_Normal",
    "rmin_normal": "rmin_Normal",
    "precip_normal": "pr_Normal",
    # Same-day weather completing the energy balance (enables Fosberg FFWI).
    "tmmn": "tmmn",
    "rmax": "rmax",
    "sph": "sph",
    "srad": "srad",
    "etr": "etr",
    # Fine fuel load -- what actually carries a wind-driven run. Cheatgrass and
    # exotic annual grasses are the documented driver of Great Basin fire.
    "rpms": "rpms",
    "cheatgrass": "CheatGrass",
    "exotic_grass": "ExoticAnnualGrass",
    # Vegetation structure (LANDFIRE): canopy cover, height, type.
    "evc": "EVC",
    "evh": "EVH",
    "evt": "EVT",
    # Terrain ruggedness -- complements slope/aspect/TPI.
    "tri": "TRI",
    # Long-run climate context for the place (not the day).
    "annual_precip": "Annual_precipitation",
    "annual_temp": "Annual_tempreture",   # sic: misspelled in the source
    "sdi": "SDI",
    # Ordinal weather percentiles joining the existing erc/vpd pair.
    "tmmx_pctl": "tmmx_Percentile",
    "wind_pctl": "vs_Percentile",
    "fm100_pctl": "fm100_Percentile",
    "bi_pctl": "bi_Percentile",
    # Cached ONLY to be banned + demonstrated as leaks (see ML_EXCLUDED_LEAKS):
    # an MTBS mapping exists only for large fires, an ICS-209 only for incidents
    # that drew a management team. Their PRESENCE encodes the outcome.
    "mtbs_id": "MTBS_ID",
    "ics209_id": "ICS_209_PLUS_INCIDENT_JOIN_ID",
}

FLOAT32_COLS = [
    "tmmx", "rmin", "wind", "precip", "vpd", "erc", "bi", "fm100", "fm1000",
    "precip_5d_mean", "vpd_5d_max", "erc_5d_max", "wind_5d_max",
    "vpd_normal", "erc_normal", "ndvi_1day", "ndvi_12m",
    "elevation", "slope", "aspect", "tpi",
    "population", "ghm", "road_county_dis", "road_interstate_dis", "svi", "aridity",
    # cache v2 numerics
    "wind_dir", "wind_dir_5d_max",
    "fm100_normal", "fm1000_normal", "bi_normal", "tmmx_normal", "rmin_normal",
    "precip_normal", "tmmn", "rmax", "sph", "srad", "etr",
    "rpms", "cheatgrass", "exotic_grass", "evc", "evh", "tri",
    "annual_precip", "annual_temp", "sdi",
]
CATEGORY_COLS = ["cause_class", "cause_general", "size_class", "state", "agency",
                 "owner", "land_cover", "frg", "ecoregion_l3", "gacc",
                 "erc_pctl", "vpd_pctl",
                 # cache v2 categoricals
                 "evt", "tmmx_pctl", "wind_pctl", "fm100_pctl", "bi_pctl"]

# 32767 is the int16-max nodata fill the audit found in the topo columns of
# non-CONUS rows (AK/HI/PR); it never appears in CONUS rows, so adding it here
# is defensive and provably changes zero bytes in the current caches.
SENTINELS = [-9999, -999, -9999.0, -999.0, 32767, 32767.0]
# Overflow-style nodata: ghm and sdi carry float32-max (3.4e38) for missing.
# Anything at or beyond this magnitude is a fill, not a measurement.
OVERFLOW_FILL = 1e30

# ---------------------------------------------------------------------------
# Geography and size classes
# ---------------------------------------------------------------------------
WEST_STATES = ["WA", "OR", "CA", "ID", "NV", "AZ", "UT", "MT", "WY", "CO", "NM"]
NON_CONUS = ["AK", "HI", "PR"]          # absent from FPA FOD-Attributes anyway

# NWCG size classes (acres): upper bound inclusive labels for figures
SIZE_CLASS_ACRES = {
    "A": (0.0, 0.25), "B": (0.26, 9.9), "C": (10.0, 99.9), "D": (100.0, 299.9),
    "E": (300.0, 999.9), "F": (1000.0, 4999.9), "G": (5000.0, float("inf")),
}
LARGE_CLASSES = ["F", "G"]              # >= 1,000 acres: the reporting-robust stratum

# ---------------------------------------------------------------------------
# Cause taxonomies
# ---------------------------------------------------------------------------
MISSING_CAUSE = "Missing data/not specified/undetermined"

# NWCG_GENERAL_CAUSE -> 7 buckets for the honest models
CAUSE_GROUPS: dict[str, str] = {
    "Natural": "Natural",
    "Arson/incendiarism": "Arson",
    "Debris and open burning": "Debris burning",
    "Equipment and vehicle use": "Equipment & vehicle",
    "Recreation and ceremony": "Recreation",
    "Power generation/transmission/distribution": "Infrastructure",
    "Railroad operations and maintenance": "Infrastructure",
    "Smoking": "Other human",
    "Fireworks": "Other human",
    "Firearms and explosives use": "Other human",
    "Misuse of fire by a minor": "Other human",
    "Other causes": "Other human",
    # MISSING_CAUSE deliberately absent -> NaN -> excluded from cause models
}

# NWCG_GENERAL_CAUSE -> the 13 STAT_CAUSE_DESCR classes of the 2015-era dataset.
# APPROXIMATE by construction (v6 retired the old taxonomy); used ONLY to rebuild
# the 2020 museum replica. 'Structure' has no NWCG source and never appears.
NWCG_TO_OLD13: dict[str, str] = {
    "Natural": "Lightning",
    "Equipment and vehicle use": "Equipment Use",
    "Smoking": "Smoking",
    "Recreation and ceremony": "Campfire",
    "Debris and open burning": "Debris Burning",
    "Railroad operations and maintenance": "Railroad",
    "Arson/incendiarism": "Arson",
    "Misuse of fire by a minor": "Children",
    "Other causes": "Miscellaneous",
    "Fireworks": "Fireworks",
    "Firearms and explosives use": "Miscellaneous",
    "Power generation/transmission/distribution": "Powerline",
    MISSING_CAUSE: "Missing/Undefined",
}

# ---------------------------------------------------------------------------
# Units + notes registry (feeds the auto-generated DATA_DICTIONARY.md).
# UNVERIFIED means: inferred from source conventions, not yet confirmed against
# the ESSD paper -- the dictionary flags these honestly instead of guessing.
# ---------------------------------------------------------------------------
KELVIN_COLS = {"tmmx"}
COLUMN_UNITS: dict[str, str] = {
    "fire_size": "acres",
    "lat": "decimal degrees (NAD83)", "lon": "decimal degrees (NAD83)",
    "tmmx": "(!) KELVIN -- subtract 273.15 for C; see examples/find_tubbs.py k_to_f()",
    "rmin": "%", "wind": "m/s (gridMET 'vs', daily mean)", "precip": "mm",
    "vpd": "kPa", "erc": "index (unitless)", "bi": "index (unitless)",
    "fm100": "% moisture", "fm1000": "% moisture",
    "precip_5d_mean": "mm", "vpd_5d_max": "kPa", "erc_5d_max": "index",
    "wind_5d_max": "m/s", "vpd_normal": "kPa", "erc_normal": "index",
    "erc_pctl": "local climatological percentile bin", "vpd_pctl": "local climatological percentile bin",
    "ndvi_1day": "unitless (MODIS-scaled)", "ndvi_12m": "unitless (MODIS-scaled)",
    "elevation": "m", "slope": "degrees", "aspect": "degrees (-1 = flat)",
    "tpi": "m (relative)", "population": "UNVERIFIED (inferred: persons per grid cell)",
    "ghm": "0-1 index", "svi": "0-1 percentile",
    "road_county_dis": "UNVERIFIED (inferred: distance, m)",
    "road_interstate_dis": "UNVERIFIED (inferred: distance, m)",
    "firestations_10km": "count within 10 km",
    "aridity": "index (unitless)", "burn_days": "days",
    "discovery_doy_raw": "day of year (dataset's own)",
}
COLUMN_NOTES: dict[str, str] = {
    "tmmx": "gridMET daily max temperature at the ignition cell",
    "wind": "daily-mean wind; cannot represent gust events (see article limits)",
    "aspect": "sentinel -1 appears to mean flat terrain",
    "doy_std": "leap-safe: post-Feb-28 days shift down 1 in leap years so Mar 1 is always 60",
    "burn_days": "nulled when negative or > 400 days (audited in data_audit.json)",
    "cause_group": "NaN when NWCG_GENERAL_CAUSE is the missing/undetermined label",
    "old13": "APPROXIMATE crosswalk to the retired 13-class taxonomy; museum replica only",
}

# The Tubbs Fire, verified in the raw 2017 file (reduce_raw re-verifies).
TUBBS_FOD_ID = 400015986

ERA_SPLIT = 2006          # grid maps: 1992-2005 vs 2006-2020
ML_TRAIN_MAX_YEAR = 2014  # temporal holdout: train <=2014, val 2015-17, test 2018-20
ML_VAL_YEARS = (2015, 2017)
ML_TEST_YEARS = (2018, 2020)

# ---------------------------------------------------------------------------
# The modeling feature contract (the data stage's formal handoff).
# At-ignition only. ndvi_12m and both road distances are dead in CA/FL
# (eda_results.json); everything in ML_EXCLUDED_LEAKS is knowable only after
# ignition and is banned from cause/growth models by name.
# ---------------------------------------------------------------------------
ML_FEATURES_NUM = ["lat", "lon", "month", "doy_std", "dow",
                   "tmmx", "rmin", "wind", "precip", "vpd", "erc", "bi", "fm100", "fm1000",
                   "precip_5d_mean", "vpd_5d_max", "erc_5d_max", "wind_5d_max",
                   "vpd_normal", "erc_normal", "ndvi_1day",
                   "elevation", "slope", "aspect", "tpi",
                   "population", "ghm", "firestations_10km", "svi", "aridity"]
ML_FEATURES_CAT = ["land_cover", "frg", "ecoregion_l3"]
# Banned by name. The first five are obvious post-outcome columns. The last three
# were added in Round 3 after the leak probe showed that an *identifier* can be a
# near-deterministic function of the outcome: MTBS maps only fires >=1,000 ac in
# the West, and an ICS-209 is filed only for incidents that drew a management
# team -- so merely HAVING the id tells you the fire got big. Neither is named
# "fire_size"; both would survive a naive "drop the obvious targets" pass.
ML_EXCLUDED_LEAKS = ["fire_size", "size_class", "burn_days", "agency", "owner",
                     "cont_date", "mtbs_id", "ics209_id"]


def doy_std(dates: pd.Series) -> pd.Series:
    """Leap-safe day-of-year: in leap years, days after Feb 28 shift down 1 so
    March 1 is 60 in every year (the 2020 notebook never got a working DOY at
    all -- its dates all parsed to 1970-01-01)."""
    doy = dates.dt.dayofyear
    return (doy - (dates.dt.is_leap_year & (doy > 59)).astype(int)).astype("int16")


# ---------------------------------------------------------------------------
# Manifest + loaders
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(*names: str, strict: bool = False) -> None:
    """Check listed cache files against MANIFEST.json. Runners pass strict=True."""
    if not MANIFEST.exists():
        msg = f"{MANIFEST} missing -- run reduce_raw.py first"
        if strict:
            raise SystemExit(msg)
        print(f"[manifest] WARNING: {msg}")
        return
    entries = json.loads(MANIFEST.read_text())["files"]
    for name in names:
        path = DATA / name
        if name not in entries:
            raise SystemExit(f"[manifest] {name} not in MANIFEST.json -- re-run reduce_raw.py")
        if not path.exists():
            raise SystemExit(f"[manifest] {path} missing -- re-run reduce_raw.py")
        actual = sha256_of(path)
        if actual != entries[name]["sha256"]:
            msg = (f"[manifest] {name} hash mismatch (expected {entries[name]['sha256'][:12]}..., "
                   f"got {actual[:12]}...) -- cache drifted; re-run reduce_raw.py")
            if strict:
                raise SystemExit(msg)
            print("WARNING: " + msg)


def load_national(strict: bool = False) -> pd.DataFrame:
    verify_manifest("national_annual.csv", strict=strict)
    return pd.read_csv(DATA / "national_annual.csv")


def load_monthly(strict: bool = False) -> pd.DataFrame:
    verify_manifest("national_monthly.csv", strict=strict)
    return pd.read_csv(DATA / "national_monthly.csv")


def load_grid(strict: bool = False) -> pd.DataFrame:
    verify_manifest("conus_grid.parquet", strict=strict)
    return pd.read_parquet(DATA / "conus_grid.parquet")


def load_state_fires(state: str, strict: bool = False) -> pd.DataFrame:
    name = f"{state.lower()}_fires.parquet"
    verify_manifest(name, strict=strict)
    df = pd.read_parquet(DATA / name)
    df["discovery_date"] = pd.to_datetime(df["discovery_date"])
    return df


def load_tubbs(strict: bool = False) -> dict:
    verify_manifest("tubbs_record.json", strict=strict)
    return json.loads((DATA / "tubbs_record.json").read_text())


def load_recent(strict: bool = False) -> pd.DataFrame:
    verify_manifest("recent_annual.csv", strict=strict)
    return pd.read_csv(DATA / "recent_annual.csv")


def load_states_geo() -> dict:
    path = GEO / "us_states_20m.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run reduce_raw.py (converts the Census shapefile)")
    return json.loads(path.read_text())
