#!/usr/bin/env python3
"""
build_panel.py
==============
Build a model-ready country-year panel for econometric work on
pollution / economy / energy / AI / well-being, from OECD (SDMX) and
World Bank (WDI) sources.

Design
------
Everything is fetched into a *long* table keyed on (iso3, year, indicator)
and then pivoted to a *wide* panel: one row per country-year, one column
per indicator. Adding an indicator = adding one dict entry below.

Two backends
------------
1. World Bank (WDI)  -> works out of the box; indicator codes are stable.
                        Good coverage for economy, energy mix, emissions.
2. OECD (SDMX)       -> paste the URL from the Data Explorer "Developer API"
                        button. OECD dataflow IDs are versioned and change,
                        so this script does NOT hardcode them for you.
                        The parsing/merge machinery is identical for both.

Usage
-----
    pip install requests pandas
    python build_panel.py

Outputs (in ./output):
    panel_long.csv        tidy long form (iso3, year, indicator, value, source)
    panel_wide.csv        wide panel of raw indicator levels (iso3, year, <indicators...>)
    panel_wide.parquet    (only if pyarrow is installed)
    panel_model_ready.csv panel_wide.csv plus engineered columns (logs, growth
                           rates, lags, carbon intensity, trend) for regression use
    panel_model_ready.parquet (only if pyarrow is installed)
    coverage.csv          per-indicator coverage summary
    availability.csv      country x indicator non-missing counts

Raw API responses are cached under ./cache so re-runs don't re-hit the APIs.
"""

from __future__ import annotations

import io
import os
import re
import time
import json
import hashlib
from pathlib import Path

import requests
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG                                                                       #
# --------------------------------------------------------------------------- #

OUTPUT_DIR = Path("output")
CACHE_DIR = Path("cache")
START_YEAR = 1995            # earliest year to keep
END_YEAR = 2024              # latest year to keep

# Country scope.
#   None  -> WORLDWIDE: keep every real country the World Bank recognises,
#            using WB country metadata to strip out regional/income aggregates.
#   list  -> restrict to these ISO3 codes (e.g. the OECD-38 set, kept below
#            for convenience if you ever want to narrow the scope again).
COUNTRIES = None

OECD_38 = [
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT",
    "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
]

# SDMX/OECD aggregate codes to drop if COUNTRIES is None.
AGGREGATE_CODES = {
    "OECD", "OECDE", "OECDAM", "EU27_2020", "EU28", "EU27", "EU",
    "EA", "EA19", "EA20", "G7", "G20", "W", "WXOECD", "WORLD", "_T",
}

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_CALLS = 1.0     # be polite; OECD API is rate-limited
MAX_RETRIES = 3

# --------------------------------------------------------------------------- #
# INDICATOR DEFINITIONS                                                        #
# --------------------------------------------------------------------------- #
# Each entry: internal column name -> spec.
#
# World Bank spec:  {"source": "wb", "code": "<WDI code>", "desc": "..."}
# OECD spec:        {"source": "oecd", "url": "<full SDMX data URL>", "desc": "..."}
#
# For OECD: go to https://data-explorer.oecd.org, pick your indicator and
# filters, click the "Developer API" icon, Copy code, and paste the URL here.
# Keep the '?...format=csvfilewithlabels' style query string. If the copied
# URL has no format param, the fetcher appends a sensible one.

INDICATORS: dict[str, dict] = {
    # ---- ECONOMY --------------------------------------------------------- #
    "gdp_pc_ppp_const": {
        "source": "wb", "code": "NY.GDP.PCAP.PP.KD",
        "desc": "GDP per capita, PPP (constant 2021 intl $)",
    },
    "gdp_growth": {
        "source": "wb", "code": "NY.GDP.MKTP.KD.ZG",
        "desc": "GDP growth (annual %)",
    },
    "unemployment": {
        "source": "wb", "code": "SL.UEM.TOTL.ZS",
        "desc": "Unemployment, total (% of labor force, modeled ILO)",
    },
    "gcf_pct_gdp": {
        "source": "wb", "code": "NE.GDI.TOTL.ZS",
        "desc": "Gross capital formation (% of GDP)",
    },
    "population": {
        "source": "wb", "code": "SP.POP.TOTL",
        "desc": "Population, total",
    },
    "urban_pct": {
        "source": "wb", "code": "SP.URB.TOTL.IN.ZS",
        "desc": "Urban population (% of total)",
    },
    # ---- POLLUTION / EMISSIONS ------------------------------------------- #
    "co2_pc": {
        "source": "wb", "code": "EN.GHG.CO2.PC.CE.AR5",
        "desc": "CO2 emissions per capita (t, excl. LULUCF)",
    },
    "co2_total_mt": {
        "source": "wb", "code": "EN.GHG.CO2.MT.CE.AR5",
        "desc": "CO2 emissions, total excl. LULUCF (Mt CO2e) -- aggregate scale, complements co2_pc",
    },
    "pm25_exposure": {
        "source": "wb", "code": "EN.ATM.PM25.MC.M3",
        "desc": "PM2.5 mean annual exposure (ug/m3)",
    },
    # ---- ENERGY ---------------------------------------------------------- #
    "renew_share": {
        "source": "wb", "code": "EG.FEC.RNEW.ZS",
        "desc": "Renewable energy (% of final energy consumption)",
    },
    "energy_use_pc": {
        "source": "wb", "code": "EG.USE.PCAP.KG.OE",
        "desc": "Energy use per capita (kg oil eq.) -- NB: WB series ends ~2014",
    },
    "energy_intensity": {
        "source": "wb", "code": "EG.EGY.PRIM.PP.KD",
        "desc": "Energy intensity of primary energy (MJ per $ 2017 PPP GDP)",
    },
    "fossil_fuel_pct": {
        "source": "wb", "code": "EG.USE.COMM.FO.ZS",
        "desc": "Fossil fuel energy consumption (% of total)",
    },
    "elec_access": {
        "source": "wb", "code": "EG.ELC.ACCS.ZS",
        "desc": "Access to electricity (% of population)",
    },
    "clean_cooking_access": {
        "source": "wb", "code": "EG.CFT.ACCS.ZS",
        "desc": "Access to clean fuels and technologies for cooking (% of population)",
    },
    # ---- DIGITAL / AI PROXY ---------------------------------------------- #
    # True AI investment indicators (OECD.AI) barely exist outside rich
    # countries; these are the best *globally available* tech-diffusion and
    # innovation-capacity proxies.
    "internet_users": {
        "source": "wb", "code": "IT.NET.USER.ZS",
        "desc": "Individuals using the internet (% of population)",
    },
    "hightech_exports_pct": {
        "source": "wb", "code": "TX.VAL.TECH.MF.ZS",
        "desc": "High-technology exports (% of manufactured exports)",
    },
    "patent_apps_resident": {
        "source": "wb", "code": "IP.PAT.RESD",
        "desc": "Patent applications, residents",
    },
    "ict_service_exports_pct": {
        "source": "wb", "code": "BX.GSR.CCIS.ZS",
        "desc": "ICT service exports (% of service exports, BoP)",
    },
    # ---- GOVERNANCE / INSTITUTIONS (WGI) ---------------------------------- #
    # Worldwide Governance Indicators, served through the same WB indicator
    # API under the GOV_WGI_* codes (the legacy bare CC.EST/GE.EST/etc. codes
    # were retired). Estimates roughly N(0,1); available from ~1996,
    # sparse/biennial before 2002.
    "control_corruption": {
        "source": "wb", "code": "GOV_WGI_CC.EST",
        "desc": "Control of Corruption (WGI estimate)",
    },
    "gov_effectiveness": {
        "source": "wb", "code": "GOV_WGI_GE.EST",
        "desc": "Government Effectiveness (WGI estimate)",
    },
    "political_stability": {
        "source": "wb", "code": "GOV_WGI_PV.EST",
        "desc": "Political Stability and Absence of Violence/Terrorism (WGI estimate)",
    },
    "regulatory_quality": {
        "source": "wb", "code": "GOV_WGI_RQ.EST",
        "desc": "Regulatory Quality (WGI estimate)",
    },
    "rule_of_law": {
        "source": "wb", "code": "GOV_WGI_RL.EST",
        "desc": "Rule of Law (WGI estimate)",
    },
    "voice_accountability": {
        "source": "wb", "code": "GOV_WGI_VA.EST",
        "desc": "Voice and Accountability (WGI estimate)",
    },
    # ---- TRADE & CAPITAL FLOWS -------------------------------------------- #
    "trade_pct_gdp": {
        "source": "wb", "code": "NE.TRD.GNFS.ZS",
        "desc": "Trade (exports + imports of goods and services, % of GDP)",
    },
    "fdi_net_inflow_pct_gdp": {
        "source": "wb", "code": "BX.KLT.DINV.WD.GD.ZS",
        "desc": "Foreign direct investment, net inflows (% of GDP)",
    },
    "gross_fixed_capital_pct_gdp": {
        "source": "wb", "code": "NE.GDI.FTOT.ZS",
        "desc": "Gross fixed capital formation (% of GDP)",
    },
    "gross_savings_pct_gdp": {
        "source": "wb", "code": "NY.GNS.ICTR.ZS",
        "desc": "Gross savings (% of GDP)",
    },
    # ---- INEQUALITY & POVERTY ---------------------------------------------- #
    "gini": {
        "source": "wb", "code": "SI.POV.GINI",
        "desc": "Gini index -- NB: survey-based, sparse coverage/irregular timing",
    },
    "poverty_215": {
        "source": "wb", "code": "SI.POV.DDAY",
        "desc": "Poverty headcount ratio at $2.15/day, 2017 PPP (% of population)",
    },
    # ---- HUMAN CAPITAL & DEMOGRAPHICS -------------------------------------- #
    "secondary_enroll": {
        "source": "wb", "code": "SE.SEC.ENRR",
        "desc": "School enrollment, secondary (% gross)",
    },
    "tertiary_enroll": {
        "source": "wb", "code": "SE.TER.ENRR",
        "desc": "School enrollment, tertiary (% gross)",
    },
    "fertility_rate": {
        "source": "wb", "code": "SP.DYN.TFRT.IN",
        "desc": "Fertility rate, total (births per woman)",
    },
    "dependency_ratio": {
        "source": "wb", "code": "SP.POP.DPND",
        "desc": "Age dependency ratio (% of working-age population)",
    },
    "health_exp_pct_gdp": {
        "source": "wb", "code": "SH.XPD.CHEX.GD.ZS",
        "desc": "Current health expenditure (% of GDP)",
    },
    "rnd_exp_pct_gdp": {
        "source": "wb", "code": "GB.XPD.RSDV.GD.ZS",
        "desc": "Research and development expenditure (% of GDP)",
    },
    # ---- WELL-BEING / DEVELOPMENT ---------------------------------------- #
    "life_expectancy": {
        "source": "wb", "code": "SP.DYN.LE00.IN",
        "desc": "Life expectancy at birth (years)",
    },

    # ---- OECD SDMX indicators: paste Developer-API URLs to activate ------- #
    # These are EXAMPLES of the URL shape. Verify/replace by copying the
    # exact current URL from the Data Explorer Developer API button, because
    # OECD dataflow IDs are versioned (the '@DF_...,1.0' part changes).
    #
    # "life_satisfaction": {
    #     "source": "oecd",
    #     "url": "https://sdmx.oecd.org/public/rest/data/OECD.WISE.WDP,DSD_HSL@DF_HSL_LV/.A.LIFESAT......?startPeriod=2010",
    #     "desc": "Life satisfaction (How's Life?)",
    # },
    # "ai_vc_investment": {
    #     "source": "oecd",
    #     "url": "<paste from OECD.AI Policy Observatory Data Explorer export>",
    #     "desc": "VC investment in AI firms (USD)",
    # },
    # "env_tax_pct_gdp": {
    #     "source": "oecd",
    #     "url": "<paste: Environmentally related tax revenue, % of GDP>",
    #     "desc": "Environmentally related taxes (% of GDP)",
    # },
}

# --------------------------------------------------------------------------- #
# CACHING                                                                      #
# --------------------------------------------------------------------------- #

def _cache_path(key: str, ext: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.{ext}"


def _get_with_retries(url: str, headers: dict | None = None) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:  # rate limited
                wait = SLEEP_BETWEEN_CALLS * (2 ** attempt)
                print(f"    rate limited, backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last_exc = exc
            print(f"    attempt {attempt} failed: {exc}")
            time.sleep(SLEEP_BETWEEN_CALLS * attempt)
    raise RuntimeError(f"Failed to fetch after {MAX_RETRIES} attempts: {url}") from last_exc


# --------------------------------------------------------------------------- #
# WORLD BANK FETCHER                                                           #
# --------------------------------------------------------------------------- #

def fetch_worldbank(code: str, name: str) -> pd.DataFrame:
    """Return long df: iso3, year, indicator=name, value, source='wb'."""
    cache = _cache_path(f"wb::{code}", "json")
    if cache.exists():
        pages = json.loads(cache.read_text())
    else:
        pages = []
        page = 1
        while True:
            url = (
                f"https://api.worldbank.org/v2/country/all/indicator/{code}"
                f"?format=json&per_page=20000&page={page}"
            )
            r = _get_with_retries(url)
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                break
            meta, rows = payload[0], payload[1]
            pages.extend(rows)
            if page >= meta.get("pages", 1):
                break
            page += 1
            time.sleep(SLEEP_BETWEEN_CALLS)
        cache.write_text(json.dumps(pages))

    recs = []
    for row in pages:
        iso3 = row.get("countryiso3code") or ""
        val = row.get("value")
        yr = row.get("date")
        if not iso3 or val is None or yr is None:
            continue
        recs.append((iso3, int(yr), float(val)))

    df = pd.DataFrame(recs, columns=["iso3", "year", "value"])
    df["indicator"] = name
    df["source"] = "wb"
    return df


def fetch_wb_country_meta() -> pd.DataFrame:
    """Return df[iso3, country_name, region, income_group] for REAL countries.

    World Bank flags aggregates (World, regions, income groups) with
    region == 'Aggregates'. We use that to keep only genuine countries and to
    attach region + income classifications for grouped EDA.
    """
    cache = _cache_path("wb::countrymeta", "json")
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
        r = _get_with_retries(url)
        payload = r.json()
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        cache.write_text(json.dumps(rows))

    recs = []
    for row in rows:
        region = (row.get("region") or {}).get("value", "")
        if region == "Aggregates":            # drop World / regions / income groups
            continue
        recs.append((
            row.get("id", ""),
            row.get("name", ""),
            region,
            (row.get("incomeLevel") or {}).get("value", ""),
        ))
    return pd.DataFrame(recs, columns=["iso3", "country_name", "region", "income_group"])


# --------------------------------------------------------------------------- #
# OECD SDMX FETCHER                                                            #
# --------------------------------------------------------------------------- #

def _ensure_csv_format(url: str) -> str:
    if "format=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}dimensionAtObservation=AllDimensions&format=csvfilewithlabels"


def fetch_oecd(url: str, name: str) -> pd.DataFrame:
    """Return long df from an OECD SDMX-CSV endpoint."""
    url = _ensure_csv_format(url)
    cache = _cache_path(f"oecd::{url}", "csv")
    if cache.exists():
        raw = cache.read_text()
    else:
        r = _get_with_retries(url, headers={"Accept": "text/csv"})
        raw = r.text
        cache.write_text(raw)
        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.read_csv(io.StringIO(raw))
    return normalize_sdmx(df, name)


def normalize_sdmx(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Turn a raw SDMX-CSV frame into long iso3/year/value, applying UNIT_MULT.

    Handles sub-annual data by averaging to annual, with a warning.
    """
    cols = {c.upper(): c for c in df.columns}
    ref = cols.get("REF_AREA")
    tim = cols.get("TIME_PERIOD")
    obs = cols.get("OBS_VALUE")
    if not (ref and tim and obs):
        raise ValueError(
            f"[{name}] SDMX frame missing REF_AREA/TIME_PERIOD/OBS_VALUE. "
            f"Got columns: {list(df.columns)}"
        )

    out = df[[ref, tim, obs]].copy()
    out.columns = ["iso3", "time", "value"]

    # apply unit multiplier if present (value * 10^UNIT_MULT)
    mult_col = cols.get("UNIT_MULT")
    if mult_col and mult_col in df.columns:
        mult = pd.to_numeric(df[mult_col], errors="coerce").fillna(0)
        out["value"] = pd.to_numeric(out["value"], errors="coerce") * (10 ** mult)
    else:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # extract 4-digit year; flag sub-annual
    out["year"] = out["time"].astype(str).str.extract(r"(\d{4})").astype("Int64")
    subannual = out["time"].astype(str).str.contains(r"-|Q|M", regex=True).any()
    out = out.dropna(subset=["year", "value"])
    out["year"] = out["year"].astype(int)

    if subannual:
        n_before = len(out)
        out = (
            out.groupby(["iso3", "year"], as_index=False)["value"]
            .mean()
        )
        print(f"    note: [{name}] sub-annual data averaged to annual "
              f"({n_before} obs -> {len(out)})")
    else:
        out = out[["iso3", "year", "value"]]

    out["indicator"] = name
    out["source"] = "oecd"
    return out


# --------------------------------------------------------------------------- #
# SANITY CHECKS                                                               #
# --------------------------------------------------------------------------- #
# Indicators that are genuine physical/consumption shares and therefore
# cannot be negative. NB: deliberately excludes fdi_net_inflow_pct_gdp,
# gcf_pct_gdp, gross_fixed_capital_pct_gdp, gross_savings_pct_gdp, and
# *growth* indicators -- those are legitimately negative in real data
# (net divestment, capital destocking/destruction, dissaving in poor or
# war-torn economies -- e.g. Sierra Leone 1997, Sao Tome, Djibouti). Verified
# against raw fetched values: of this whole indicator set, only
# fossil_fuel_pct actually has negative observations, and they're a WB
# series-transition artifact concentrated in unrelated countries in 2015
# (LTU/KWT/SGP/SUR/NER/BLR) rather than a real economic pattern.
SHARE_FLOOR_ZERO_INDICATORS = {
    "urban_pct", "unemployment", "internet_users", "elec_access",
    "clean_cooking_access", "renew_share", "fossil_fuel_pct",
    "trade_pct_gdp", "hightech_exports_pct", "ict_service_exports_pct",
    "secondary_enroll", "tertiary_enroll", "gini", "poverty_215",
    "health_exp_pct_gdp", "rnd_exp_pct_gdp", "dependency_ratio",
}
# Indicators with a physically plausible absolute range; WDI occasionally
# publishes single-year glitches (e.g. life expectancy dropping to ~15 for
# one year and rebounding) that are data errors, not real crises.
PLAUSIBLE_RANGE = {
    "life_expectancy": (25, 90),
}
# Indicators sparse enough that they never reach ~50% cross-sectional
# coverage in any single year (survey-based, irregular timing); excluding
# them from a complete-case panel is far less punishing than including them.
SPARSE_INDICATORS = [
    "gini", "poverty_215", "rnd_exp_pct_gdp", "hightech_exports_pct",
    "patent_apps_resident", "secondary_enroll", "tertiary_enroll",
]


def sanity_clip(long_df: pd.DataFrame) -> pd.DataFrame:
    """Drop physically-impossible raw observations (WDI data artifacts)
    instead of silently passing them downstream. Long format, so a dropped
    row is equivalent to a missing/NaN cell after pivoting to wide."""
    df = long_df
    mask = df["indicator"].isin(SHARE_FLOOR_ZERO_INDICATORS) & (df["value"] < 0)
    if mask.any():
        bad_inds = sorted(df.loc[mask, "indicator"].unique())
        print(f"sanity_clip: dropping {mask.sum()} negative share-indicator "
              f"values {bad_inds}")
        df = df[~mask]
    for name, (lo, hi) in PLAUSIBLE_RANGE.items():
        m = (df["indicator"] == name) & ((df["value"] < lo) | (df["value"] > hi))
        if m.any():
            print(f"sanity_clip: dropping {m.sum()} out-of-range [{lo},{hi}] "
                  f"values for {name}")
            df = df[~m]
    return df.reset_index(drop=True)


def add_micro_state_flag(wide: pd.DataFrame, threshold: int = 1_000_000) -> pd.DataFrame:
    """Flag entities whose most recent available population is below
    `threshold`. Many are dependencies/territories, not sovereign states, and
    disproportionately drive coverage/outlier issues in a cross-country panel.
    Rows are kept (not dropped) -- callers filter with `df[~df.is_micro_state]`."""
    df = wide.copy()
    if "population" not in df.columns:
        df["is_micro_state"] = False
        return df
    latest_pop = (
        df.dropna(subset=["population"])
        .sort_values("year")
        .groupby("iso3")["population"]
        .last()
    )
    df["is_micro_state"] = (df["iso3"].map(latest_pop) < threshold).fillna(False)
    return df


# --------------------------------------------------------------------------- #
# ASSEMBLY                                                                     #
# --------------------------------------------------------------------------- #

def build_long(valid_iso3: set[str] | None = None) -> pd.DataFrame:
    frames = []
    for name, spec in INDICATORS.items():
        src = spec["source"]
        print(f"[{src:4}] {name}: {spec.get('desc', '')}")
        try:
            if src == "wb":
                frame = fetch_worldbank(spec["code"], name)
            elif src == "oecd":
                frame = fetch_oecd(spec["url"], name)
            else:
                print(f"    skipped: unknown source '{src}'")
                continue
        except Exception as exc:  # keep going if one indicator fails
            print(f"    ERROR: {exc}")
            continue
        print(f"    -> {len(frame):,} obs, "
              f"{frame['iso3'].nunique()} countries")
        frames.append(frame)
        time.sleep(SLEEP_BETWEEN_CALLS)  # be polite between indicators too;
        # firing ~30 requests back-to-back triggers spurious 400s from the WB API

    if not frames:
        raise RuntimeError("No indicators fetched. Check network / URLs.")

    long_df = pd.concat(frames, ignore_index=True)

    # country filtering
    if COUNTRIES:
        long_df = long_df[long_df["iso3"].isin(COUNTRIES)]
    elif valid_iso3 is not None:
        # worldwide, but only rows that match a real WB country (drops aggregates)
        long_df = long_df[long_df["iso3"].isin(valid_iso3)]
    else:
        long_df = long_df[~long_df["iso3"].isin(AGGREGATE_CODES)]
        long_df = long_df[long_df["iso3"].str.fullmatch(r"[A-Z]{3}")]

    # year window + de-dup (last write wins on collisions)
    long_df = long_df[(long_df["year"] >= START_YEAR) & (long_df["year"] <= END_YEAR)]
    long_df = long_df.drop_duplicates(subset=["iso3", "year", "indicator"], keep="last")
    return long_df.reset_index(drop=True)


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        long_df.pivot_table(
            index=["iso3", "year"], columns="indicator", values="value"
        )
        .reset_index()
        .sort_values(["iso3", "year"])
    )
    wide.columns.name = None
    return wide


def coverage_report(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby("indicator")
    rep = pd.DataFrame({
        "n_obs": g.size(),
        "n_countries": g["iso3"].nunique(),
        "year_min": g["year"].min(),
        "year_max": g["year"].max(),
        "source": g["source"].first(),
    }).reset_index().sort_values(["source", "indicator"])
    return rep


def availability_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.pivot_table(
            index="iso3", columns="indicator", values="value", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
    )


def add_engineered_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `wide` with econometrics-ready derived columns.

    All lag/growth features are computed within each country (grouped on
    iso3) over rows sorted by year, so they respect panel structure.
    """
    df = wide.sort_values(["iso3", "year"]).reset_index(drop=True).copy()

    if "gdp_pc_ppp_const" in df.columns:
        gdp = pd.to_numeric(df["gdp_pc_ppp_const"], errors="coerce")
        df["log_gdp_pc"] = np.log(gdp.where(gdp > 0))
        df["gdp_pc_growth_calc"] = df.groupby("iso3")["gdp_pc_ppp_const"].pct_change()
        df["gdp_pc_ppp_const_lag1"] = df.groupby("iso3")["gdp_pc_ppp_const"].shift(1)

    if "population" in df.columns:
        pop = pd.to_numeric(df["population"], errors="coerce")
        df["log_population"] = np.log(pop.where(pop > 0))

    if "co2_pc" in df.columns:
        co2_pc = pd.to_numeric(df["co2_pc"], errors="coerce")
        df["log_co2_pc"] = np.log1p(co2_pc)

    if have_cols(df, "co2_pc", "gdp_pc_ppp_const"):
        co2_pc = pd.to_numeric(df["co2_pc"], errors="coerce")
        gdp = pd.to_numeric(df["gdp_pc_ppp_const"], errors="coerce")
        df["co2_per_1000gdp"] = (co2_pc / gdp.where(gdp > 0)) * 1000

    df["trend"] = df["year"] - START_YEAR
    return df


def have_cols(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def summarize_balance(wide: pd.DataFrame) -> None:
    meta_cols = {"iso3", "country_name", "region", "income_group", "year",
                 "is_micro_state"}
    ind_cols = [c for c in wide.columns if c not in meta_cols]
    complete = wide.dropna(subset=ind_cols)
    print("\n--- PANEL BALANCE ---")
    print(f"countries: {wide['iso3'].nunique()}   "
          f"years: {wide['year'].min()}-{wide['year'].max()}   "
          f"indicators: {len(ind_cols)}")
    print(f"rows (any data): {len(wide):,}")
    print(f"rows complete on ALL indicators: {len(complete):,}")
    if len(complete):
        print(f"balanced window (complete rows): "
              f"{complete['year'].min()}-{complete['year'].max()}, "
              f"{complete['iso3'].nunique()} countries")

    # a more realistic "how much do I actually have to model on" number:
    # drop the structurally-sparse indicators and micro-states/territories
    if "is_micro_state" in wide.columns:
        core_ind_cols = [c for c in ind_cols if c not in SPARSE_INDICATORS]
        core = (wide[~wide["is_micro_state"]]
                .dropna(subset=core_ind_cols))
        print(f"\nrows complete on the {len(core_ind_cols)} non-sparse "
              f"indicators, excluding micro-states: {len(core):,}")
        if len(core):
            print(f"  window: {core['year'].min()}-{core['year'].max()}, "
                  f"{core['iso3'].nunique()} countries")

    # per-indicator missingness
    miss = wide[ind_cols].isna().mean().sort_values(ascending=False)
    print("\nmissing share by indicator (highest first):")
    for k, v in miss.items():
        print(f"  {k:22s} {v:5.1%}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Country metadata first: defines the real-country universe + region/income.
    meta = None
    valid_iso3 = None
    if not COUNTRIES:
        print("Fetching World Bank country metadata (real countries + region/income)...")
        try:
            meta = fetch_wb_country_meta()
            valid_iso3 = set(meta["iso3"])
            print(f"    -> {len(meta)} countries recognised\n")
        except Exception as exc:
            print(f"    WARNING: metadata fetch failed ({exc}); "
                  f"falling back to code-pattern filtering\n")

    print("Fetching indicators...\n")
    long_df = build_long(valid_iso3=valid_iso3)
    long_df = sanity_clip(long_df)
    wide = to_wide(long_df)
    wide = add_micro_state_flag(wide)

    # attach geography as leading columns for grouped EDA
    if meta is not None:
        wide = wide.merge(meta, on="iso3", how="left")
        lead = ["iso3", "country_name", "region", "income_group", "year"]
        wide = wide[lead + [c for c in wide.columns if c not in lead]]

    long_df.to_csv(OUTPUT_DIR / "panel_long.csv", index=False)
    wide.to_csv(OUTPUT_DIR / "panel_wide.csv", index=False)
    coverage_report(long_df).to_csv(OUTPUT_DIR / "coverage.csv", index=False)
    availability_matrix(long_df).to_csv(OUTPUT_DIR / "availability.csv")
    if meta is not None:
        meta.to_csv(OUTPUT_DIR / "country_meta.csv", index=False)

    try:
        wide.to_parquet(OUTPUT_DIR / "panel_wide.parquet", index=False)
    except Exception:
        pass  # pyarrow not installed; CSV is enough

    model_ready = add_engineered_features(wide)
    model_ready.to_csv(OUTPUT_DIR / "panel_model_ready.csv", index=False)
    try:
        model_ready.to_parquet(OUTPUT_DIR / "panel_model_ready.parquet", index=False)
    except Exception:
        pass  # pyarrow not installed; CSV is enough

    summarize_balance(wide)
    print(f"\nWrote outputs to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
