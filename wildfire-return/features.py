from __future__ import annotations

"""Feature engineering with fire physics -- the Round-3 upgrade.

Round 1 fed the model raw cached columns. That left three kinds of value on the
table:

1. CIRCULAR VARIABLES TREATED AS LINEAR. Aspect went in as degrees, so a
   north-facing slope at 359 deg and one at 1 deg looked maximally far apart.
   Wind direction was not even cached.
2. NO PLACE-RELATIVE CONTEXT. An ERC of 70 means something different in the
   Mojave than in a coastal forest. The dataset ships climatological normals;
   nobody subtracted them.
3. NO DOMAIN INDICES. Fire science has published operational indices that
   combine these variables in known-useful ways (Hot-Dry-Windy, Fosberg). They
   are three lines of arithmetic each.

FEATURE_SETS defines a cumulative ablation ladder so the article can report what
each family actually bought, with confidence intervals, instead of asserting
that feature engineering helped.
"""

import numpy as np
import pandas as pd

import wildfire as wf

HISTORY_CELL_DEG = 0.25
HISTORY_WINDOW_YEARS = 5
LARGE_FIRE_ACRES = 100.0


# ---------------------------------------------------------------------------
# Unit helpers (the dataset's units are documented in DATA_DICTIONARY.md)
# ---------------------------------------------------------------------------
def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


def ms_to_mph(v):
    return v * 2.236936


def kpa_to_hpa(v):
    return v * 10.0


# ---------------------------------------------------------------------------
# Operational fire-weather indices
# ---------------------------------------------------------------------------
def hot_dry_windy(vpd_kpa, wind_ms):
    """Hot-Dry-Windy Index (Srock et al. 2018): the product of vapour-pressure
    deficit and wind speed. Designed to capture the atmosphere's capacity to
    dry fuels AND push a fire, which is exactly the wind-driven-run regime."""
    return kpa_to_hpa(np.asarray(vpd_kpa, dtype="float64")) * np.asarray(wind_ms, dtype="float64")


def fosberg_ffwi(tmmx_k, rh_pct, wind_ms):
    """Fosberg Fire Weather Index. Equilibrium moisture content has three
    humidity branches; eta is the moisture damping term; the wind term enters
    as sqrt(1 + U^2). Output is conventionally capped at 100."""
    t = k_to_f(np.asarray(tmmx_k, dtype="float64"))
    h = np.asarray(rh_pct, dtype="float64")
    u = ms_to_mph(np.asarray(wind_ms, dtype="float64"))

    m = np.where(
        h < 10,
        0.03229 + 0.281073 * h - 0.000578 * h * t,
        np.where(
            h < 50,
            2.22749 + 0.160107 * h - 0.014784 * t,
            21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t - 0.483199 * h,
        ),
    )
    m30 = np.clip(m, 0, 30) / 30.0
    eta = 1 - 2 * m30 + 1.5 * m30 ** 2 - 0.5 * m30 ** 3
    ffwi = eta * np.sqrt(1 + u ** 2) / 0.3002
    return np.clip(ffwi, 0, 100)


def _pctl_to_midpoint(s: pd.Series) -> pd.Series:
    """'70-90%' -> 0.80, '>90%' -> 0.95, '<10%' -> 0.05. The percentile columns
    ship as strings; an ordinal midpoint is monotone and model-friendly."""
    txt = s.astype("string").str.strip().str.replace("%", "", regex=False)

    def one(v):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        v = str(v)
        try:
            if v.startswith(">"):
                lo = float(v[1:])
                return (lo + 100) / 2 / 100
            if v.startswith("<"):
                hi = float(v[1:])
                return hi / 2 / 100
            if "-" in v:
                a, b = v.split("-", 1)
                return (float(a) + float(b)) / 2 / 100
            return float(v) / 100
        except ValueError:
            return np.nan

    return txt.map(one).astype("float32")


# ---------------------------------------------------------------------------
# Leakage-safe spatiotemporal history
# ---------------------------------------------------------------------------
def add_history(df: pd.DataFrame, embargo_large: bool = False) -> pd.DataFrame:
    """Prior-fire context per 0.25-degree cell, using STRICTLY EARLIER fires only.

    For each ignition: how many fires this cell already had in the preceding 5
    years, how long since its last >=100-acre fire, and how many OTHER fires
    ignited in the same cell on the same day (the resource-competition signal
    the Round-4 red team found missing: 28.7% of CA rows share a cell-day).

    embargo_large (Round-4 red team, F17): the default "years since large fire"
    marks prior fires large by their FINAL size -- but a fire that ignited two
    days ago and is still burning is not yet KNOWN to be large at deployment
    time. With embargo_large=True, a prior fire only counts as a known large
    fire once its containment date (discovery + burn_days) precedes the new
    ignition; fires with unknown containment never qualify (conservative).

    Caveat carried in print: same-day counts are date-resolution -- some
    same-day fires ignite later in the day than the target fire. Operational
    use would consume time-stamped feeds.
    """
    d = df.copy()
    d["_cell"] = (np.floor(d["lat"] / HISTORY_CELL_DEG).astype("Int64").astype(str) + "_"
                  + np.floor(d["lon"] / HISTORY_CELL_DEG).astype("Int64").astype(str))
    d = d.sort_values("discovery_date", kind="mergesort").reset_index(drop=False)

    prior_n = np.zeros(len(d), dtype="float32")
    yrs_since_large = np.full(len(d), np.nan, dtype="float32")
    window = np.timedelta64(365 * HISTORY_WINDOW_YEARS, "D")

    for _, grp in d.groupby("_cell", sort=False):
        pos = grp.index.to_numpy()
        dates = grp["discovery_date"].to_numpy(dtype="datetime64[ns]")
        # strictly-earlier count inside the 5-year window
        left = np.searchsorted(dates, dates, side="left")            # excludes ties
        lo = np.searchsorted(dates, dates - window, side="left")
        prior_n[pos] = (left - lo).astype("float32")
        # time since the cell's last KNOWN large fire, strictly before this one
        big = grp["fire_size"].to_numpy(dtype="float64") >= LARGE_FIRE_ACRES
        if embargo_large:
            # size becomes KNOWN at containment = discovery + burn_days;
            # unknown containment (46% of rows) -> never qualifies (conservative)
            burn = grp["burn_days"].to_numpy(dtype="float64")
            big = big & np.isfinite(burn)
            cont = dates + np.nan_to_num(burn, nan=0.0).astype("int64").astype("timedelta64[D]")
            big_dates_all = cont[big]
        else:
            big_dates_all = dates[big]
        big_dates = np.sort(big_dates_all)
        if big_dates.size:
            k = np.searchsorted(big_dates, dates, side="left")        # strictly earlier
            has_prev = k > 0
            delta = np.full(len(dates), np.nan, dtype="float64")
            delta[has_prev] = ((dates[has_prev] - big_dates[k[has_prev] - 1])
                               / np.timedelta64(1, "D")) / 365.25
            yrs_since_large[pos] = np.maximum(delta, 0).astype("float32")

    d["hist_prior_fires_5y"] = prior_n
    d["hist_years_since_large"] = yrs_since_large
    d["hist_has_prior_large"] = (~np.isnan(yrs_since_large)).astype("float32")
    # same-day cell ignition load: other fires reported in this cell today
    d["same_day_cell_ignitions"] = (
        d.groupby(["_cell", d["discovery_date"].dt.floor("D")])["fod_id"]
        .transform("size") - 1).astype("float32")
    out = d.sort_values("index", kind="mergesort").set_index("index")
    out.index.name = None
    return out.drop(columns=["_cell"])


def assert_history_is_causal(df: pd.DataFrame, n_samples: int = 300, seed: int = 42) -> dict:
    """Proof, not vibes: recompute the history features for random rows from the
    full frame filtered to strictly-earlier dates, and require equality.

    Round-4 red team: the original version verified ONLY hist_prior_fires_5y
    while the article claimed the assertion covered 'the feature'. It now
    brute-force-verifies all three history columns plus the same-day count.
    (This checks the default, non-embargoed years_since_large definition.)
    """
    have = df[df["hist_prior_fires_5y"].notna()]
    sample = have.sample(min(n_samples, len(have)), random_state=seed)
    window = pd.Timedelta(days=365 * HISTORY_WINDOW_YEARS)
    cells = np.floor(df[["lat", "lon"]] / HISTORY_CELL_DEG)
    bad: dict[str, int] = {"prior_5y": 0, "years_since_large": 0, "same_day": 0}
    for _, row in sample.iterrows():
        cl = np.floor(row["lat"] / HISTORY_CELL_DEG)
        cn = np.floor(row["lon"] / HISTORY_CELL_DEG)
        m = (cells["lat"] == cl) & (cells["lon"] == cn)
        cell_df = df[m]
        earlier = cell_df[cell_df["discovery_date"] < row["discovery_date"]]
        # 1) prior-fire count in the window
        expect_n = int((earlier["discovery_date"] >= row["discovery_date"] - window).sum())
        if expect_n != int(row["hist_prior_fires_5y"]):
            bad["prior_5y"] += 1
        # 2) years since last large fire (strictly earlier, final-size definition)
        big_earlier = earlier[earlier["fire_size"] >= LARGE_FIRE_ACRES]
        if len(big_earlier):
            expect_y = (row["discovery_date"] - big_earlier["discovery_date"].max()).days / 365.25
            got = row["hist_years_since_large"]
            if not (np.isfinite(got) and abs(expect_y - got) < 0.02):
                bad["years_since_large"] += 1
        elif np.isfinite(row["hist_years_since_large"]):
            bad["years_since_large"] += 1
        # 3) same-day cell count excludes nothing but itself
        same = cell_df[cell_df["discovery_date"].dt.floor("D")
                       == row["discovery_date"].floor("D")]
        if len(same) - 1 != int(row["same_day_cell_ignitions"]):
            bad["same_day"] += 1
    total_bad = sum(bad.values())
    if total_bad:
        raise AssertionError(f"history features look forward: {bad} in {len(sample)} sampled rows")
    return {"checked": int(len(sample)), "violations": 0, "columns_verified": 3,
            "note": "all three history columns + same-day count recomputed from "
                    "strictly-earlier/same-day rows and matched"}


# ---------------------------------------------------------------------------
# The feature builder
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame, with_history: bool = True) -> pd.DataFrame:
    """Add every engineered column. Missing inputs degrade gracefully -- a
    feature whose source column is absent is simply not created, and the rung
    lists below intersect with what exists."""
    d = add_history(df) if with_history else df.copy()

    # Byte-raster nodata: the invasive-grass layers use 255 as "no data", which
    # would otherwise read as 255% cover. Found by inspecting the v2 cache.
    for col in ("cheatgrass", "exotic_grass"):
        if col in d.columns:
            d[col] = d[col].mask(d[col] >= 255)

    have = set(d.columns)

    def has(*cols):
        return all(c in have for c in cols)

    # -- circular encodings (the bug fix) -----------------------------------
    if has("aspect"):
        asp = d["aspect"].where(d["aspect"] >= 0)          # -1 = flat terrain
        rad = np.radians(asp.astype("float64"))
        d["northness"] = np.cos(rad).astype("float32")
        d["eastness"] = np.sin(rad).astype("float32")
        d["is_flat"] = (d["aspect"] < 0).astype("float32")
    if has("wind_dir", "wind"):
        wr = np.radians(d["wind_dir"].astype("float64"))
        d["wind_u"] = (d["wind"] * np.sin(wr)).astype("float32")
        d["wind_v"] = (d["wind"] * np.cos(wr)).astype("float32")
    if has("doy_std"):
        ang = 2 * np.pi * d["doy_std"].astype("float64") / 365.0
        d["doy_sin"] = np.sin(ang).astype("float32")
        d["doy_cos"] = np.cos(ang).astype("float32")

    # -- wind-terrain alignment: the Diablo/Santa Ana mechanism -------------
    # Wind direction is the direction wind blows FROM, so it blows TOWARD
    # wind_dir+180. Alignment with the slope's aspect means downslope-driven.
    if has("wind_dir", "aspect", "slope", "wind"):
        toward = np.radians((d["wind_dir"].astype("float64") + 180.0) % 360.0)
        asp_r = np.radians(d["aspect"].where(d["aspect"] >= 0).astype("float64"))
        align = np.cos(toward - asp_r)
        d["wind_slope_align"] = align.astype("float32")
        d["downslope_wind"] = (align * d["wind"].astype("float64")
                               * np.sin(np.radians(d["slope"].astype("float64")))).astype("float32")

    # -- place-relative anomalies ------------------------------------------
    for col, norm in (("erc", "erc_normal"), ("vpd", "vpd_normal"), ("fm100", "fm100_normal"),
                      ("fm1000", "fm1000_normal"), ("bi", "bi_normal"), ("tmmx", "tmmx_normal"),
                      ("rmin", "rmin_normal"), ("precip", "precip_normal")):
        if has(col, norm):
            d[f"{col}_anom"] = (d[col].astype("float32") - d[norm].astype("float32"))

    # -- operational indices ------------------------------------------------
    if has("vpd", "wind"):
        d["hdw"] = hot_dry_windy(d["vpd"], d["wind"]).astype("float32")
    if has("vpd_5d_max", "wind_5d_max"):
        d["hdw_5d_max"] = hot_dry_windy(d["vpd_5d_max"], d["wind_5d_max"]).astype("float32")
    if has("tmmx", "rmin", "wind"):
        d["ffwi"] = fosberg_ffwi(d["tmmx"], d["rmin"], d["wind"]).astype("float32")

    # -- drought memory -----------------------------------------------------
    if has("erc_5d_max", "erc"):
        d["erc_rising"] = (d["erc_5d_max"].astype("float32") - d["erc"].astype("float32"))
    if has("precip_5d_mean"):
        d["dry_spell"] = (d["precip_5d_mean"].fillna(0) <= 0.01).astype("float32")
    if has("tmmx", "tmmn"):
        d["diurnal_range"] = (d["tmmx"].astype("float32") - d["tmmn"].astype("float32"))
    if has("rmax", "rmin"):
        d["rh_range"] = (d["rmax"].astype("float32") - d["rmin"].astype("float32"))

    # -- fuels: load x dryness ---------------------------------------------
    if has("evc"):     # LANDFIRE codes: 100s tree, 200s shrub, 300s herb; %cover = value % 100
        v = d["evc"].astype("float64")     # <100 are water/barren/developed classes
        d["evc_cover_pct"] = np.where(v >= 100, v % 100, np.nan).astype("float32")
        d["evc_lifeform"] = np.where(v >= 100, v // 100, np.nan).astype("float32")
        d["evc_nonveg"] = (v < 100).astype("float32")
    if has("evh"):
        v = d["evh"].astype("float64")
        d["evh_height"] = np.where(v >= 100, v % 100, np.nan).astype("float32")
    if has("rpms", "fm1000"):
        d["fuel_load_dryness"] = (d["rpms"].astype("float32")
                                  / (d["fm1000"].astype("float32") + 1.0))
    if has("cheatgrass", "vpd"):
        d["cheatgrass_x_vpd"] = (d["cheatgrass"].astype("float32") * d["vpd"].astype("float32"))
    if has("exotic_grass", "erc"):
        d["exotic_x_erc"] = (d["exotic_grass"].astype("float32") * d["erc"].astype("float32"))

    # -- human exposure / suppression capacity ------------------------------
    if has("population"):
        d["log_population"] = np.log1p(d["population"].clip(lower=0)).astype("float32")
    if has("population", "ghm"):
        d["pop_x_ghm"] = (np.log1p(d["population"].clip(lower=0)) * d["ghm"]).astype("float32")
    if has("firestations_10km"):
        # missing here means "none found within 10 km", not "unknown"
        d["firestations_10km_f"] = d["firestations_10km"].fillna(0).astype("float32")

    # -- ordinal percentiles -------------------------------------------------
    # Several *_Percentile columns ship degenerate (tmmx_pctl carries only the
    # '>90%' bin in CA). A constant is not a feature; skip it rather than
    # feeding the model a column with zero information.
    for col in ("erc_pctl", "vpd_pctl", "tmmx_pctl", "wind_pctl", "fm100_pctl", "bi_pctl"):
        if col in have and d[col].nunique(dropna=True) > 1:
            d[f"{col}_mid"] = _pctl_to_midpoint(d[col])

    return d


# ---------------------------------------------------------------------------
# The ablation ladder
# ---------------------------------------------------------------------------
RUNGS: dict[str, list[str]] = {
    "R0_base": list(wf.ML_FEATURES_NUM),
    "R1_circular": ["northness", "eastness", "is_flat", "doy_sin", "doy_cos",
                    "wind_u", "wind_v"],
    "R2_anomalies": ["erc_anom", "vpd_anom", "fm100_anom", "fm1000_anom", "bi_anom",
                     "tmmx_anom", "rmin_anom", "precip_anom",
                     "erc_pctl_mid", "vpd_pctl_mid", "tmmx_pctl_mid", "wind_pctl_mid",
                     "fm100_pctl_mid", "bi_pctl_mid"],
    "R3_indices": ["hdw", "hdw_5d_max", "ffwi", "erc_rising", "dry_spell",
                   "diurnal_range", "rh_range"],
    "R4_windterrain": ["wind_slope_align", "downslope_wind", "wind_dir", "wind_dir_5d_max",
                       "tri", "srad", "etr", "sph", "tmmn", "rmax"],
    "R5_fuels": ["evc_cover_pct", "evc_lifeform", "evc_nonveg", "evh_height", "rpms",
                 "cheatgrass", "exotic_grass", "fuel_load_dryness", "cheatgrass_x_vpd",
                 "exotic_x_erc", "annual_precip", "annual_temp", "sdi"],
    "R6_history": ["hist_prior_fires_5y", "hist_years_since_large", "hist_has_prior_large",
                   "log_population", "pop_x_ghm", "firestations_10km_f"],
}
CAT_FEATURES = list(wf.ML_FEATURES_CAT) + ["evt"]


def cumulative_rungs() -> dict[str, list[str]]:
    """R0, R0+R1, R0+R1+R2, ... -- the ladder the ablation walks."""
    out, acc = {}, []
    for name, cols in RUNGS.items():
        acc = acc + cols
        out[name] = list(acc)
    return out


def feature_list(df: pd.DataFrame, rung: str = "R6_history",
                 include_cats: bool = True) -> tuple[list[str], list[str]]:
    """Numeric + categorical feature names for a rung, intersected with reality."""
    cum = cumulative_rungs()
    if rung not in cum:
        raise KeyError(f"unknown rung {rung}; have {list(cum)}")
    have = set(df.columns)
    num = [c for c in cum[rung] if c in have]
    cats = [c for c in CAT_FEATURES if c in have] if include_cats else []
    banned = set(wf.ML_EXCLUDED_LEAKS)
    leaked = [c for c in num + cats if c in banned]
    if leaked:
        raise AssertionError(f"leak columns present in feature list: {leaked}")
    return num, cats


def build_matrix(df: pd.DataFrame, rung: str = "R6_history",
                 include_cats: bool = True) -> pd.DataFrame:
    """float32 design matrix (memory matters: 7 GB box) with pandas categoricals
    that HistGradientBoosting/LightGBM consume natively."""
    num, cats = feature_list(df, rung, include_cats)
    X = df[num].astype("float32").copy()
    # belt-and-braces: a derived feature can overflow float32 even when its
    # inputs are clean (a product of two large values). Non-finite -> NaN, which
    # the tree models handle natively and the linear models impute.
    X = X.replace([np.inf, -np.inf], np.nan)
    # drop constants: a zero-variance column cannot help and pollutes importance
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[keep]
    for c in cats:
        if df[c].nunique(dropna=True) > 1:
            X[c] = df[c].astype("category")
    return X


def univariate_tripwire(X: pd.DataFrame, y, max_auc: float = 0.90) -> dict:
    """v1 leak detector -- KEPT FOR THE RECORD, KNOWN INSUFFICIENT.

    The Round-4 red team proved this catches only near-perfect continuous
    leaks (fire_size, AUC 1.0) and is blind to rare presence-flag leaks:
    has_mtbs has univariate AUC ~0.66 while inflating model AP 3.5x, because
    AUC cannot see a flag that covers 31% of positives at 99% precision.
    It also ran on the post-ban matrix, so it could never even see the banned
    columns -- circular by construction. Use leak_tripwire_v2 instead; this
    stays so the published Round-3 numbers remain reproducible."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    flags, worst = [], []
    for c in X.columns:
        v = X[c]
        if str(v.dtype) == "category":
            continue
        v = v.to_numpy(dtype="float64")
        m = np.isfinite(v)
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            continue
        auc = roc_auc_score(y[m], v[m])
        auc = max(auc, 1 - auc)
        worst.append((c, float(auc)))
        if auc > max_auc:
            flags.append({"feature": c, "auc": float(auc)})
    worst.sort(key=lambda t: -t[1])
    return {"threshold": max_auc, "flagged": flags,
            "top_single_feature_auc": worst[:5], "passed": not flags}


def leak_tripwire_v2(X: pd.DataFrame, y, ap_lift_max: float = 5.0,
                     presence_lift_max: float = 20.0, auc_max: float = 0.80,
                     enforce: bool = False) -> dict:
    """v2 leak detector, built after v1 failed its own showcase.

    Thresholds are CALIBRATED FROM MEASURED DATA on this project's own leak
    ladder, not chosen as round numbers: honest at-ignition features top out
    at 1.72x AP lift and ~0.68 AUC; the weakest known leak (burn_days) sits at
    7.1x and 0.796. The 5.0x AP-lift threshold therefore splits the two
    populations with ~3x margin above honest and ~30% below the weakest leak.

    Three complementary criteria per column:
      1. SINGLE-FEATURE RANKING AP / base_rate > ap_lift_max. Catches
         continuous label-source leaks (fire_size, ~46x) and rare
         high-precision flags (has_mtbs/has_ics209) that AUC is blind to,
         plus dense monotone leaks (burn_days, 7.1x).
      2. PRESENCE PRECISION for sparse/binary columns (<20% present or <=2
         values): P(y=1 | present) / base_rate > presence_lift_max.
      3. UNIVARIATE AUC > auc_max, as a belt for monotone leaks whose AP
         lift is diluted by partial coverage.

    Run it on a frame that INCLUDES candidate columns (the raw frame with the
    banned columns joined back) -- running only on the post-ban matrix is
    circular and detects nothing by construction.
    With enforce=True, a flagged column raises instead of merely reporting.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, dtype=int)
    base = max(y.mean(), 1e-9)
    flags, table = [], []
    for c in X.columns:
        v = X[c]
        if str(v.dtype) == "category":
            continue
        vals = pd.to_numeric(v, errors="coerce").to_numpy(dtype="float64")
        present = np.isfinite(vals)
        if present.sum() < 100 or len(np.unique(y)) < 2:
            continue
        # criterion 1: single-feature ranking AP (missing ranked last; sklearn
        # rejects infinities, so "last" is one unit below the observed minimum)
        lo_fill = np.nanmin(vals[present]) - 1.0
        hi_fill = np.nanmax(vals[present]) + 1.0
        ap_up = average_precision_score(y, np.where(present, vals, lo_fill))
        ap_dn = average_precision_score(y, np.where(present, -vals, -hi_fill))
        ap1 = max(ap_up, ap_dn)   # either direction
        lift = ap1 / base
        # criterion 2: presence precision for sparse columns
        presence_rate = present.mean()
        p_given = float(y[present].mean()) if present.any() else 0.0
        presence_lift = p_given / base
        sparse = presence_rate < 0.20 or pd.Series(vals[present]).nunique() <= 2
        # criterion 3: univariate AUC on present rows (dense monotone leaks)
        auc = 0.5
        if present.sum() >= 100 and len(np.unique(y[present])) == 2:
            auc = roc_auc_score(y[present], vals[present])
            auc = max(auc, 1 - auc)
        fired = (lift > ap_lift_max
                 or (sparse and presence_lift > presence_lift_max)
                 or auc > auc_max)
        row = {"feature": c, "ap_lift": round(float(lift), 2),
               "auc": round(float(auc), 3),
               "presence_rate": round(float(presence_rate), 4),
               "presence_lift": round(float(presence_lift), 2) if sparse else None,
               "flagged": bool(fired)}
        table.append(row)
        if fired:
            flags.append(row)
    table.sort(key=lambda r: -r["ap_lift"])
    result = {"ap_lift_max": ap_lift_max, "presence_lift_max": presence_lift_max,
              "flagged": flags, "top": table[:8], "passed": not flags}
    if enforce and flags:
        raise AssertionError(f"leak tripwire v2 fired on: {[f['feature'] for f in flags]}")
    return result
