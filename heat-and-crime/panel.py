"""
Single source of truth for the Chicago heat–crime daily panel.
"""
from __future__ import annotations

from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

VIOLENT = [
    "BATTERY",
    "ASSAULT",
    "ROBBERY",
    "HOMICIDE",
    "CRIM SEXUAL ASSAULT",
    "CRIMINAL SEXUAL ASSAULT",
]
PROPERTY = ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE"]
OUTCOMES = ["total", "violent", "property"]

FETCH_START = "2010-01-01"
WIN_START, WIN_END = "2015-01-01", "2025-12-31"
CTRL_M2 = "C(ym) + C(dow) + holiday + rain + snow_day"
HAC_DEFAULT = 9  # Model Lab selected


def heat_index_f(temp_f: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    NWS Rothfusz regression heat index (°F), with simple blend below 80°F.
    temp_f: air temperature °F; rh: relative humidity %.
    """
    T = np.asarray(temp_f, dtype=float)
    R = np.asarray(rh, dtype=float)
    # Steadman simple formula for moderate conditions
    hi_simple = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))
    # Full Rothfusz
    hi = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 0.00683783 * T * T
        - 0.05481717 * R * R
        + 0.00122874 * T * T * R
        + 0.00085282 * T * R * R
        - 0.00000199 * T * T * R * R
    )
    # adjustments omitted for city-day robustness; blend when HI_simple < 80
    out = np.where(hi_simple < 80, 0.5 * (T + hi_simple), hi)
    return out


def build_panel(
    data_dir: Path | None = None,
    winsor_crime: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (dfa analysis 2015-2025, dffull 2010-2025).
    """
    data_dir = Path(data_dir) if data_dir else DATA
    crime = pd.read_csv(data_dir / "chicago_crime_daily_by_type.csv")
    crime["day"] = pd.to_datetime(crime["day"])
    wx = pd.read_csv(data_dir / "chicago_weather_daily.csv")
    wx["date"] = pd.to_datetime(wx["date"])
    wx = wx.rename(
        columns={
            "temperature_2m_max": "tmax",
            "temperature_2m_mean": "tmean",
            "apparent_temperature_max": "app_tmax",
            "relative_humidity_2m_mean": "rh",
            "precipitation_sum": "precip_mm",
            "snowfall_sum": "snow_cm",
        }
    )

    # category + component outcomes
    crime["cat"] = np.select(
        [crime["primary_type"].isin(VIOLENT), crime["primary_type"].isin(PROPERTY)],
        ["violent", "property"],
        default="other",
    )
    crime["is_battery"] = (crime["primary_type"] == "BATTERY").astype(int)
    crime["is_assault"] = (crime["primary_type"] == "ASSAULT").astype(int)
    crime["is_robbery"] = (crime["primary_type"] == "ROBBERY").astype(int)
    crime["is_homicide"] = (crime["primary_type"] == "HOMICIDE").astype(int)
    crime["is_theft"] = (crime["primary_type"] == "THEFT").astype(int)

    daily = crime.pivot_table(
        index="day", columns="cat", values="n", aggfunc="sum", fill_value=0
    )
    daily["total"] = daily.sum(axis=1)
    other = daily["other"].copy() if "other" in daily.columns else 0
    daily = daily.drop(columns=[c for c in ["other"] if c in daily.columns])

    for name, flag in [
        ("battery", "is_battery"),
        ("assault", "is_assault"),
        ("robbery", "is_robbery"),
        ("homicide", "is_homicide"),
        ("theft", "is_theft"),
    ]:
        s = (
            crime.loc[crime[flag] == 1]
            .groupby("day")["n"]
            .sum()
        )
        daily[name] = s

    cal = pd.date_range(FETCH_START, "2025-12-31", freq="D")
    daily = daily.reindex(cal).fillna(0).astype(int)
    if isinstance(other, pd.Series):
        other = other.reindex(cal).fillna(0).astype(int)

    df = daily.join(wx.set_index("date"), how="left")
    assert df[["tmax", "tmean"]].notna().all().all()
    df["other"] = other

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["ym"] = df.index.strftime("%Y-%m")
    df["week_id"] = df.index.to_period("W").astype(str)
    us_hol = holidays_lib.US(years=range(2010, 2026), observed=True)
    df["holiday"] = [int(d in us_hol) for d in df.index.date]

    df["tmax10"] = df["tmax"] / 10.0
    df["tmean10"] = df["tmean"] / 10.0
    df["app_tmax10"] = df["app_tmax"] / 10.0
    df["hi"] = heat_index_f(df["tmax"].values, df["rh"].values)
    df["hi10"] = df["hi"] / 10.0
    df["rh_c"] = df["rh"] - df["rh"].mean()
    df["rain"] = (df["precip_mm"] >= 1.0).astype(int)
    df["snow_day"] = (df["snow_cm"] > 0).astype(int)
    df["unrest"] = ((df.index >= "2020-05-29") & (df.index <= "2020-06-07")).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_nyd"] = ((df.index.month == 1) & (df.index.day == 1)).astype(int)
    df["tmax10_x_weekend"] = df["tmax10"] * df["weekend"]
    df["tmax10_x_summer"] = df["tmax10"] * df["is_summer"]
    df["tmax10_x_rh"] = df["tmax10"] * df["rh_c"]

    # lags 1..7 and leads 1..3 (future temperature = placebo)
    for k in range(1, 8):
        df[f"tmax10_L{k}"] = df["tmax10"].shift(k)
    for k in range(1, 4):
        df[f"tmax10_F{k}"] = df["tmax10"].shift(-k)
    df["tmax10_roll3"] = df["tmax10"].rolling(3, min_periods=1).mean()

    # within-ym demeaned helpers for standardized effects
    df["tmax10_dm"] = df["tmax10"] - df.groupby("ym")["tmax10"].transform("mean")
    df["tmax_dm"] = df["tmax"] - df.groupby("ym")["tmax"].transform("mean")

    if winsor_crime:
        for col in ["total", "violent", "property", "battery"]:
            lo, hi = df[col].quantile(0.001), df[col].quantile(0.999)
            df[col] = df[col].clip(lo, hi)

    dffull = df
    dfa = df.loc[WIN_START:WIN_END].copy()
    p90 = float(dfa["tmax"].quantile(0.90))
    dfa["heat_p90"] = (dfa["tmax"] >= p90).astype(int)
    dfa["heat90"] = (dfa["tmax"] >= 90).astype(int)
    dfa["p90_threshold"] = p90
    assert len(dfa) == 4018
    assert (dfa["violent"] + dfa["property"] + dfa["other"] == dfa["total"]).all()
    return dfa, dffull
