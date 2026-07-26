"""Analytic sample flags and exclusion flowchart."""

from __future__ import annotations

import pandas as pd


def add_sample_flags(df: pd.DataFrame, min_age: int = 20) -> pd.DataFrame:
    """Add boolean inclusion flags; does not drop rows."""
    out = df.copy()
    out["flag_age"] = out["age"] >= min_age
    # RIDEXPRG: 1 yes pregnant, 2 no, 3 uncertain — exclude definite pregnancy
    preg = pd.to_numeric(out.get("pregnancy_status"), errors="coerce")
    out["flag_not_pregnant"] = preg.isna() | (preg != 1)
    # Reliable Day-1 dietary recall only (DR1DRSTZ == 1). Missing status excluded.
    st = pd.to_numeric(out.get("diet_recall_status"), errors="coerce")
    out["flag_diet_ok"] = st == 1
    # Has MEC weight
    out["flag_mec_weight"] = pd.to_numeric(out.get("WTMEC2YR"), errors="coerce").fillna(0) > 0
    # Has exposure row (Day-1 IFF join)
    out["flag_has_exposure"] = out.get("bev_group").notna() if "bev_group" in out.columns else False

    out["in_analytic"] = (
        out["flag_age"]
        & out["flag_not_pregnant"]
        & out["flag_diet_ok"]
        & out["flag_mec_weight"]
        & out["flag_has_exposure"]
    )
    out["in_fasting"] = out["in_analytic"] & out.get("wtsaf2yr", pd.Series(dtype=float)).fillna(0).gt(0)
    if "wtsaf2yr" not in out.columns:
        out["in_fasting"] = False
    return out


def exclusion_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Sequential exclusion counts for documentation."""
    steps = []
    n = len(df)
    steps.append({"step": "All merged rows", "n": n})
    m = df["flag_age"]
    steps.append({"step": "Age >= 20", "n": int(m.sum())})
    m = m & df["flag_not_pregnant"]
    steps.append({"step": "+ not pregnant", "n": int(m.sum())})
    m = m & df["flag_mec_weight"]
    steps.append({"step": "+ MEC weight > 0", "n": int(m.sum())})
    m = m & df["flag_has_exposure"]
    steps.append({"step": "+ Day-1 dietary exposure", "n": int(m.sum())})
    m = m & df["flag_diet_ok"]
    steps.append({"step": "+ diet recall status OK (or missing)", "n": int(m.sum())})
    steps.append({"step": "Analytic sample (in_analytic)", "n": int(df["in_analytic"].sum())})
    if "in_fasting" in df.columns:
        steps.append({"step": "Fasting subsample", "n": int(df["in_fasting"].sum())})
    return pd.DataFrame(steps)
