"""Cardiometabolic outcomes from exam/lab files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import get_paths, load_config
from src.data.load_xpt import read_xpt


def _mean_bp(bpx: pd.DataFrame, prefix: str) -> pd.Series:
    """Mean of valid BP readings. Diastolic 0 = NHANES 'cannot obtain' → treat as missing."""
    cols = [c for c in bpx.columns if c.startswith(prefix) and c[-1].isdigit()]
    if not cols:
        return pd.Series(np.nan, index=bpx.index)
    vals = bpx[cols].apply(pd.to_numeric, errors="coerce")
    if prefix.startswith("BPXDI"):
        vals = vals.mask(vals <= 0)  # 0 mmHg diastolic is not a real measurement
    if prefix.startswith("BPXSY"):
        vals = vals.mask(vals <= 0)
    return vals.mean(axis=1, skipna=True)


def person_outcomes_for_cycle(cycle_years: str, suffix: str, raw: Path) -> pd.DataFrame:
    base = read_xpt(raw / cycle_years / f"DEMO_{suffix}.xpt")[["SEQN"]].copy()

    def _left(stem: str, cols: list[str]) -> None:
        nonlocal base
        path = raw / cycle_years / f"{stem}_{suffix}.xpt"
        if not path.exists() or path.stat().st_size == 0:
            for c in cols:
                if c != "SEQN" and c not in base.columns:
                    base[c] = np.nan
            return
        df = read_xpt(path)
        keep = [c for c in cols if c in df.columns]
        base = base.merge(df[keep], on="SEQN", how="left")

    _left("BMX", ["SEQN", "BMXBMI", "BMXWAIST"])
    _left("GHB", ["SEQN", "LBXGH"])
    _left("HDL", ["SEQN", "LBDHDD"])
    _left("TCHOL", ["SEQN", "LBXTC"])
    _left("TRIGLY", ["SEQN", "LBXTR", "LBDLDL"])

    # Glucose / insulin (2011-12 insulin in GLU)
    glu_path = raw / cycle_years / f"GLU_{suffix}.xpt"
    if glu_path.exists():
        glu = read_xpt(glu_path)
        gcols = [c for c in ["SEQN", "LBXGLU", "LBXIN", "WTSAF2YR"] if c in glu.columns]
        base = base.merge(glu[gcols], on="SEQN", how="left")
    ins_path = raw / cycle_years / f"INS_{suffix}.xpt"
    if ins_path.exists() and ins_path.stat().st_size > 0:
        ins = read_xpt(ins_path)
        if "LBXIN" in ins.columns:
            # prefer INS file if present (H/I/J); fill gaps only
            tmp = ins[["SEQN", "LBXIN"]].rename(columns={"LBXIN": "LBXIN_ins"})
            base = base.merge(tmp, on="SEQN", how="left")
            if "LBXIN" not in base.columns:
                base["LBXIN"] = base["LBXIN_ins"]
            else:
                base["LBXIN"] = base["LBXIN"].fillna(base["LBXIN_ins"])
            base = base.drop(columns=["LBXIN_ins"], errors="ignore")

    bpx_path = raw / cycle_years / f"BPX_{suffix}.xpt"
    if bpx_path.exists():
        bpx = read_xpt(bpx_path)
        bpx = bpx.copy()
        bpx["sbp_mean"] = _mean_bp(bpx, "BPXSY")
        bpx["dbp_mean"] = _mean_bp(bpx, "BPXDI")
        base = base.merge(bpx[["SEQN", "sbp_mean", "dbp_mean"]], on="SEQN", how="left")

    # Rename to analysis names
    base = base.rename(
        columns={
            "BMXBMI": "bmi",
            "BMXWAIST": "waist",
            "LBXGH": "hba1c",
            "LBXGLU": "glucose",
            "LBXIN": "insulin",
            "LBDHDD": "hdl",
            "LBXTC": "tc",
            "LBXTR": "tg",
            "LBDLDL": "ldl",
            "WTSAF2YR": "wtsaf2yr",
        }
    )

    # Derived
    base["obesity"] = (base["bmi"] >= 30).astype("float")
    base.loc[base["bmi"].isna(), "obesity"] = np.nan
    base["hba1c_elevated"] = (base["hba1c"] >= 6.5).astype("float")
    base.loc[base["hba1c"].isna(), "hba1c_elevated"] = np.nan
    base["hypertension_bp"] = ((base["sbp_mean"] >= 130) | (base["dbp_mean"] >= 80)).astype("float")
    base.loc[base["sbp_mean"].isna() & base["dbp_mean"].isna(), "hypertension_bp"] = np.nan
    # HOMA-IR when both present
    base["homa_ir"] = np.where(
        base["glucose"].notna() & base["insulin"].notna(),
        base["glucose"] * base["insulin"] / 405.0,
        np.nan,
    )
    base["cycle"] = cycle_years
    return base


def build_all_outcomes(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    frames = []
    for cycle in cfg["cycles"]:
        print(f"Outcomes {cycle['years']} ...", flush=True)
        frames.append(person_outcomes_for_cycle(cycle["years"], cycle["suffix"], paths["raw_nhanes"]))
    return pd.concat(frames, ignore_index=True)
