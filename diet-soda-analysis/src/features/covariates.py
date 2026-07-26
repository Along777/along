"""Demographics and lifestyle covariates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import get_paths, load_config
from src.data.load_xpt import read_xpt


def _smoking_status(smq: pd.DataFrame) -> pd.Series:
    """0 never, 1 former, 2 current (simple)."""
    ever = pd.to_numeric(smq.get("SMQ020"), errors="coerce")
    now = pd.to_numeric(smq.get("SMQ040"), errors="coerce")
    out = pd.Series(np.nan, index=smq.index, dtype="float")
    # SMQ020: 1=yes smoked 100 cigs, 2=no
    out.loc[ever == 2] = 0  # never
    # SMQ040: 1 every day, 2 some days, 3 not at all
    out.loc[(ever == 1) & (now == 3)] = 1  # former
    out.loc[(ever == 1) & (now.isin([1, 2]))] = 2  # current
    return out


def person_covariates_for_cycle(cycle_years: str, suffix: str, raw: Path) -> pd.DataFrame:
    demo = read_xpt(raw / cycle_years / f"DEMO_{suffix}.xpt")
    cols = [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "SDMVPSU",
        "SDMVSTRA",
        "WTMEC2YR",
        "WTINT2YR",
        "RIDEXPRG",
    ]
    keep = [c for c in cols if c in demo.columns]
    out = demo[keep].copy()

    # Diet totals
    tot_path = raw / cycle_years / f"DR1TOT_{suffix}.xpt"
    if tot_path.exists():
        tot = read_xpt(tot_path)
        tcols = [c for c in ["SEQN", "DR1TKCAL", "DR1DRSTZ"] if c in tot.columns]
        out = out.merge(tot[tcols], on="SEQN", how="left")

    # Smoking
    smq_path = raw / cycle_years / f"SMQ_{suffix}.xpt"
    if smq_path.exists():
        smq = read_xpt(smq_path)
        smq = smq.copy()
        smq["smoking_status"] = _smoking_status(smq)
        out = out.merge(smq[["SEQN", "smoking_status"]], on="SEQN", how="left")

    # Sedentary minutes PAD680 if present
    paq_path = raw / cycle_years / f"PAQ_{suffix}.xpt"
    if paq_path.exists():
        paq = read_xpt(paq_path)
        if "PAD680" in paq.columns:
            out = out.merge(paq[["SEQN", "PAD680"]], on="SEQN", how="left")
            out = out.rename(columns={"PAD680": "sedentary_min"})

    # Diabetes self-report
    diq_path = raw / cycle_years / f"DIQ_{suffix}.xpt"
    if diq_path.exists():
        diq = read_xpt(diq_path)
        if "DIQ010" in diq.columns:
            out = out.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
            # 1=yes, 2=no, 3=borderline
            out["diabetes_sr"] = (pd.to_numeric(out["DIQ010"], errors="coerce") == 1).astype("float")
            out.loc[pd.to_numeric(out["DIQ010"], errors="coerce").isna(), "diabetes_sr"] = np.nan

    # Cancer ever
    mcq_path = raw / cycle_years / f"MCQ_{suffix}.xpt"
    if mcq_path.exists():
        mcq = read_xpt(mcq_path)
        if "MCQ220" in mcq.columns:
            out = out.merge(mcq[["SEQN", "MCQ220"]], on="SEQN", how="left")
            out["cancer_ever"] = (pd.to_numeric(out["MCQ220"], errors="coerce") == 1).astype("float")
            out.loc[~pd.to_numeric(out["MCQ220"], errors="coerce").isin([1, 2]), "cancer_ever"] = np.nan

    # Alcohol — drinks past year rough if available
    alq_path = raw / cycle_years / f"ALQ_{suffix}.xpt"
    if alq_path.exists():
        alq = read_xpt(alq_path)
        # ALQ121 / ALQ130 vary by cycle; keep first available frequency-like
        for c in ("ALQ121", "ALQ130", "ALQ101"):
            if c in alq.columns:
                out = out.merge(alq[["SEQN", c]].rename(columns={c: "alcohol_var"}), on="SEQN", how="left")
                break

    out = out.rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "sex",  # 1 male 2 female
            "RIDRETH3": "race_eth",
            "DMDEDUC2": "education",
            "INDFMPIR": "pir",
            "DR1TKCAL": "total_kcal_d1",
            "DR1DRSTZ": "diet_recall_status",
            "RIDEXPRG": "pregnancy_status",
        }
    )
    out["female"] = (out["sex"] == 2).astype("float")

    # NHANES special missing codes → NaN (codebook: 7/9 refused/DK; 77/99; 7777/9999)
    if "education" in out.columns:
        out["education"] = pd.to_numeric(out["education"], errors="coerce")
        out.loc[out["education"].isin([7, 9]), "education"] = np.nan
    if "sedentary_min" in out.columns:
        out["sedentary_min"] = pd.to_numeric(out["sedentary_min"], errors="coerce")
        out.loc[out["sedentary_min"].isin([7777, 9999]), "sedentary_min"] = np.nan
        out.loc[out["sedentary_min"] > 1440, "sedentary_min"] = np.nan  # impossible minutes/day
    if "pir" in out.columns:
        out["pir"] = pd.to_numeric(out["pir"], errors="coerce")
        # PIR already continuous; leave as is

    out["cycle"] = cycle_years
    return out


def build_all_covariates(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    frames = []
    for cycle in cfg["cycles"]:
        print(f"Covariates {cycle['years']} ...", flush=True)
        frames.append(person_covariates_for_cycle(cycle["years"], cycle["suffix"], paths["raw_nhanes"]))
    return pd.concat(frames, ignore_index=True)
