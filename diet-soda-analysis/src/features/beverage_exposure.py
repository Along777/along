"""Engineer diet soft drink / SSB exposure from DR1IFF + WWEIA categories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import get_paths, load_config
from src.data.load_xpt import read_xpt

# Grams per ~12 fl oz can
GRAMS_PER_SERVING = 355.0

DIET_SOFT = 7102
DIET_SPORT = 7104
OTHER_DIET = 7106
SOFT_DRINKS = 7202
FRUIT_DRINKS = 7204
SPORT_ENERGY = 7206
WATER_CATS = {7702, 7704}  # tap + bottled plain water


def load_wweia_map(cycle_years: str, xlsx_path: Path) -> pd.DataFrame:
    """Load food_code → category_number map from USDA WWEIA Excel."""
    xl = pd.ExcelFile(xlsx_path)
    sheet = next(s for s in xl.sheet_names if "foodcat" in s.lower() or "FNDDS" in s)
    cat = pd.read_excel(xlsx_path, sheet_name=sheet)
    cat.columns = [str(c).strip().lower().replace(" ", "_") for c in cat.columns]
    code_col = next(c for c in cat.columns if "food_code" in c)
    num_col = next(c for c in cat.columns if "category_number" in c)
    out = cat[[code_col, num_col]].copy()
    out.columns = ["food_code", "category_number"]
    out["food_code"] = pd.to_numeric(out["food_code"], errors="coerce")
    out["category_number"] = pd.to_numeric(out["category_number"], errors="coerce")
    out = out.dropna().astype({"food_code": "int64", "category_number": "int64"})
    out["cycle"] = cycle_years
    return out


def _iff_person_day(
    iff: pd.DataFrame,
    wweia: pd.DataFrame,
    code_col: str,
    gram_col: str,
    day_label: str,
) -> pd.DataFrame:
    """Aggregate beverage grams by SEQN for one recall day."""
    df = iff[["SEQN", code_col, gram_col]].copy()
    df["food_code"] = pd.to_numeric(df[code_col], errors="coerce")
    df["grams"] = pd.to_numeric(df[gram_col], errors="coerce").fillna(0.0)
    df = df.merge(wweia[["food_code", "category_number"]], on="food_code", how="left")

    def _sum_cats(cats: set[int] | int) -> pd.Series:
        if isinstance(cats, int):
            cats = {cats}
        mask = df["category_number"].isin(cats)
        return df.loc[mask].groupby("SEQN")["grams"].sum()

    # FNDDS soft drinks 924xxxxx
    df["is_924"] = (df["food_code"] // 100_000) == 924

    g = df.groupby("SEQN", as_index=False).size().rename(columns={"size": f"n_foods_{day_label}"})
    g = g.set_index("SEQN")
    g[f"asb_g_{day_label}"] = _sum_cats(DIET_SOFT).reindex(g.index).fillna(0.0)
    g[f"ssb_g_{day_label}"] = _sum_cats(SOFT_DRINKS).reindex(g.index).fillna(0.0)
    g[f"asb_broad_g_{day_label}"] = _sum_cats({DIET_SOFT, DIET_SPORT, OTHER_DIET}).reindex(g.index).fillna(0.0)
    g[f"ssb_broad_g_{day_label}"] = _sum_cats({SOFT_DRINKS, FRUIT_DRINKS, SPORT_ENERGY}).reindex(g.index).fillna(0.0)
    g[f"water_g_{day_label}"] = _sum_cats(WATER_CATS).reindex(g.index).fillna(0.0)
    g[f"soft_924_g_{day_label}"] = df.loc[df["is_924"]].groupby("SEQN")["grams"].sum().reindex(g.index).fillna(0.0)
    return g.reset_index()


def diet_keyword_codes(drxfcd: pd.DataFrame) -> set[int]:
    """FNDDS codes whose descriptions look like diet soft drinks (sensitivity)."""
    desc_col = next((c for c in ("DRXFCLD", "DRXFCSD", "DRXFDLD") if c in drxfcd.columns), None)
    if desc_col is None or "DRXFDCD" not in drxfcd.columns:
        return set()
    d = drxfcd[["DRXFDCD", desc_col]].copy()
    d[desc_col] = d[desc_col].astype(str).str.lower()
    mask = d[desc_col].str.contains(
        r"soft drink.*diet|diet.*soft drink|cola,\s*diet|diet,\s*cola|soft drink, cola, diet",
        regex=True,
        na=False,
    )
    return set(pd.to_numeric(d.loc[mask, "DRXFDCD"], errors="coerce").dropna().astype(int))


def person_exposure_for_cycle(
    cycle_years: str,
    suffix: str,
    raw_nhanes: Path,
    wweia: pd.DataFrame,
) -> pd.DataFrame:
    """Person-level exposure features for one NHANES cycle."""
    dr1 = read_xpt(raw_nhanes / cycle_years / f"DR1IFF_{suffix}.xpt")
    exp = _iff_person_day(dr1, wweia, "DR1IFDCD", "DR1IGRMS", "d1")

    # Day-2 if present
    p2 = raw_nhanes / cycle_years / f"DR2IFF_{suffix}.xpt"
    if p2.exists() and p2.stat().st_size > 0:
        dr2 = read_xpt(p2)
        exp2 = _iff_person_day(dr2, wweia, "DR2IFDCD", "DR2IGRMS", "d2")
        exp = exp.merge(exp2, on="SEQN", how="left")
    else:
        exp["asb_g_d2"] = pd.NA

    # Keyword sensitivity Day-1
    fcd_path = raw_nhanes / cycle_years / f"DRXFCD_{suffix}.xpt"
    kw_codes: set[int] = set()
    if fcd_path.exists():
        fcd = read_xpt(fcd_path)
        kw_codes = diet_keyword_codes(fcd)
    if kw_codes:
        codes = pd.to_numeric(dr1["DR1IFDCD"], errors="coerce")
        grams = pd.to_numeric(dr1["DR1IGRMS"], errors="coerce").fillna(0.0)
        tmp = pd.DataFrame({"SEQN": dr1["SEQN"], "g": grams, "code": codes})
        tmp = tmp[tmp["code"].isin(kw_codes)]
        kw_g = tmp.groupby("SEQN")["g"].sum()
        exp["asb_fndds_kw_g_d1"] = exp["SEQN"].map(kw_g).fillna(0.0)
    else:
        exp["asb_fndds_kw_g_d1"] = 0.0

    # Primary Day-1 indicators
    exp["asb_any_d1"] = (exp["asb_g_d1"] > 0).astype(int)
    exp["ssb_any_d1"] = (exp["ssb_g_d1"] > 0).astype(int)
    exp["asb_broad_any_d1"] = (exp["asb_broad_g_d1"] > 0).astype(int)
    exp["ssb_broad_any_d1"] = (exp["ssb_broad_g_d1"] > 0).astype(int)
    exp["asb_fndds_kw_any_d1"] = (exp["asb_fndds_kw_g_d1"] > 0).astype(int)
    exp["asb_serv_d1"] = exp["asb_g_d1"] / GRAMS_PER_SERVING
    exp["ssb_serv_d1"] = exp["ssb_g_d1"] / GRAMS_PER_SERVING

    def _group(row) -> str:
        a, s = row["asb_any_d1"], row["ssb_any_d1"]
        if a and s:
            return "Both"
        if a:
            return "ASB-only"
        if s:
            return "SSB-only"
        return "Neither"

    exp["bev_group"] = exp.apply(_group, axis=1)
    soft = exp["asb_g_d1"] + exp["ssb_g_d1"]
    exp["asb_share_soft"] = soft.where(soft > 0, pd.NA)
    exp.loc[soft > 0, "asb_share_soft"] = exp.loc[soft > 0, "asb_g_d1"] / soft[soft > 0]

    if "asb_g_d2" in exp.columns:
        exp["asb_either_day"] = (
            (exp["asb_g_d1"].fillna(0) > 0) | (exp["asb_g_d2"].fillna(0) > 0)
        ).astype(int)
    else:
        exp["asb_either_day"] = exp["asb_any_d1"]

    exp["cycle"] = cycle_years
    return exp


def build_all_exposures(cfg: dict | None = None) -> pd.DataFrame:
    """Stack person-level exposures across configured cycles."""
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    frames = []
    for cycle in cfg["cycles"]:
        years = cycle["years"]
        suffix = cycle["suffix"]
        # find wweia xlsx
        wdir = paths["raw_wweia"] / years
        xlsx = next(wdir.glob("WWEIA*.xlsx"), None)
        if xlsx is None:
            raise FileNotFoundError(f"No WWEIA xlsx for {years} in {wdir}")
        wmap = load_wweia_map(years, xlsx)
        print(f"Exposure {years} ...", flush=True)
        frames.append(person_exposure_for_cycle(years, suffix, paths["raw_nhanes"], wmap))
    return pd.concat(frames, ignore_index=True)
