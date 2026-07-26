"""Canonical population filters — single source of truth for de-dupe / quality."""
from __future__ import annotations

import pandas as pd

PRIMARY_ALLOWLIST = [
    "SWE",
    "FRANCE:_TOTAL_POPULATION",
    "UK:_ENGLAND_&_WALES_TOTAL_POPULATION",
    "DNK",
    "NOR",
    "NLD",
    "BEL",
    "ITA",
    "CHE",
    "FIN",
    "ISL",
    "USA",
    "AUS",
    "CAN",
    "JPN",
]

LABELS = {
    "SWE": "Sweden",
    "FRANCE:_TOTAL_POPULATION": "France",
    "UK:_ENGLAND_&_WALES_TOTAL_POPULATION": "England & Wales",
    "UK:_SCOTLAND": "Scotland",
    "DNK": "Denmark",
    "NOR": "Norway",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "ITA": "Italy",
    "CHE": "Switzerland",
    "FIN": "Finland",
    "ISL": "Iceland",
    "USA": "United States",
    "AUS": "Australia",
    "CAN": "Canada",
    "JPN": "Japan",
    "ESP": "Spain",
}


def is_duplicate_subseries(region_id: str) -> bool:
    r = str(region_id).upper()
    if "CIVILIAN" in r:
        return True
    if "EAST" in r or "WEST" in r:
        return True
    if "MAORI" in r and "TOTAL" not in r:
        return True
    if "NON-MAORI" in r or "NON_MAORI" in r:
        return True
    return False


def dedupe_hmd(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[~panel["region_id"].map(is_duplicate_subseries)].copy()
    return out


def label_region(region_id: str) -> str:
    return LABELS.get(str(region_id), str(region_id).replace("_", " ")[:40])


def filter_primary(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["region_id"].isin(PRIMARY_ALLOWLIST)].copy()


def hld_quality_slice(hld: pd.DataFrame, mode: str = "gold") -> pd.DataFrame:
    """mode: gold (n_tables==1) | median (all rows already median-collapsed)."""
    d = hld.copy()
    if mode == "gold" and "n_tables" in d.columns:
        d = d[d["n_tables"] == 1].copy()
        d["quality"] = "hld_n_tables_1"
    else:
        d["quality"] = "hld_median"
    return d
