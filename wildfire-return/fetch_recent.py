from __future__ import annotations

"""Recency layer: WFIGS large-fire aggregates for 2021-2025.

FPA FOD-Attributes ends in 2020. To let the trend charts speak to the present,
this script pulls ANNUAL AGGREGATES of large wildfires (>= 1,000 acres) from the
NIFC WFIGS Interagency Fire Perimeters service -- counts and acres only, grouped
server-side, so no feature pagination and no big downloads. The result is a
small committed cache; article charts draw it as a dashed, visibly-different
segment and never mix it into the FPA-FOD attribute ML.

Honesty notes baked into the output:
- WFIGS is a different reporting system than FPA-FOD (different completeness);
  we only compare like-for-like: fires >= 1,000 acres in both sources.
- Wildfires only (attr_IncidentTypeCategory = 'WF'; prescribed burns excluded).
- Complex children are excluded when flagged, to avoid double counting; both
  filtered and unfiltered sums are stored so the choice is auditable.
- National scope excludes AK/HI/PR to match FPA FOD-Attributes' CONUS coverage.

Usage:  python fetch_recent.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

SERVICE = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
           "WFIGS_Interagency_Perimeters/FeatureServer/0/query")
YEARS = range(2021, 2026)
LARGE_ACRES = 1000

NON_CONUS = ("US-AK", "US-HI", "US-PR")

STATS = json.dumps([
    {"statisticType": "count", "onStatisticField": "OBJECTID", "outStatisticFieldName": "n_fires"},
    {"statisticType": "sum", "onStatisticField": "attr_IncidentSize", "outStatisticFieldName": "acres_incident"},
    {"statisticType": "sum", "onStatisticField": "poly_GISAcres", "outStatisticFieldName": "acres_poly"},
])


def where_clause(year: int, scope: str, exclude_cpx_children: bool) -> str:
    parts = [
        "attr_IncidentTypeCategory = 'WF'",
        f"attr_IncidentSize >= {LARGE_ACRES}",
        f"attr_FireDiscoveryDateTime >= TIMESTAMP '{year}-01-01 00:00:00'",
        f"attr_FireDiscoveryDateTime < TIMESTAMP '{year + 1}-01-01 00:00:00'",
    ]
    if scope == "conus":
        parts.append("(attr_POOState NOT IN ('" + "','".join(NON_CONUS) + "') OR attr_POOState IS NULL)")
    elif scope == "CA":
        parts.append("attr_POOState = 'US-CA'")
    else:
        raise ValueError(scope)
    if exclude_cpx_children:
        parts.append("(attr_IsCpxChild = 0 OR attr_IsCpxChild IS NULL)")
    return " AND ".join(parts)


def query_stats(session: requests.Session, where: str) -> dict:
    r = session.get(SERVICE, params={"where": where, "outStatistics": STATS,
                                     "returnGeometry": "false", "f": "json"}, timeout=(10, 120))
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise SystemExit(f"ArcGIS error for where=[{where}]: {payload['error']}")
    attrs = payload["features"][0]["attributes"]
    return {k.lower(): attrs[k] for k in ("n_fires", "acres_incident", "acres_poly")}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "wildfire-return/1.0 (portfolio project; annual aggregates only)"

    rows, queries = [], []
    for scope in ("conus", "CA"):
        for year in YEARS:
            row: dict = {"scope": scope, "year": year, "large_acres_min": LARGE_ACRES}
            for label, excl in (("", True), ("_incl_cpx_children", False)):
                where = where_clause(year, scope, exclude_cpx_children=excl)
                stats = query_stats(session, where)
                row[f"n_fires{label}"] = stats["n_fires"]
                row[f"acres_incident{label}"] = stats["acres_incident"]
                row[f"acres_poly{label}"] = stats["acres_poly"]
                queries.append({"scope": scope, "year": year, "where": where})
            rows.append(row)
            print(f"{scope:6s} {year}: {row['n_fires']:4d} fires >= {LARGE_ACRES} ac, "
                  f"{row['acres_incident']:>12,.0f} incident acres", flush=True)

    df = pd.DataFrame(rows)
    out_csv = DATA / "recent_annual.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service": SERVICE,
        "source": "NIFC Open Data / WFIGS Interagency Fire Perimeters (public domain)",
        "definition": f"wildfires (WF) with attr_IncidentSize >= {LARGE_ACRES} acres, by discovery year; "
                      "complex children excluded in headline columns; conus scope excludes AK/HI/PR",
        "caveats": [
            "Different reporting system than FPA-FOD; compare only like-for-like large-fire series.",
            "Perimeter-based dataset; completeness below 1,000 acres is poor, above it is good.",
            "acres_incident = sum of reported incident size; acres_poly = sum of GIS perimeter acres.",
        ],
        "queries": queries,
    }
    (DATA / "recent_annual_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_csv} ({len(df)} rows) + recent_annual_meta.json")


if __name__ == "__main__":
    main()
