"""Download Our World in Data life expectancy series."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.paths import RAW, ensure_dirs

UA = "age-myth-historical-le-db/0.1 (research; local reproducible pipeline)"

SERIES = {
    "life-expectancy": {
        "url": "https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "life-expectancy.csv",
        "source_id": "owid_le_longrun",
    },
    "life-expectancy-at-age-15": {
        "url": "https://ourworldindata.org/grapher/life-expectancy-at-age-15.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "life-expectancy-at-age-15.csv",
        "source_id": "owid_le_age15",
    },
    "life-expectancy-hmd-unwpp": {
        "url": "https://ourworldindata.org/grapher/life-expectancy-hmd-unwpp.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "life-expectancy-hmd-unwpp.csv",
        "source_id": "owid_le_hmd_unwpp",
    },
    "infant-mortality": {
        "url": "https://ourworldindata.org/grapher/infant-mortality.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "infant-mortality.csv",
        "source_id": "owid_imr",
    },
    "child-mortality": {
        "url": "https://ourworldindata.org/grapher/child-mortality.csv?v=1&csvType=full&useColumnShortNames=false",
        "filename": "child-mortality.csv",
        "source_id": "owid_u5mr",
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_one(name: str, meta: dict, out_dir: Path) -> dict:
    headers = {"User-Agent": UA}
    r = requests.get(meta["url"], headers=headers, timeout=120)
    r.raise_for_status()
    path = out_dir / meta["filename"]
    path.write_bytes(r.content)
    # companion metadata when available
    meta_url = meta["url"].replace(".csv?", ".metadata.json?").replace(
        "grapher/life-expectancy.csv", "grapher/life-expectancy.metadata.json"
    )
    # simpler: derive metadata URL from grapher slug
    slug = name
    murl = f"https://ourworldindata.org/grapher/{slug}.metadata.json?v=1&csvType=full&useColumnShortNames=false"
    meta_path = None
    try:
        mr = requests.get(murl, headers=headers, timeout=60)
        if mr.ok:
            meta_path = out_dir / f"{slug}.metadata.json"
            meta_path.write_bytes(mr.content)
    except requests.RequestException:
        pass

    entry = {
        "name": name,
        "source_id": meta["source_id"],
        "url": meta["url"],
        "path": str(path.relative_to(out_dir.parent.parent.parent) if False else path),
        "bytes": len(r.content),
        "sha256": _sha256(r.content),
        "status_code": r.status_code,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(meta_path) if meta_path else None,
    }
    print(f"OK {name}: {entry['bytes']:,} bytes -> {path}")
    return entry


def main() -> None:
    ensure_dirs()
    out_dir = RAW / "owid"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"series": [], "errors": []}
    for name, meta in SERIES.items():
        try:
            log["series"].append(download_one(name, meta, out_dir))
        except Exception as e:  # noqa: BLE001 — log and continue other series
            print(f"FAIL {name}: {e}")
            log["errors"].append({"name": name, "error": str(e)})
    log_path = out_dir / "download_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote {log_path}")
    if log["errors"] and not log["series"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
