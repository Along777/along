"""Download Eurostat demo_mlexpec (life expectancy by age and sex)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.paths import RAW, ensure_dirs

UA = "age-myth-historical-le-db/0.1 (research; Eurostat demo_mlexpec)"
# SDMX-CSV bulk for life expectancy by age and sex
URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/demo_mlexpec"
    "?format=SDMX-CSV&compressed=false"
)


def main() -> None:
    ensure_dirs()
    out = RAW / "eurostat"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "demo_mlexpec.csv"
    print(f"Downloading Eurostat demo_mlexpec (~36MB) ...")
    r = requests.get(URL, headers={"User-Agent": UA}, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    log = {
        "url": URL,
        "path": str(dest),
        "bytes": len(r.content),
        "sha256": hashlib.sha256(r.content).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"OK {dest} ({len(r.content):,} bytes)")


if __name__ == "__main__":
    main()
