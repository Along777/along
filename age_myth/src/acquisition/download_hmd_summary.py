"""Download HMD public summary Excel files (no registration required)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.paths import RAW, ensure_dirs

UA = "age-myth-historical-le-db/0.1 (research; HMD public summary)"

FILES = {
    "hmd_summary_IMR.xlsx": (
        "https://www.mortality.org/File/GetDocument/Public/HMD_summary/hmd_summary_IMR.xlsx"
    ),
    "hmd_summary_px_0_to_65.xlsx": (
        "https://www.mortality.org/File/GetDocument/Public/HMD_summary/hmd_summary_px_0_to_65.xlsx"
    ),
    "hmd_summary_ex_0_65_80.xlsx": (
        "https://www.mortality.org/File/GetDocument/Public/HMD_summary/hmd_summary_ex_0_65_80.xlsx"
    ),
}


def main() -> None:
    ensure_dirs()
    out = RAW / "hmd_summary"
    out.mkdir(parents=True, exist_ok=True)
    log: dict = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "errors": [],
    }
    for name, url in FILES.items():
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
            r.raise_for_status()
            path = out / name
            path.write_bytes(r.content)
            log["files"].append(
                {
                    "name": name,
                    "url": url,
                    "bytes": len(r.content),
                    "sha256": hashlib.sha256(r.content).hexdigest(),
                }
            )
            print(f"OK {name}: {len(r.content):,} bytes")
        except Exception as e:  # noqa: BLE001
            log["errors"].append({"name": name, "error": str(e)})
            print(f"FAIL {name}: {e}")
    (out / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    if not log["files"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
