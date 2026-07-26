"""Best-effort download of Clio-Infra / Zijdeman life expectancy at birth."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.paths import RAW, ensure_dirs

UA = "age-myth-historical-le-db/0.1 (research; Clio-Infra acquisition)"

# URLs change; probe known public endpoints
CANDIDATES = [
    # IISH Dataverse-style (file id may change)
    "https://datasets.iisg.amsterdam/api/access/datafile/181",
    "https://datasets.iisg.amsterdam/file.xhtml?fileId=181&version=1.1",
]


def main() -> None:
    ensure_dirs()
    out_dir = RAW / "clio_infra"
    out_dir.mkdir(parents=True, exist_ok=True)
    log: dict = {"attempts": [], "success": None, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    headers = {"User-Agent": UA}

    for url in CANDIDATES:
        print(f"Trying {url}")
        try:
            r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "")
            # Accept spreadsheet or octet-stream
            if r.status_code == 200 and len(r.content) > 1000:
                if "html" in ctype.lower() and b"PK" not in r.content[:4]:
                    log["attempts"].append({"url": url, "ok": False, "reason": f"HTML response ({ctype})"})
                    continue
                # guess extension
                ext = ".xlsx"
                if r.content[:2] == b"PK":
                    ext = ".xlsx"
                elif b"," in r.content[:200]:
                    ext = ".csv"
                dest = out_dir / f"zijdeman_life_expectancy_at_birth{ext}"
                dest.write_bytes(r.content)
                log["success"] = {
                    "url": url,
                    "path": str(dest),
                    "bytes": len(r.content),
                    "content_type": ctype,
                }
                print(f"OK -> {dest} ({len(r.content):,} bytes)")
                break
            log["attempts"].append({"url": url, "ok": False, "status": r.status_code, "bytes": len(r.content)})
        except Exception as e:  # noqa: BLE001
            log["attempts"].append({"url": url, "ok": False, "error": str(e)})
            print(f"  failed: {e}")

    if not log["success"]:
        manual = out_dir / "MANUAL_DOWNLOAD.md"
        manual.write_text(
            "# Clio-Infra manual download\n\n"
            "Automatic download failed. Obtain Zijdeman & Ribeira da Silva (2015)\n"
            "Life Expectancy at Birth (Total):\n\n"
            "- Handle: http://hdl.handle.net/10622/LKYT53\n"
            "- Clio Infra indicator pages / IISH Dataverse\n\n"
            f"Place the file in `{out_dir}` and re-run cleaning when implemented.\n",
            encoding="utf-8",
        )
        print(f"Clio download failed. See {manual}")

    (out_dir / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
