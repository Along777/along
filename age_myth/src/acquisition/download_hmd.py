"""Download selected HMD country life tables (requires free registration)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

from src.paths import RAW, ensure_dirs

UA = "age-myth-historical-le-db/0.1 (research; HMD acquisition)"

# Priority countries for Phase 1
COUNTRIES = {
    "SWE": "Sweden",
    "GBRTENW": "England and Wales",
    "FRATNP": "France",
    "JPN": "Japan",
}

# Period both-sex 1x1 life tables (path pattern used by HMD file server)
FILE_TEMPLATES = [
    # v6-style document paths (may require session cookie after login)
    "https://www.mortality.org/File/GetDocument/hmd.v6/{code}/STATS/bltper_1x1.txt",
    "https://www.mortality.org/File/GetDocument/hmd.v6/{code}/STATS/fltper_1x1.txt",
    "https://www.mortality.org/File/GetDocument/hmd.v6/{code}/STATS/mltper_1x1.txt",
]


def main() -> None:
    ensure_dirs()
    out_dir = RAW / "hmd"
    out_dir.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("HMD_USER", "").strip()
    password = os.environ.get("HMD_PASSWORD", "").strip()
    log: dict = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "errors": [],
        "auth_present": bool(user and password),
    }

    if not user or not password:
        msg = (
            "HMD credentials not set. Register free at https://www.mortality.org/ then:\n"
            "  $env:HMD_USER = 'you@example.com'\n"
            "  $env:HMD_PASSWORD = 'your-password'\n"
            "Or use the fixture pipeline: python -m src.cleaning.hmd_life_table --fixture"
        )
        print(msg)
        (out_dir / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        (out_dir / "README_AUTH.md").write_text(msg + "\n", encoding="utf-8")
        raise SystemExit(0)

    session = requests.Session()
    session.headers["User-Agent"] = UA
    auth = HTTPBasicAuth(user, password)

    for code, name in COUNTRIES.items():
        country_dir = out_dir / code
        country_dir.mkdir(exist_ok=True)
        for tmpl in FILE_TEMPLATES:
            url = tmpl.format(code=code)
            fname = url.rsplit("/", 1)[-1]
            dest = country_dir / fname
            try:
                r = session.get(url, auth=auth, timeout=120)
                if r.status_code != 200 or len(r.content) < 100:
                    # try without basic auth (cookie/login sites differ)
                    r = session.get(url, timeout=120)
                if r.status_code == 200 and (
                    b"Year" in r.content[:2000] or b"Age" in r.content[:2000]
                ):
                    dest.write_bytes(r.content)
                    log["files"].append(
                        {
                            "country": code,
                            "name": name,
                            "file": fname,
                            "bytes": len(r.content),
                            "url": url,
                        }
                    )
                    print(f"OK {code}/{fname} ({len(r.content):,} bytes)")
                else:
                    err = f"{code}/{fname}: HTTP {r.status_code}, bytes={len(r.content)}"
                    log["errors"].append(err)
                    print(f"FAIL {err}")
            except Exception as e:  # noqa: BLE001
                log["errors"].append(f"{code}/{fname}: {e}")
                print(f"FAIL {code}/{fname}: {e}")

    (out_dir / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'download_log.json'}")
    if not log["files"]:
        print(
            "No HMD files downloaded. HMD may require browser login rather than basic auth.\n"
            "Manually place bltper_1x1.txt files under data/raw/hmd/<CODE>/ or use --fixture."
        )


if __name__ == "__main__":
    main()
