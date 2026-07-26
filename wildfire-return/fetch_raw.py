from __future__ import annotations

"""One-time download of the raw inputs for Return to Fire.

Pulls the FPA FOD-Attributes annual CSVs (Zenodo record 8381129, CC-BY 4.0;
~5.1 GB for all 29 years) into data/raw/, plus the US Census state cartographic
boundary shapefile into data/raw/geo/. Every other script in this project runs
offline against the reduced caches in data/ -- this is the only script that
needs the network (besides fetch_recent.py).

Usage:
    python fetch_raw.py                      # print the download plan and exit
    python fetch_raw.py --yes                # download all years (~5.1 GB)
    python fetch_raw.py --yes --years 2015-2020
    python fetch_raw.py --yes --years 2017

Files already on disk with a matching size and md5 are skipped, so re-running
after an interrupted session only fetches what is missing.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
GEO_RAW = RAW / "geo"

ZENODO_RECORD = "8381129"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ANNUAL_RE = re.compile(r"^(\d{4})_FPA_FOD_cons\.csv$")

CENSUS_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"

CHUNK = 1 << 20  # 1 MiB
PROGRESS_EVERY = 25 * (1 << 20)  # progress line every 25 MB
ATTEMPTS = 3


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_years(spec: str | None) -> set[int] | None:
    """'2015-2020' | '2017' | '1992,1995,2020' -> set of years; None -> all."""
    if not spec:
        return None
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        else:
            years.add(int(part))
    return years


def list_annual_files(session: requests.Session) -> list[dict]:
    """Enumerate the per-year CSVs on the Zenodo record (skips the 5 GB blob)."""
    r = session.get(ZENODO_API, timeout=(10, 60))
    r.raise_for_status()
    record = r.json()
    files = []
    for entry in record.get("files", []):
        m = ANNUAL_RE.match(entry["key"])
        if not m:
            continue  # FPA_FOD_Plus.csv (consolidated) and anything else
        checksum = entry.get("checksum", "")
        files.append(
            {
                "key": entry["key"],
                "year": int(m.group(1)),
                "size": int(entry["size"]),
                "md5": checksum.split(":", 1)[1] if checksum.startswith("md5:") else None,
                "url": f"{ZENODO_API}/files/{quote(entry['key'])}/content",
                "fallback_url": f"https://zenodo.org/records/{ZENODO_RECORD}/files/{quote(entry['key'])}?download=1",
            }
        )
    files.sort(key=lambda f: f["year"])
    if not files:
        raise SystemExit("Zenodo record listed no annual CSVs -- API shape changed? Inspect: " + ZENODO_API)
    return files


def stream_download(session: requests.Session, url: str, dest: Path, expected_size: int | None) -> str:
    """Stream url -> dest, hashing md5 while writing. Returns the md5 hex.

    If a .part file exists from an interrupted run, resume it with an HTTP
    Range request: the existing prefix is re-hashed from disk (reads only the
    bytes already present -- zero cost in the uninterrupted case), then the
    stream appends. Servers that ignore Range (HTTP 200) trigger a clean
    restart. The final digest is always over the complete file either way.
    """
    h = hashlib.md5()
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = 0
    headers: dict[str, str] = {}
    if tmp.exists():
        sz = tmp.stat().st_size
        if expected_size and 0 < sz < expected_size:
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    h.update(chunk)
            start = sz
            headers["Range"] = f"bytes={sz}-"
        else:
            tmp.unlink()  # zero-byte or oversized partial: restart cleanly
    with session.get(url, stream=True, timeout=(10, 120), headers=headers) as r:
        r.raise_for_status()
        if start and r.status_code == 206:
            mode = "ab"
            print(f"    [resume] {dest.name} continuing at {start // (1 << 20)} MB (HTTP 206)", flush=True)
        else:
            if start:
                print(f"    [resume] server ignored Range for {dest.name}; restarting", flush=True)
            h = hashlib.md5()
            start = 0
            mode = "wb"
        got = start
        with open(tmp, mode) as f:
            next_mark = (got // PROGRESS_EVERY + 1) * PROGRESS_EVERY
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if got >= next_mark:
                    total = f"/{expected_size // (1 << 20)}" if expected_size else ""
                    print(f"    {dest.name}  {got // (1 << 20)}{total} MB", flush=True)
                    next_mark += PROGRESS_EVERY
    tmp.replace(dest)
    return h.hexdigest()


def fetch_file(session: requests.Session, spec: dict, dest: Path) -> dict:
    """Download one annual CSV with skip/resume + verification. Returns a log row."""
    started = time.time()
    if dest.exists():
        if spec["md5"] and dest.stat().st_size == spec["size"] and md5_of(dest) == spec["md5"]:
            print(f"  [skip] {dest.name} already complete ({spec['size'] / (1 << 20):.0f} MB)")
            return {"file": dest.name, "status": "already_complete", "bytes": spec["size"], "md5": spec["md5"]}
        print(f"  [redo] {dest.name} exists but is partial or mismatched -- re-downloading")
        dest.unlink()

    last_err: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        url = spec["url"] if attempt < ATTEMPTS else spec["fallback_url"]
        try:
            print(f"  [get ] {dest.name} ({spec['size'] / (1 << 20):.0f} MB) attempt {attempt}", flush=True)
            digest = stream_download(session, url, dest, spec["size"])
            if spec["md5"] and digest != spec["md5"]:
                raise IOError(f"md5 mismatch: got {digest}, want {spec['md5']}")
            elapsed = time.time() - started
            print(f"  [ ok ] {dest.name} verified in {elapsed:.0f}s", flush=True)
            return {"file": dest.name, "status": "downloaded", "bytes": spec["size"], "md5": digest, "seconds": round(elapsed)}
        except Exception as e:  # noqa: BLE001 -- log and retry, fail loudly at the end
            last_err = e
            print(f"  [err ] {dest.name}: {e}", flush=True)
            if dest.exists():
                dest.unlink()
            # a stream error leaves the .part as the resume asset for the next
            # attempt; an md5 mismatch means the bytes are bad -- start clean
            tmp = dest.with_suffix(dest.suffix + ".part")
            if "md5 mismatch" in str(e) and tmp.exists():
                tmp.unlink()
            time.sleep(5 * attempt)
    return {"file": dest.name, "status": "FAILED", "error": str(last_err)}


def fetch_census_boundaries(session: requests.Session) -> dict:
    dest = GEO_RAW / "cb_2023_us_state_20m.zip"
    if dest.exists() and dest.stat().st_size > 100_000 and dest.read_bytes()[:2] == b"PK":
        print(f"  [skip] {dest.name} already present")
        return {"file": dest.name, "status": "already_complete", "bytes": dest.stat().st_size}
    print(f"  [get ] {dest.name} (US Census cartographic boundaries, ~1 MB)")
    digest = stream_download(session, CENSUS_URL, dest, None)
    if dest.read_bytes()[:2] != b"PK":
        raise SystemExit(f"{dest} is not a zip -- Census URL changed? {CENSUS_URL}")
    return {"file": dest.name, "status": "downloaded", "bytes": dest.stat().st_size, "md5": digest}


def verify_only(session: requests.Session) -> int:
    """Re-hash everything on disk against the Zenodo checksums. No downloads."""
    files = list_annual_files(session)
    failures = 0
    print(f"{'file':28s} {'bytes':>6s} {'md5':>5s}  status")
    for spec in files:
        dest = RAW / spec["key"]
        if not dest.exists():
            print(f"{spec['key']:28s} {'-':>6s} {'-':>5s}  MISSING")
            failures += 1
            continue
        size_ok = dest.stat().st_size == spec["size"]
        md5_ok = size_ok and (spec["md5"] is None or md5_of(dest) == spec["md5"])
        status = "OK" if (size_ok and md5_ok) else ("SIZE MISMATCH" if not size_ok else "MD5 MISMATCH")
        print(f"{spec['key']:28s} {'ok' if size_ok else 'BAD':>6s} "
              f"{'ok' if md5_ok else 'BAD':>5s}  {status}")
        failures += 0 if status == "OK" else 1
    census = GEO_RAW / "cb_2023_us_state_20m.zip"
    census_ok = census.exists() and census.stat().st_size > 100_000 and census.read_bytes()[:2] == b"PK"
    print(f"{census.name:28s} {'ok' if census_ok else 'BAD':>6s} {'-':>5s}  "
          f"{'OK' if census_ok else 'MISSING/CORRUPT'}")
    failures += 0 if census_ok else 1
    print(f"\n{len(files) + 1 - failures}/{len(files) + 1} files verified ok.")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Download FPA FOD-Attributes raw CSVs + Census boundaries")
    ap.add_argument("--yes", action="store_true", help="actually download (required; the full pull is ~5.1 GB)")
    ap.add_argument("--years", help="subset like 2015-2020 or 2017 or 1992,1995 (default: all 1992-2020)")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-hash existing files against Zenodo checksums; download nothing")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    GEO_RAW.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "wildfire-return/1.0 (portfolio project; one-time bulk fetch)"

    if args.verify_only:
        sys.exit(1 if verify_only(session) else 0)

    print(f"Enumerating Zenodo record {ZENODO_RECORD} ...")
    files = list_annual_files(session)
    wanted_years = parse_years(args.years)
    if wanted_years is not None:
        missing = wanted_years - {f["year"] for f in files}
        if missing:
            raise SystemExit(f"Years not on the record: {sorted(missing)} (record covers "
                             f"{files[0]['year']}-{files[-1]['year']})")
        files = [f for f in files if f["year"] in wanted_years]

    total = sum(f["size"] for f in files)
    print(f"\nPlan: {len(files)} annual CSVs, {total / (1 << 30):.2f} GB total -> {RAW}")
    print(f"      years {files[0]['year']}-{files[-1]['year']}  +  Census state boundaries -> {GEO_RAW}")
    if not args.yes:
        print("\nDry run. Re-run with --yes to download.")
        return

    log: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"zenodo_record": ZENODO_RECORD, "api": ZENODO_API, "license": "CC-BY-4.0"},
        "files": [],
    }
    log_path = RAW / "download_log.json"

    for spec in files:
        row = fetch_file(session, spec, RAW / spec["key"])
        log["files"].append(row)
        log_path.write_text(json.dumps(log, indent=2))  # incremental, survives interrupts

    log["files"].append(fetch_census_boundaries(session))
    log["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_path.write_text(json.dumps(log, indent=2))

    failed = [r["file"] for r in log["files"] if r.get("status") == "FAILED"]
    done = [r for r in log["files"] if r.get("status") in ("downloaded", "already_complete")]
    print(f"\nDone: {len(done)} files ok, {len(failed)} failed.")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
