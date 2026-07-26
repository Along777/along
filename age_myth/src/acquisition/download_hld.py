"""Download Human Life-Table Database bulk data (no registration)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.paths import RAW, ensure_dirs

UA = "age-myth-historical-le-db/0.1 (research; HLD acquisition)"

# Candidate bulk URLs (HLD site structure can shift)
CANDIDATE_URLS = [
    "https://www.lifetable.de/File/GetDocument/data/hld.zip",
    "https://lifetable.de/File/GetDocument/data/hld.zip",
    "http://www.lifetable.de/File/GetDocument/data/hld.zip",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_download(url: str, dest: Path) -> dict:
    headers = {"User-Agent": UA}
    with requests.get(url, headers=headers, timeout=300, stream=True) as r:
        r.raise_for_status()
        total = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    return {
        "url": url,
        "path": str(dest),
        "bytes": total,
        "sha256": _sha256_file(dest),
        "status_code": 200,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_zip(zip_path: Path, out_dir: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
        names = zf.namelist()
    # HLD bulk zip often contains a single member named "res" (CSV without extension)
    res = out_dir / "res"
    if res.exists() and res.is_file():
        # leave as `res`; also write a small header peek for humans
        peek = out_dir / "res_HEADER.txt"
        try:
            with res.open("r", encoding="utf-8", errors="replace") as f:
                peek.write_text("".join(f.readline() for _ in range(3)), encoding="utf-8")
        except OSError:
            pass
    return names


def inventory_extracted(root: Path) -> dict:
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.suffix.lower() in {".csv", ".txt", ".dat", ".xlsx", ".xls"}
            or p.name.lower() in {"res", "res.csv"}
        )
    ]
    by_ext: dict[str, int] = {}
    for p in files:
        key = p.suffix.lower() if p.suffix else f"(noext:{p.name})"
        by_ext[key] = by_ext.get(key, 0) + 1
    sample = [str(p.relative_to(root)) for p in files[:30]]
    res = root / "res"
    res_note = None
    if res.exists():
        res_note = {
            "path": str(res),
            "bytes": res.stat().st_size,
            "format_hint": "HLD bulk CSV (Country, Year1, Year2, Sex, Age, e(x), ...)",
        }
    return {
        "n_data_files": len(files),
        "by_extension": by_ext,
        "sample_paths": sample,
        "res_table": res_note,
    }


def main() -> None:
    ensure_dirs()
    out_dir = RAW / "hld"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "hld.zip"
    log: dict = {"attempts": [], "success": None, "extract": None, "inventory": None, "errors": []}

    if zip_path.exists() and zip_path.stat().st_size > 1000:
        print(f"Using existing {zip_path} ({zip_path.stat().st_size:,} bytes)")
        log["success"] = {
            "url": "(cached local file)",
            "path": str(zip_path),
            "bytes": zip_path.stat().st_size,
            "sha256": _sha256_file(zip_path),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        for url in CANDIDATE_URLS:
            print(f"Trying {url} ...")
            try:
                entry = try_download(url, zip_path)
                log["attempts"].append({"url": url, "ok": True, "bytes": entry["bytes"]})
                log["success"] = entry
                print(f"Downloaded {entry['bytes']:,} bytes")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  failed: {e}")
                log["attempts"].append({"url": url, "ok": False, "error": str(e)})
                log["errors"].append(str(e))

    if not log["success"]:
        readme = out_dir / "MANUAL_DOWNLOAD.md"
        readme.write_text(
            "# HLD manual download\n\n"
            "Automatic download failed. Please download the full HLD data zip from "
            "https://www.lifetable.de/ and place it at:\n\n"
            f"`{zip_path}`\n\n"
            "Then re-run: `python -m src.acquisition.download_hld`\n",
            encoding="utf-8",
        )
        print(f"HLD bulk download failed. See {readme}")
        (out_dir / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        raise SystemExit(1)

    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    try:
        names = extract_zip(zip_path, extract_dir)
        log["extract"] = {"n_members": len(names), "sample": names[:40]}
        log["inventory"] = inventory_extracted(extract_dir)
        print(f"Extracted {len(names)} members; data files: {log['inventory']['n_data_files']}")
    except zipfile.BadZipFile as e:
        log["errors"].append(f"Bad zip: {e}")
        print(f"Zip extract failed: {e}")

    (out_dir / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'download_log.json'}")


if __name__ == "__main__":
    main()
