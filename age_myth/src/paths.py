"""Project path helpers. Resolve relative to age_myth root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
FIXTURES = RAW / "fixtures"


def ensure_dirs() -> None:
    for p in (
        RAW / "owid",
        RAW / "hmd",
        RAW / "hmd_summary",
        RAW / "hld",
        RAW / "eurostat",
        RAW / "clio_infra",
        RAW / "cambridge_group",
        RAW / "gurven_kaplan",
        RAW / "elite_series",
        FIXTURES,
        INTERIM,
        PROCESSED,
        OUTPUTS,
    ):
        p.mkdir(parents=True, exist_ok=True)
