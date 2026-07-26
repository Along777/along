"""Run Phase 1 end-to-end: templates → downloads → clean → validate."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(mod: str, *extra: str) -> None:
    cmd = [sys.executable, "-m", mod, *extra]
    print("\n==>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    run("src.schema.create_templates")
    run("src.acquisition.download_owid")
    run("src.acquisition.download_hmd_summary")
    run("src.acquisition.download_hld")
    run("src.acquisition.download_clio")
    run("src.acquisition.download_eurostat")
    run("src.acquisition.download_hmd")  # no-op without credentials
    run("src.cleaning.owid_to_long")
    run("src.cleaning.owid_mortality_to_long")
    run("src.cleaning.hmd_summary_to_long")
    run("src.cleaning.hld_to_long")
    run("src.cleaning.clio_to_long")
    run("src.cleaning.eurostat_to_long")
    run("src.cleaning.hmd_life_table", "--fixture")
    run("src.cleaning.merge_fact_tables")
    run("src.cleaning.owid_mortality_to_long")  # re-join IMR after merge
    run("src.cleaning.build_modeling_view")
    run("src.validation.checks")
    print("\nPhase 1 + 1.5 pipeline complete.")


if __name__ == "__main__":
    main()
