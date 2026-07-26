"""One-shot A-grade finish — strict by default for ship demos."""
from __future__ import annotations

import argparse
import json
import sys

from src.paths import OUTPUTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ship pipeline: claims → figures → scorecard → report")
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Continue on non-critical figure/report errors (dev only)",
    )
    args = parser.parse_args(argv)
    soft = args.soft

    def step(title: str, fn, critical: bool = True) -> None:
        print(f"=== {title} ===")
        try:
            fn()
        except Exception as e:
            print(f"ERROR in {title}: {e}")
            if critical and not soft:
                raise
            if not soft:
                raise
            print(f"(soft mode) continuing after: {e}")

    step("1/5 Bulletproof suite (claims + hard gates)", _run_suite, critical=True)
    step("2/5 Final A-grade metrics (pre-figure)", _run_final_metrics, critical=True)
    step("3/5 Figures (definitive + final)", _run_figures, critical=not soft)
    step("4/5 Recompute scorecard AFTER figures", _rescore, critical=True)
    step("5/5 Reports + HTML board", _run_reports, critical=not soft)

    sc_path = OUTPUTS / "bulletproof" / "agrade_scorecard.json"
    sc = json.loads(sc_path.read_text(encoding="utf-8"))
    print(f"GRADE: {sc['letter']} ({sc['total']}/100)")
    print(f"Figure counts: {sc.get('figure_counts')}")
    print("Read: outputs/reports/FINAL_A_GRADE_REPORT.md")
    print("Gates: outputs/bulletproof/claim_gate_results.json")
    # Stakeholder caution: don't treat letter as external validation
    print("NOTE: Scorecard is an internal engineering checklist, not peer review.")


def _run_suite() -> None:
    from src.analysis.bulletproof_suite import run as suite

    suite()


def _run_final_metrics() -> None:
    from src.analysis.final_agrade import run as final_run

    final_run()


def _run_figures() -> None:
    from src.analysis.figures_definitive import run_all as dfigs
    from src.analysis.figures_final import run_all as ffigs

    dfigs()
    ffigs()


def _rescore() -> None:
    """Re-run final_agrade after figures so chart_pack counts are honest."""
    from src.analysis.final_agrade import run as final_run

    final_run()


def _run_reports() -> None:
    from src.analysis.build_html_board import main as board
    from src.analysis.write_final_report import main as report

    report()
    board()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
