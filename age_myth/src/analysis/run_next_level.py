"""One command: bulletproof myth-bust suite end-to-end."""
from __future__ import annotations


def main() -> None:
    print(
        "DEPRECATED: prefer `python -m src.analysis.run_final_agrade` for ship demos.\n"
    )
    print("=== 1/5 Ladders + claim gates ===")
    from src.analysis.bulletproof_suite import run as suite

    suite()

    print("=== 2/5 Definitive figures D1-D12 ===")
    from src.analysis.figures_definitive import run_all as figs

    figs()

    print("=== 3/5 Peer pack (bonus charts) ===")
    try:
        from src.analysis.figures_peer_pack import run_all as peer_figs

        peer_figs()
    except Exception as e:  # noqa: BLE001
        print(f"Peer pack skipped: {e}")

    print("=== 4/5 Reports ===")
    from src.analysis.write_bulletproof_report import main as report
    from src.analysis.build_html_board import main as board

    report()
    board()

    print("=== 5/5 Done ===")
    print("Read: outputs/reports/MYTH_BUST_BULLETPROOF.md")
    print("Open: outputs/reports/myth_bust_board.html")
    print("Gates: outputs/bulletproof/claim_gate_results.json")


if __name__ == "__main__":
    main()
