"""
F1-style leads check: future heat should not carry the same-day effect.

  python examples/leads_placebo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import fit_ols, fmt_pct, pct_ci  # noqa: E402
from panel import HAC_DEFAULT, build_panel  # noqa: E402


def main() -> None:
    dfa, _ = build_panel(data_dir=ROOT / "data")
    use = dfa.dropna(subset=["tmax10_F1", "tmax10_F2", "tmax10_F3"])
    res = fit_ols(
        "violent",
        "tmax10 + tmax10_F1 + tmax10_F2 + tmax10_F3",
        use,
        hac=HAC_DEFAULT,
    )
    for term, label in [
        ("tmax10", "same-day"),
        ("tmax10_F1", "lead +1 day"),
        ("tmax10_F2", "lead +2 day"),
        ("tmax10_F3", "lead +3 day"),
    ]:
        print(f"{label:14s}  {fmt_pct(pct_ci(res, term))}")

    same = pct_ci(res, "tmax10")["pct"]
    lead1 = abs(pct_ci(res, "tmax10_F1")["pct"])
    ok = lead1 < 0.5 * abs(same)
    print()
    print(f"pass ( |lead1| < 0.5 * |same| ): {ok}")


if __name__ == "__main__":
    main()
