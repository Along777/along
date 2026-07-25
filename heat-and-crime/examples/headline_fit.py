"""
Minimal path to the public headline number.

  cd heat-and-crime   # or heat/
  python examples/headline_fit.py

Expect: violent ~ +5.6% per +10F daily max under month x year FE, HAC(9).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import fit_ols, fmt_pct, pct_ci  # noqa: E402
from panel import CTRL_M2, HAC_DEFAULT, build_panel  # noqa: E402


def main() -> None:
    dfa, _ = build_panel(data_dir=ROOT / "data")
    print(f"panel: {len(dfa)} days | mean violent/day={dfa['violent'].mean():.1f}")
    print(f"controls: {CTRL_M2}")
    print(f"HAC maxlags: {HAC_DEFAULT}")
    print()

    for y in ["total", "violent", "property", "battery"]:
        data = dfa[dfa[y] > 0] if y == "battery" else dfa
        res = fit_ols(y, "tmax10", data, hac=HAC_DEFAULT)
        print(f"{y:10s}  {fmt_pct(pct_ci(res, 'tmax10'))}")

    print()
    print("Identification: within month x year; DOW, holiday, rain, snow controlled.")
    print("Effect: 100 * expm1(beta) on log daily counts.")


if __name__ == "__main__":
    main()
