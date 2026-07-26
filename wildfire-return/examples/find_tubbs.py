from __future__ import annotations

"""The row that was past the edge of the data in 2020.

The original wildfire trilogy ran on a dataset that ended in 2015. The fire
that shaped this project's author -- the 2017 Tubbs Fire -- was 22 months past
the last record. The current dataset reaches 2020. This script prints the row.

Run (offline, instant):  python examples\\find_tubbs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wildfire as wf  # noqa: E402


def k_to_f(kelvin: float) -> float:
    return (kelvin - 273.15) * 9 / 5 + 32


def main() -> None:
    t = wf.load_tubbs()
    c = t["curated"]
    print("=" * 62)
    print(f"  {c['fire_name']} FIRE  --  FOD_ID {c['fod_id']}  (FPA-FOD v6 + attributes)")
    print("=" * 62)
    print(f"  discovered      {c['discovery_date'][:10]}  (day {c['doy_std']} of the year)")
    print(f"  final size      {c['fire_size']:,.0f} acres  (class {c['size_class']})")
    print(f"  burn duration   {c['burn_days']:.0f} days to containment")
    print(f"  ignition point  {c['lat']:.5f}, {c['lon']:.5f}  ({c['county']} County, {c['state']})")
    print(f"  cause on file   {c['cause_class']} / {c['cause_general']}")
    print("  -- at-ignition conditions written into the row --")
    print(f"  max temp        {k_to_f(c['tmmx']):.0f} F")
    print(f"  min humidity    {c['rmin']:.1f} %")
    print(f"  wind (daily)    {c['wind']:.1f} m/s  (gridMET daily mean; the night's gusts ran far higher)")
    print(f"  vapor deficit   {c['vpd']:.2f} kPa")
    print(f"  ERC             {c['erc']:.0f}  (local climatological percentile: {c['erc_pctl']})")
    print("-" * 62)
    print("  In 2020 this row did not exist to be found: the data ended in")
    print("  2015. The data caught up to the story. That is the whole project.")


if __name__ == "__main__":
    main()
