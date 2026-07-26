from __future__ import annotations

"""The headline trend numbers, reproduced in ~40 lines.

2020's version of this claim was one straight line on raw annual counts.
This version: negative-binomial IRR per decade with state fixed effects,
restricted to large fires (>= 1,000 acres) where reporting is complete,
cross-checked with a nonparametric Mann-Kendall test -- West vs East vs
national, so the SPLIT verdict on "wildfires are increasing" is visible.

Run (offline, instant):  python examples\\headline_trend.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402
import wildfire as wf  # noqa: E402


def main() -> None:
    ann = wf.load_national()
    # match the lab's CONUS scope: PR (discontinuous reporting) and AK/HI (records
    # present but attribute-less -- found by the Round-2 data audit) are excluded
    ann = ann[~ann["state"].isin(wf.NON_CONUS)]
    large = ann[ann["size_class"].isin(wf.LARGE_CLASSES)]
    total = ann.groupby(["state", "fire_year"], as_index=False)["n"].sum()

    west_large = (large[large["state"].isin(wf.WEST_STATES)]
                  .groupby(["state", "fire_year"], as_index=False)["n"].sum())
    east_large = (large[~large["state"].isin(wf.WEST_STATES)]
                  .groupby(["state", "fire_year"], as_index=False)["n"].sum())

    print("Large fires (>= 1,000 acres), 1992-2020, NB2 GLM with state fixed effects:")
    print(f"  West (11 states):  {models.fmt_irr(models.nb_trend(west_large))}")
    print(f"  East (the rest):   {models.fmt_irr(models.nb_trend(east_large))}")
    print("All reported ignitions, any size (partly a reporting series -- see article):")
    print(f"  National:          {models.fmt_irr(models.nb_trend(total))}")


if __name__ == "__main__":
    main()
