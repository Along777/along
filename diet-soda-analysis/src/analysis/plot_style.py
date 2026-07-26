"""Shared plot style for Myth Lab figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

BEV_ORDER = ["Neither", "SSB-only", "ASB-only", "Both"]
BEV_COLORS = {
    "Neither": "#7f7f7f",
    "SSB-only": "#ff7f0e",
    "ASB-only": "#1f77b4",
    "Both": "#9467bd",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
