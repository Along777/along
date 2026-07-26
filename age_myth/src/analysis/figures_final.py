"""Final A-grade figures (lean set)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.ladder import BULLET, load_hmd_wide
from src.analysis.populations import label_region
from src.paths import OUTPUTS, ensure_dirs

FIGS = OUTPUTS / "figures" / "final"
sns.set_theme(style="whitegrid", context="notebook")


def fa1_year_vs_equal_country() -> Path:
    data = json.loads((BULLET / "final_claims.json").read_text(encoding="utf-8"))
    yw = data["claim_A_year_weighted"]["year_weighted"]["median_exp65"]
    ec = data["claim_A_year_weighted"]["equal_country"]
    countries = pd.DataFrame(ec["country_table"]).sort_values("median_exp65")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        ["Year-weighted\ncountry-years", "Equal-country\nmedian of medians"],
        [yw, ec["median_of_country_medians_exp65"]],
        color=["#1f77b4", "#2ca02c"],
    )
    axes[0].axhline(35, color="salmon", ls="--", label="Myth 35")
    axes[0].axhline(70, color="black", ls=":", label="70")
    axes[0].set_ylabel("Median expected age if alive at 65")
    axes[0].set_title("Weighting does not create the result")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 90)
    y = np.arange(len(countries))
    axes[1].barh(y, countries["median_exp65"], color="steelblue")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(countries["region_id"].map(label_region), fontsize=8)
    axes[1].axvline(35, color="salmon", ls="--")
    axes[1].axvline(70, color="black", ls=":")
    axes[1].set_xlabel("Country median age|65 when e0<40")
    axes[1].set_title("Every country median ~75")
    fig.suptitle("F-A1 Year-weighted vs equal-country (Claim A robustness)", y=1.02)
    fig.tight_layout()
    path = FIGS / "FA1_year_vs_equal_country.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fa2_strict_band(panel: pd.DataFrame) -> Path:
    band = panel[(panel.e0 >= 30) & (panel.e0 <= 35)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].hist(band.e0, bins=12, color="#1f77b4", edgecolor="white")
    axes[0].set_title("e0 in [30, 35]")
    axes[1].hist(band.exp_death_65, bins=12, color="#2ca02c", edgecolor="white")
    axes[1].axvline(35, color="salmon", ls="--")
    axes[1].axvline(70, color="black", ls="--")
    axes[1].set_title("Expected age if alive at 65")
    axes[2].hist(band.s_to_65 * 100, bins=12, color="#9467bd", edgecolor="white")
    axes[2].set_title("S(0→65) %")
    fig.suptitle(
        f"F-A2 Strict myth band (n={len(band)} country-years, {band.region_id.nunique()} countries)\n"
        "Even when e0 is 30–35, age|65 stays ~75; survival to 65 is low",
        y=1.05,
    )
    fig.tight_layout()
    path = FIGS / "FA2_strict_band_30_35.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fa3_dual_ladder() -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False
    for mode, color, offset in [("gold", "#1f77b4", -0.15), ("median", "#2ca02c", 0.15)]:
        p = BULLET / f"final_ladder_hld_{mode}_e0lt40.csv"
        if not p.exists():
            p = BULLET / f"ladder_hld_{mode}_agg_e0lt40.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        x = d["age_x"].to_numpy() + offset
        ax.bar(x, d["median_expected_age"], width=0.3, label=f"HLD {mode}", color=color, alpha=0.85)
        plotted = True
    ax.axhline(35, color="salmon", ls="--", lw=1.5, label="Myth: 35")
    ax.set_xlabel("Age x (alive at x)")
    ax.set_ylabel("Median expected age at death (x + e(x))")
    ax.set_title("F-A3 Dual age ladder when e0<40 — gold (n_tables=1) vs median")
    ax.legend()
    if not plotted:
        ax.text(0.5, 0.5, "ladder files missing", transform=ax.transAxes, ha="center")
    path = FIGS / "FA3_dual_hld_ladder.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fa4_sex(panel_f=None) -> Path:
    from src.analysis.panels import ANALYSIS_DIR
    from src.analysis.populations import dedupe_hmd

    f = dedupe_hmd(pd.read_parquet(ANALYSIS_DIR / "hmd_summary_wide_female.parquet"))
    m = dedupe_hmd(pd.read_parquet(ANALYSIS_DIR / "hmd_summary_wide_male.parquet"))
    # Sweden lines + low-e0 strip
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for sex, df, color in [("female", f, "#e377c2"), ("male", m, "#1f77b4")]:
        s = df[df.region_id == "SWE"].sort_values("year")
        axes[0].plot(s.year, s.e0, label=f"{sex} e0", color=color)
        axes[1].plot(s.year, s.exp_death_65, label=f"{sex} age|65", color=color)
    for ax in axes:
        ax.axhline(35, color="salmon", ls="--", lw=1)
    axes[0].set_title("Sweden e0 by sex")
    axes[1].set_title("Sweden expected age|65 by sex")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    # annotate Iceland female floor
    data = json.loads((BULLET / "final_claims.json").read_text(encoding="utf-8"))
    mc = data["sex_specific_claim_A"]["female"]["min_case"]
    axes[1].annotate(
        f"Female min: {mc['region_id']} {mc['year']}\n"
        f"age|65={mc['exp_death_65']:.1f}, S65={mc['s_to_65']:.2f}, IMR={mc['imr']:.0f}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.9),
    )
    fig.suptitle("F-A4 Sex honesty — myth fails for both sexes (crisis years still ~70, not 30)")
    fig.tight_layout()
    path = FIGS / "FA4_sex_honesty.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fa5_collage() -> Path:
    """2x2 collage of key definitive figures via matplotlib imshow."""
    base = OUTPUTS / "figures" / "definitive"
    names = [
        "D1_sweden_storyboard.png",
        "D2_age_ladder_when_e0_under_40.png",
        "D6_survival_composition.png",
        "D10_mythB_dumbbell.png",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        p = base / name
        ax.axis("off")
        if p.exists():
            ax.imshow(plt.imread(p))
            ax.set_title(name.replace(".png", ""), fontsize=9)
        else:
            ax.text(0.5, 0.5, f"missing\n{name}", ha="center", va="center")
    fig.suptitle("F-A5 Myth-bust collage — core evidence", y=0.98)
    fig.tight_layout()
    path = FIGS / "FA5_collage.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run_all() -> list[Path]:
    ensure_dirs()
    FIGS.mkdir(parents=True, exist_ok=True)
    if not (BULLET / "final_claims.json").exists():
        from src.analysis.final_agrade import run

        run()
    panel = load_hmd_wide("both")
    paths = [
        fa1_year_vs_equal_country(),
        fa2_strict_band(panel),
        fa3_dual_ladder(),
        fa4_sex(),
    ]
    try:
        paths.append(fa5_collage())
    except Exception as e:  # noqa: BLE001
        print(f"collage skip: {e}")
    for p in paths:
        print(f"Wrote {p}")
    return paths


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
