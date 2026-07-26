from __future__ import annotations

"""The full look at the data before any model exists (Part I's role, 2026-grade).

Produces the EDA figures + eda_results.json that the article's #eda section and
the labs' design choices are built on:

    eda_missingness.png          which attributes can be trusted, and where
    eda_cause_taxonomy.png       the 13 NWCG general causes, colored by class
    eda_size_distribution.png    log-log survival curves: the heavy tail
    eda_seasonality.png          CA vs FL fire calendars (the two-regime story)
    eda_reporting_coverage.png   state x year count heatmap: reporting is visible
    eda_weather_by_size.png      at-ignition fire weather, large vs small (CA)

Run (offline, after reduce_raw.py):  python run_eda.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wildfire as wf

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}


def style_ax(ax, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COL["axis"])
    ax.tick_params(colors=COL["ink2"], labelsize=9)
    ax.xaxis.label.set_color(COL["ink2"])
    ax.yaxis.label.set_color(COL["ink2"])
    ax.title.set_color(COL["ink"])


def savefig(fig, name: str) -> None:
    wf.FIG.mkdir(exist_ok=True)
    fig.savefig(wf.FIG / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig ] {name}")


def main() -> None:
    for f in ("ca_fires.parquet", "fl_fires.parquet", "national_annual.csv",
              "national_monthly.csv", "reduce_report.json"):
        wf.verify_manifest(f, strict=True)
    report = json.loads((wf.DATA / "reduce_report.json").read_text())
    ca = wf.load_state_fires("CA")
    fl = wf.load_state_fires("FL")
    ann = wf.load_national()
    results: dict = {"corpus_rows": report["rows_total"]}

    # ---------------------------------------------------- 1. missingness
    both = pd.concat([ca.assign(st="CA"), fl.assign(st="FL")], ignore_index=True)
    miss = both.drop(columns=["st"]).isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0.0005]
    fig, ax = plt.subplots(figsize=(8.6, 0.28 * len(miss) + 1.2))
    colors = [COL["red"] if v > 0.30 else COL["gold"] if v > 0.05 else COL["blue"] for v in miss.values]
    ax.barh(miss.index[::-1], miss.values[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("missing (%) -- CA + FL fire-level rows, 1992-2020")
    ax.set_title("What the attribute table actually delivers", loc="left", fontsize=11)
    style_ax(ax, grid_axis="x")
    savefig(fig, "eda_missingness.png")
    results["missing_shares_ca_fl"] = {k: round(float(v), 4) for k, v in miss.items()}
    results["missing_shares_corpus_keycols"] = report["corpus"]["missing_shares"]

    # ---------------------------------------------------- 2. cause taxonomy
    causes = pd.Series(report["corpus"]["cause_general_counts"]).drop("nan", errors="ignore")
    causes = causes.sort_values()
    total = causes.sum()
    cls_color = {}
    for cause in causes.index:
        if cause == "Natural":
            cls_color[cause] = COL["green"]
        elif cause == wf.MISSING_CAUSE:
            cls_color[cause] = COL["muted"]
        else:
            cls_color[cause] = COL["fire"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh(causes.index, causes.values / 1000, color=[cls_color[c] for c in causes.index])
    for i, (name, v) in enumerate(causes.items()):
        ax.text(v / 1000 + 4, i, f"{v / total:.1%}", va="center", fontsize=8, color=COL["ink2"])
    ax.set_xlabel("fires, 1992-2020 (thousands)")
    ax.set_title("13 NWCG general causes -- human (orange), natural (green), unknown (gray)",
                 loc="left", fontsize=11)
    style_ax(ax, grid_axis="x")
    savefig(fig, "eda_cause_taxonomy.png")
    cc = report["corpus"]["cause_class_counts"]
    known = sum(v for k, v in cc.items() if k in ("Human", "Natural"))
    results["cause_class_counts"] = cc
    results["human_share_of_known"] = round(cc.get("Human", 0) / known, 4)
    results["missing_cause_share"] = round(
        sum(v for k, v in cc.items() if k not in ("Human", "Natural")) / report["rows_total"], 4)

    # ---------------------------------------------------- 3. size distribution (CCDF)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for df, name, color in ((ca, "California", COL["fire"]), (fl, "Florida", COL["blue"])):
        sizes = np.sort(df["fire_size"].dropna().values)
        ccdf = 1 - np.arange(1, len(sizes) + 1) / len(sizes)
        keep = sizes > 0
        ax.loglog(sizes[keep], np.maximum(ccdf[keep], 1e-7), color=color, lw=1.6, label=name)
    for acres, label in ((1000, "class F"), (5000, "class G")):
        ax.axvline(acres, color=COL["axis"], lw=0.8, ls=":")
        ax.text(acres * 1.15, 0.4, label, fontsize=8, color=COL["muted"], rotation=90, va="top")
    ax.set_xlabel("fire size (acres, log)")
    ax.set_ylabel("P(size > x)  (log)")
    ax.set_title("The heavy tail: most fires are tiny, the acres live in the giants",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=COL["ink2"])
    style_ax(ax, grid_axis="")
    savefig(fig, "eda_size_distribution.png")
    for df, key in ((ca, "ca"), (fl, "fl")):
        big = df[df["size_class"].isin(wf.LARGE_CLASSES)]
        results[f"{key}_fires_total"] = int(len(df))
        results[f"{key}_large_fires"] = int(len(big))
        results[f"{key}_large_share_of_acres"] = round(
            float(big["fire_size"].sum() / df["fire_size"].sum()), 4)

    # ---------------------------------------------------- 4. seasonality
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bins = np.arange(1, 367, 7)
    for df, name, color in ((ca, "California", COL["fire"]), (fl, "Florida", COL["blue"])):
        h, edges = np.histogram(df["doy_std"], bins=bins, density=True)
        ax.plot(edges[:-1] + 3.5, h * 100, color=color, lw=1.7, label=f"{name} -- all fires")
        big = df[df["size_class"].isin(wf.LARGE_CLASSES)]
        h2, _ = np.histogram(big["doy_std"], bins=bins, density=True)
        ax.plot(edges[:-1] + 3.5, h2 * 100, color=color, lw=1.2, ls="--",
                label=f"{name} -- large (F/G)")
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("share of fires per week (%)")
    ax.set_title("Two fire regimes: Florida burns in spring, California in summer-fall",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"], ncols=2)
    style_ax(ax)
    savefig(fig, "eda_seasonality.png")
    results["peak_month"] = {
        "ca_all": int(ca["month"].mode().iloc[0]), "fl_all": int(fl["month"].mode().iloc[0]),
        "ca_large": int(ca.loc[ca["size_class"].isin(wf.LARGE_CLASSES), "month"].mode().iloc[0]),
        "fl_large": int(fl.loc[fl["size_class"].isin(wf.LARGE_CLASSES), "month"].mode().iloc[0]),
    }
    oct_share = float((ca.loc[ca["size_class"] == "G", "month"].isin([9, 10, 11])).mean())
    results["ca_classG_sep_nov_share"] = round(oct_share, 4)

    # ---------------------------------------------------- 5. reporting coverage
    counts = ann.groupby(["state", "fire_year"])["n"].sum().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    logc = np.log10(counts + 1)
    disc = (np.abs(np.log(counts.where(counts > 0) + 1).diff(axis=1))).max(axis=1)
    top_disc = disc.sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(9.2, 8.8))
    im = ax.imshow(logc.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(logc.index)))
    ax.set_yticklabels(logc.index, fontsize=6.2)
    years = list(logc.columns)
    ax.set_xticks(range(0, len(years), 4))
    ax.set_xticklabels(years[::4], fontsize=8)
    for st in top_disc.index:
        i = list(logc.index).index(st)
        ax.text(len(years) - 0.3, i, "<- step change", fontsize=6.5, color=COL["red"], va="center")
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.09)
    cbar.set_label("log10(reported fires + 1)", fontsize=8, color=COL["ink2"])
    cbar.ax.tick_params(labelsize=7, colors=COL["ink2"])
    ax.set_title("Reported fires by state and year: the raw count series partly measures paperwork",
                 loc="left", fontsize=11)
    style_ax(ax, grid_axis="")
    savefig(fig, "eda_reporting_coverage.png")
    results["reporting_step_change_states"] = {k: round(float(v), 3) for k, v in top_disc.items()}

    # ---------------------------------------------------- 6. fire weather by outcome size (CA)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    panels = [("erc", "ERC (energy release component)"), ("vpd", "VPD (kPa)"),
              ("rmin", "min relative humidity (%)"), ("wind", "wind speed (m/s, daily mean)")]
    big_mask = ca["size_class"].isin(wf.LARGE_CLASSES)
    for ax, (col, label) in zip(axes.ravel(), panels):
        small_v = ca.loc[~big_mask, col].dropna()
        big_v = ca.loc[big_mask, col].dropna()
        lo, hi = np.nanpercentile(pd.concat([small_v, big_v]), [0.5, 99.5])
        grid = np.linspace(lo, hi, 40)
        for vals, name, color in ((small_v, "< 1,000 ac", COL["muted"]),
                                  (big_v, ">= 1,000 ac", COL["fire"])):
            h, edges = np.histogram(vals, bins=grid, density=True)
            ax.plot(edges[:-1], h, color=color, lw=1.5, label=name)
        ax.set_xlabel(label, fontsize=8.5)
        ax.set_yticks([])
        style_ax(ax, grid_axis="")
        results[f"ca_{col}_median_small"] = round(float(small_v.median()), 2)
        results[f"ca_{col}_median_large"] = round(float(big_v.median()), 2)
    axes[0, 0].legend(frameon=False, fontsize=8.5, labelcolor=COL["ink2"])
    fig.suptitle("California ignitions: the fires that got big started on different days",
                 x=0.02, ha="left", fontsize=11, color=COL["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "eda_weather_by_size.png")

    # top absolute correlations among numeric features (for the JSON, not a figure)
    num = ca[wf.FLOAT32_COLS].corr(numeric_only=True)
    pairs = (num.where(np.triu(np.ones_like(num, dtype=bool), 1))
             .stack().sort_values(key=np.abs, ascending=False).head(12))
    results["top_feature_correlations_ca"] = {f"{a} x {b}": round(float(v), 3)
                                              for (a, b), v in pairs.items()}

    # ------------------------------------------------ modeling handoff (stage review)
    # The label-selection facts the modeling stage must inherit: cause-label
    # availability is not constant across the temporal splits -- CA's labeling
    # collapsed exactly in the eras used for validation and test.
    handoff: dict = {"feature_contract": {
        "numeric": wf.ML_FEATURES_NUM, "categorical": wf.ML_FEATURES_CAT,
        "excluded_leaks": wf.ML_EXCLUDED_LEAKS,
        "splits": {"train": f"<= {wf.ML_TRAIN_MAX_YEAR}",
                   "val": f"{wf.ML_VAL_YEARS[0]}-{wf.ML_VAL_YEARS[1]}",
                   "test": f"{wf.ML_TEST_YEARS[0]}-{wf.ML_TEST_YEARS[1]}"}}}
    for st, df in (("ca", ca), ("fl", fl)):
        eras = {"train": df["fire_year"] <= wf.ML_TRAIN_MAX_YEAR,
                "val": df["fire_year"].between(*wf.ML_VAL_YEARS),
                "test": df["fire_year"].between(*wf.ML_TEST_YEARS)}
        block = {}
        for era, m in eras.items():
            sub = df[m]
            labeled = sub["cause_class"].isin(["Human", "Natural"])
            block[era] = {
                "n": int(len(sub)),
                "class_missing": round(float((~labeled).mean()), 4),
                "group_missing": round(float(sub["cause_group"].isna().mean()), 4),
                "natural_share_of_labeled": round(
                    float((sub.loc[labeled, "cause_class"] == "Natural").mean()), 4),
            }
        handoff[f"{st}_label_by_era"] = block
    # near-duplicate overlap with the ML caches (same rule as the corpus audit)
    for st, df in (("ca", ca), ("fl", fl)):
        d = df.dropna(subset=["lat", "lon", "fire_size"])
        d = d[d["fire_size"] >= 10]
        key = pd.DataFrame({"k": list(zip(d["discovery_date"].dt.date,
                                          d["lat"].round(3), d["lon"].round(3))),
                            "fod_id": d["fod_id"], "size": d["fire_size"]})
        flagged: set = set()
        for _, grp in key.groupby("k"):
            if len(grp) < 2:
                continue
            rows = list(zip(grp["fod_id"], grp["size"]))
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    if a[0] != b[0] and abs(a[1] - b[1]) / max(a[1], b[1]) <= 0.10:
                        flagged.update((a[0], b[0]))
        handoff[f"{st}_tier1_dup_rows_in_cache"] = len(flagged)
    results["modeling_handoff"] = handoff
    print(f"  [hand] CA test-era class-missing: "
          f"{handoff['ca_label_by_era']['test']['class_missing']:.1%} "
          f"(train {handoff['ca_label_by_era']['train']['class_missing']:.1%})")

    (wf.DATA / "eda_results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] eda_results.json + 6 figures in {wf.FIG}")


if __name__ == "__main__":
    main()
