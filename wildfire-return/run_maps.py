from __future__ import annotations

"""Part II redeemed: static maps that carry the argument, zero JavaScript.

2020's Part II dropped thousands of identical leaflet pins -- a 300,000-acre
fire rendered exactly like a 1,200-acre one. These maps fix that:

    article_map_conus_pair.png   ignition count vs burned acres, same 0.1-degree
                                 cells, same projection -- the Southeast lights
                                 up by count, the West by acres
    article_map_ca_october.png   CA large fires, summer vs fall wind season,
                                 sized by acreage, the Tubbs Fire starred
    article_map_fl_spring.png    FL large fires by season: the spring-peak
                                 regime the 2020 FL section never described

Projection: hand-rolled Albers equal-area conic (parallels 29.5/45.5, the same
projection 2020-me reached for in ggplot) for CONUS; plate carree with cosine
aspect for single states. State lines from the Census 1:20m boundaries, no geo
dependencies.

Outputs the three PNGs + data/maps_summary.json.
Run (offline):  python run_maps.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import wildfire as wf

COL = {"fire": "#d9481f", "blue": "#2a78d6", "green": "#1baf7a", "gold": "#eda100",
       "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}


# ---------------------------------------------------------------- Albers projection
LAT0, LON0, PHI1, PHI2 = 23.0, -96.0, 29.5, 45.5


def albers(lon, lat):
    """Albers equal-area conic (the 15 lines 2020-me got from ggplot for free)."""
    lon, lat = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
    lat0, lon0 = np.radians(LAT0), np.radians(LON0)
    p1, p2 = np.radians(PHI1), np.radians(PHI2)
    n = (np.sin(p1) + np.sin(p2)) / 2
    C = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(np.maximum(C - 2 * n * np.sin(lat), 0)) / n
    rho0 = np.sqrt(C - 2 * n * np.sin(lat0)) / n
    theta = n * (lon - lon0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def draw_states(ax, states: list[dict], proj: bool, color: str = "#c3c2b7", lw: float = 0.5):
    for st in states:
        for ring in st["rings"]:
            arr = np.asarray(ring)
            if proj:
                x, y = albers(arr[:, 0], arr[:, 1])
            else:
                x, y = arr[:, 0], arr[:, 1]
            ax.plot(x, y, color=color, lw=lw, zorder=3)


def clean_map_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_aspect("equal")


def savefig(fig, name: str) -> None:
    wf.FIG.mkdir(exist_ok=True)
    fig.savefig(wf.FIG / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig ] {name}")


def main() -> None:
    for f in ("conus_grid.parquet", "ca_fires.parquet", "fl_fires.parquet", "tubbs_record.json"):
        wf.verify_manifest(f, strict=True)
    grid = wf.load_grid()
    ca = wf.load_state_fires("CA")
    fl = wf.load_state_fires("FL")
    tubbs = wf.load_tubbs()["curated"]
    geo = wf.load_states_geo()["states"]
    S: dict = {}

    # -------------------------------------------------- 1+2. CONUS pair: counts vs acres
    cells = grid.groupby(["lat_bin", "lon_bin"], as_index=False).agg(
        n=("n", "sum"), acres=("acres", "sum"))
    cells = cells[(cells["lon_bin"] > -130) & (cells["lat_bin"] > 22) & (cells["lat_bin"] < 51)]
    x, y = albers(cells["lon_bin"] + 0.05, cells["lat_bin"] + 0.05)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 10.6))
    panels = [
        (axes[0], cells["n"], "Where fires START: reported ignitions per 0.1-degree cell, 1992-2020",
         "count of fires (log color scale)"),
        (axes[1], cells["acres"].clip(lower=0.1), "Where the ACRES are: burned area in the same cells",
         "burned acres (log color scale)"),
    ]
    for ax, vals, title, cbar_label in panels:
        sc = ax.scatter(x, y, c=vals, s=1.6, marker="s", cmap="YlOrRd",
                        norm=LogNorm(vmin=max(float(vals[vals > 0].min()), 1e-1),
                                     vmax=float(vals.max())), linewidths=0, zorder=2)
        draw_states(ax, geo, proj=True)
        ax.set_title(title, loc="left", fontsize=11, color=COL["ink"])
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.01)
        cb.set_label(cbar_label, fontsize=8, color=COL["ink2"])
        cb.ax.tick_params(labelsize=7, colors=COL["ink2"])
        clean_map_ax(ax)
    fig.tight_layout()
    savefig(fig, "article_map_conus_pair.png")
    S["conus_cells"] = int(len(cells))

    # -------------------------------------------------- 3. California, the wind season
    big = ca[ca["size_class"].isin(wf.LARGE_CLASSES)].copy()
    season = np.select([big["month"].isin([6, 7, 8]), big["month"].isin([9, 10, 11])],
                       ["summer", "fall"], default="other")
    ca_geo = [s for s in geo if s["stusps"] == "CA"]
    fig, ax = plt.subplots(figsize=(7.6, 8.6))
    draw_states(ax, ca_geo, proj=False, color=COL["axis"], lw=0.9)
    sizes = np.sqrt(big["fire_size"]) / 6
    for name, color, z in (("other", COL["muted"], 4), ("summer", COL["gold"], 5), ("fall", COL["fire"], 6)):
        m = season == name
        label = {"summer": "summer (Jun-Aug): lightning country", "fall": "fall (Sep-Nov): wind season",
                 "other": "winter-spring"}[name]
        ax.scatter(big.loc[m, "lon"], big.loc[m, "lat"], s=sizes[m], color=color, alpha=0.45,
                   linewidths=0, zorder=z, label=label)
    ax.scatter([tubbs["lon"]], [tubbs["lat"]], marker="*", s=380, color=COL["red"],
               edgecolors="white", linewidths=0.8, zorder=9)
    ax.annotate(f"TUBBS -- Oct 8, 2017\n{tubbs['fire_size']:,.0f} acres\nthe one that took our house",
                (tubbs["lon"], tubbs["lat"]), textcoords="offset points", xytext=(-168, 10),
                fontsize=8.6, color=COL["ink"],
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COL["red"], "lw": 1})
    biggest = big.loc[big["fire_size"].idxmax()]
    ax.annotate(f"{str(biggest['fire_name']).title()} Fire {int(biggest['fire_year'])}, "
                f"{biggest['fire_size'] / 1e6:.2f}M acres\n(largest member of the August Complex --\n"
                f"the FOD records component fires)",
                (biggest["lon"], biggest["lat"]), textcoords="offset points", xytext=(14, -2),
                fontsize=7.2, color=COL["ink2"])
    ax.set_aspect(1 / np.cos(np.radians(37)))
    ax.set_xlim(-125.2, -113.8)
    ax.set_ylim(32.2, 42.4)
    clean_map_ax(ax)
    ax.set_aspect(1 / np.cos(np.radians(37)))
    ax.legend(frameon=False, fontsize=8.6, loc="lower left", labelcolor=COL["ink2"], markerscale=0.5)
    ax.set_title("California fires >= 1,000 acres, 1992-2020 -- dot area = acreage",
                 loc="left", fontsize=11.5, color=COL["ink"])
    savefig(fig, "article_map_ca_october.png")
    S["ca_large_n"] = int(len(big))
    S["ca_fall_share"] = round(float((season == "fall").mean()), 4)
    S["ca_fall_acres_share"] = round(float(big.loc[season == "fall", "fire_size"].sum()
                                           / big["fire_size"].sum()), 4)
    S["tubbs"] = {"lat": tubbs["lat"], "lon": tubbs["lon"], "acres": tubbs["fire_size"],
                  "date": tubbs["discovery_date"][:10]}
    S["biggest_ca"] = {"name": str(biggest["fire_name"]), "acres": float(biggest["fire_size"]),
                       "year": int(biggest["fire_year"])}

    # -------------------------------------------------- 4. Florida, the spring regime
    fbig = fl[fl["size_class"].isin(wf.LARGE_CLASSES)].copy()
    fseason = np.where(fbig["month"].isin([2, 3, 4, 5]), "spring", "other")
    fl_geo = [s for s in geo if s["stusps"] == "FL"]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    draw_states(ax, fl_geo, proj=False, color=COL["axis"], lw=0.9)
    fsizes = np.sqrt(fbig["fire_size"]) / 4
    for name, color, z in (("other", COL["muted"], 4), ("spring", COL["blue"], 5)):
        m = fseason == name
        label = "spring (Feb-May): the Florida fire season" if name == "spring" else "rest of year"
        ax.scatter(fbig.loc[m, "lon"], fbig.loc[m, "lat"], s=fsizes[m], color=color, alpha=0.5,
                   linewidths=0, zorder=z, label=label)
    ax.set_aspect(1 / np.cos(np.radians(28)))
    ax.set_xlim(-87.8, -79.8)
    ax.set_ylim(24.4, 31.2)
    clean_map_ax(ax)
    ax.set_aspect(1 / np.cos(np.radians(28)))
    ax.legend(frameon=False, fontsize=8.6, loc="lower left", labelcolor=COL["ink2"], markerscale=0.5)
    ax.set_title("Florida fires >= 1,000 acres, 1992-2020 -- a spring story, flat over time",
                 loc="left", fontsize=11.5, color=COL["ink"])
    savefig(fig, "article_map_fl_spring.png")
    S["fl_large_n"] = int(len(fbig))
    S["fl_spring_share"] = round(float((fseason == "spring").mean()), 4)

    (wf.DATA / "maps_summary.json").write_text(json.dumps(S, indent=2))
    print("[done] maps_summary.json + 3 maps")


if __name__ == "__main__":
    main()
