"""
EDA diagnostics for A+ science pass: leads placebo, type gradients.
  py -3 run_eda_diagnostics.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from models import fit_ols, fmt_pct, pct_ci
from panel import DATA, HAC_DEFAULT, ROOT, build_panel

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

COL = {
    "violent": "#e34948",
    "battery": "#c0392b",
    "theft": "#2a78d6",
    "robbery": "#8e44ad",
    "assault": "#e67e22",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def style_ax(ax):
    ax.grid(True, axis="y", color=COL["grid"], lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    dfa, _ = build_panel()
    hac = HAC_DEFAULT
    out = {}

    d = dfa.copy()
    d["logv"] = np.log(d["violent"])
    d["logv_dm"] = d["logv"] - d.groupby("ym")["logv"].transform("mean")
    d["x_dm"] = d["tmax10"] - d.groupby("ym")["tmax10"].transform("mean")
    for k in range(1, 4):
        d[f"xf{k}"] = d[f"tmax10_F{k}"] - d.groupby("ym")[f"tmax10_F{k}"].transform("mean")

    corrs = {
        "same_day": float(d["logv_dm"].corr(d["x_dm"])),
        "lead1": float(d["logv_dm"].corr(d["xf1"])),
        "lead2": float(d["logv_dm"].corr(d["xf2"])),
        "lead3": float(d["logv_dm"].corr(d["xf3"])),
    }
    out["residual_corrs"] = corrs
    print("residual corrs", corrs)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = list(corrs.keys())
    vals = [corrs[k] for k in labels]
    colors = [COL["violent"] if k == "same_day" else COL["ink2"] for k in labels]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color=COL["axis"])
    ax.set_ylabel("corr(log violent residual, temp residual)")
    ax.set_title("EDA: within-ym residual association — same-day vs future tmax")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "eda_leads_residual_corr.png", dpi=140)
    plt.close(fig)

    use = dfa.dropna(subset=["tmax10_F1", "tmax10_F2", "tmax10_F3"])
    res = fit_ols("violent", "tmax10 + tmax10_F1 + tmax10_F2 + tmax10_F3", use, hac=hac)
    lead_tab = {
        "same_day": pct_ci(res, "tmax10"),
        "lead1": pct_ci(res, "tmax10_F1"),
        "lead2": pct_ci(res, "tmax10_F2"),
        "lead3": pct_ci(res, "tmax10_F3"),
    }
    out["fe_leads"] = lead_tab
    for k, v in lead_tab.items():
        print(f"  FE {k}: {fmt_pct(v)}")

    types = ["battery", "robbery", "theft", "assault"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, y in zip(axes.ravel(), types):
        dd = dfa[dfa[y] > 0].copy()
        dd["logy"] = np.log(dd[y])
        dd["y_dm"] = dd["logy"] - dd.groupby("ym")["logy"].transform("mean")
        dd["x_dm"] = dd["tmax"] - dd.groupby("ym")["tmax"].transform("mean")
        ax.scatter(
            dd["x_dm"],
            dd["y_dm"],
            s=6,
            alpha=0.12,
            color=COL.get(y, COL["violent"]),
            linewidths=0,
        )
        sm = lowess(dd["y_dm"], dd["x_dm"], frac=0.35)
        ax.plot(sm[:, 0], sm[:, 1], color="black", lw=1.5)
        ax.axhline(0, color=COL["axis"], lw=0.8)
        ax.axvline(0, color=COL["axis"], lw=0.8)
        r = float(dd["x_dm"].corr(dd["y_dm"]))
        ax.set_title(f"{y} (resid r={r:.2f})")
        style_ax(ax)
    axes[1, 0].set_xlabel("tmax residual °F")
    axes[1, 1].set_xlabel("tmax residual °F")
    axes[0, 0].set_ylabel("log count residual")
    axes[1, 0].set_ylabel("log count residual")
    fig.suptitle("EDA: type-level within-ym residual heat gradients", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "eda_type_residual_gradients.png", dpi=140)
    plt.close(fig)

    type_fe = {}
    for y in types + ["violent", "property"]:
        use = dfa[dfa[y] > 0]
        type_fe[y] = pct_ci(fit_ols(y, "tmax10", use, hac=hac), "tmax10")
        print(f"  FE {y}: {fmt_pct(type_fe[y])}")
    out["type_fe"] = type_fe

    def _default(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(type(o))

    (DATA / "eda_diagnostics.json").write_text(
        json.dumps(out, indent=2, default=_default), encoding="utf-8"
    )
    print("Wrote data/eda_diagnostics.json and figures/eda_*.png")


if __name__ == "__main__":
    main()
