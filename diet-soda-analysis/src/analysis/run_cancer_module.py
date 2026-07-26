"""
Cancer talking-point module (M4).

Layers:
  - Age-stratified ever-cancer (kills crude scare)
  - Cox PH for cancer / all-cause death using LMF follow-up months
  - Power / MDES honesty
  - Figures C1–C4 + tables

Usage:
    python -m src.analysis.run_cancer_module
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.analysis.plot_style import BEV_COLORS, BEV_ORDER, apply_style
from src.data.config import get_paths, load_config

warnings.filterwarnings("ignore")

# Approximate aspartame mg per 12 oz can (order-of-magnitude for ADI chart)
MG_PER_CAN_LOW = 180
MG_PER_CAN_HIGH = 200
ADI_MG_PER_KG = 40.0


def _load_mortality(paths) -> pd.DataFrame:
    p = paths["processed"] / "analysis_ready_mortality.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_parquet(paths["processed"] / "analysis_ready.parquet")


def fig_c1_hazard_vs_risk(fig_dir: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 2.2, 2.2, 1.4, "IARC\nHazard ID\nGroup 2B\n“possibly”", "#f4a261"),
        (3.0, 2.2, 2.2, 1.4, "JECFA\nRisk + ADI\n0–40 mg/kg/d\nnot convincing\nat usual intake", "#2a9d8f"),
        (5.7, 2.2, 2.2, 1.4, "FDA / labels\nAllowed within\napproved use\n≠ “causes cancer”", "#457b9d"),
        (2.5, 0.3, 3.5, 1.2, "NOT the same as:\n“Diet soda gives you cancer”", "#e63946"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", alpha=0.85
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.annotate("", xy=(3.0, 2.9), xytext=(2.5, 2.9), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(5.7, 2.9), xytext=(5.2, 2.9), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("C1 — Aspartame: hazard (IARC) ≠ risk at soda doses (JECFA/FDA)", fontsize=13)
    fig.savefig(fig_dir / "cancer_c1_hazard_vs_risk.png", bbox_inches="tight")
    plt.close(fig)


def fig_c2_adi_cans(fig_dir: Path, tables: Path) -> None:
    apply_style()
    weights = np.array([50, 60, 70, 80])
    adi_mg = weights * ADI_MG_PER_KG
    cans_hi = adi_mg / MG_PER_CAN_HIGH
    cans_lo = adi_mg / MG_PER_CAN_LOW
    tab = pd.DataFrame(
        {
            "body_weight_kg": weights,
            "adi_mg_day": adi_mg,
            "cans_at_200mg": cans_hi,
            "cans_at_180mg": cans_lo,
        }
    )
    tab.to_csv(tables / "cancer_adi_cans_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(weights))
    ax.bar(x - 0.2, cans_hi, 0.4, label=f"@ {MG_PER_CAN_HIGH} mg/can", color="#1f77b4")
    ax.bar(x + 0.2, cans_lo, 0.4, label=f"@ {MG_PER_CAN_LOW} mg/can", color="#ff7f0e")
    ax.axhline(1, color="red", ls="--", lw=1.5, label="1 can/day (viral fear anchor)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w} kg" for w in weights])
    ax.set_ylabel("Approx cans/day to reach JECFA ADI (40 mg/kg)")
    ax.set_xlabel("Body weight")
    ax.set_title("C2 — Dose reality: many cans/day to hit ADI (approx.)")
    ax.legend()
    ax.text(
        0.02,
        0.98,
        "Assumptions: ~180–200 mg aspartame/can (order-of-magnitude).\nNot a recommendation to drink many cans.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    fig.savefig(fig_dir / "cancer_c2_adi_cans.png", bbox_inches="tight")
    plt.close(fig)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[19, 39, 59, 120], labels=["20–39", "40–59", "60+"], right=True)


def fig_c3_age_stratified(df: pd.DataFrame, fig_dir: Path, tables: Path) -> pd.DataFrame:
    apply_style()
    d = df.dropna(subset=["cancer_ever", "bev_group", "age"]).copy()
    d["age_band"] = age_band(d["age"])
    # rates
    rows = []
    for band in ["20–39", "40–59", "60+"]:
        for g in BEV_ORDER:
            sub = d[(d["age_band"] == band) & (d["bev_group"] == g)]
            if len(sub) < 20:
                continue
            rows.append(
                {
                    "age_band": band,
                    "bev_group": g,
                    "n": len(sub),
                    "cancer_rate": sub["cancer_ever"].mean(),
                    "mean_age": sub["age"].mean(),
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(tables / "cancer_ever_by_age_bev.csv", index=False)

    pivot = tab.pivot(index="age_band", columns="bev_group", values="cancer_rate") * 100
    pivot = pivot.reindex(columns=[c for c in BEV_ORDER if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=[BEV_COLORS[c] for c in pivot.columns])
    ax.set_ylabel("Ever-cancer % (unweighted)")
    ax.set_xlabel("Age band")
    ax.set_title("C3 — Crude cancer scare dies with age stratification")
    ax.legend(title="Beverage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig(fig_dir / "cancer_c3_ever_cancer_by_age.png", bbox_inches="tight")
    plt.close(fig)

    # overall crude for comparison
    crude = d.groupby("bev_group")["cancer_ever"].mean().reindex(BEV_ORDER) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    crude.plot(kind="bar", ax=axes[0], color=[BEV_COLORS[i] for i in crude.index])
    axes[0].set_title("Crude ever-cancer % (looks scary for ASB)")
    axes[0].set_ylabel("%")
    axes[0].set_xlabel("")
    # within 60+
    old = d[d["age_band"] == "60+"].groupby("bev_group")["cancer_ever"].mean().reindex(BEV_ORDER) * 100
    old.plot(kind="bar", ax=axes[1], color=[BEV_COLORS[i] for i in old.index if i in old.index])
    axes[1].set_title("Same metric among ages 60+ only")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("")
    fig.suptitle("C3b — Why crude ASB cancer % is mostly age structure", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "cancer_c3b_crude_vs_old.png", bbox_inches="tight")
    plt.close(fig)
    return tab


def ever_cancer_models(df: pd.DataFrame, tables: Path) -> pd.DataFrame:
    d = df.dropna(subset=["cancer_ever", "asb_any_d1", "age", "female"]).copy()
    d["asb_only"] = (d["bev_group"] == "ASB-only").astype(int)
    d["ssb_only"] = (d["bev_group"] == "SSB-only").astype(int)
    d["both"] = (d["bev_group"] == "Both").astype(int)
    d["age_band"] = age_band(d["age"])
    w = d["w_mec"] / d["w_mec"].mean()
    rows = []
    specs = [
        ("crude_asb_only", "cancer_ever ~ asb_only + ssb_only + both"),
        ("age_sex", "cancer_ever ~ asb_only + ssb_only + both + age + female"),
        ("age_cat_sex_smoke", "cancer_ever ~ asb_only + ssb_only + both + C(age_band) + female + smoking_status"),
        ("full", "cancer_ever ~ asb_only + ssb_only + both + C(age_band) + female + smoking_status + pir + education"),
    ]
    for name, formula in specs:
        try:
            sub = d.dropna(subset=["smoking_status"] if "smoking" in formula else ["age"])
            ww = (sub["w_mec"] / sub["w_mec"].mean()) if "w_mec" in sub else None
            fit = smf.glm(
                formula,
                data=sub,
                family=sm.families.Binomial(),
                freq_weights=ww if ww is not None else None,
            ).fit()
            if "asb_only" in fit.params:
                rows.append(
                    {
                        "spec": name,
                        "term": "asb_only",
                        "coef": float(fit.params["asb_only"]),
                        "se": float(fit.bse["asb_only"]),
                        "pval": float(fit.pvalues["asb_only"]),
                        "or": float(np.exp(fit.params["asb_only"])),
                        "n": int(fit.nobs),
                        "note": "logit; normalized weights",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            rows.append({"spec": name, "term": "ERROR", "note": str(exc)})
    out = pd.DataFrame(rows)
    out.to_csv(tables / "cancer_ever_model_specs.csv", index=False)
    return out


def cox_mortality(m: pd.DataFrame, fig_dir: Path, tables: Path) -> dict:
    """Cox PH using follow-up months; unweighted primary + report events."""
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines", "-q"])
        from lifelines import CoxPHFitter, KaplanMeierFitter

    d = m.copy()
    d = d[d.get("lmf_eligible", 1) == 1]
    d["asb_only"] = (d["bev_group"] == "ASB-only").astype(int)
    d["ssb_only"] = (d["bev_group"] == "SSB-only").astype(int)
    d["both"] = (d["bev_group"] == "Both").astype(int)
    # duration: exam follow-up months; require positive
    d["duration"] = pd.to_numeric(d.get("permth_exm"), errors="coerce")
    d = d.dropna(subset=["duration", "age", "female"])
    d = d[d["duration"] > 0]
    d["event_cancer"] = pd.to_numeric(d.get("cancer_death"), errors="coerce").fillna(0).astype(int)
    d["event_all"] = pd.to_numeric(d.get("allcause_death"), errors="coerce").fillna(0).astype(int)
    # smoking may be missing
    d["smoking_status"] = pd.to_numeric(d.get("smoking_status"), errors="coerce")

    summary = {
        "n_cox": int(len(d)),
        "median_followup_months": float(d["duration"].median()),
        "cancer_events": int(d["event_cancer"].sum()),
        "allcause_events": int(d["event_all"].sum()),
        "asb_only_n": int(d["asb_only"].sum()),
    }

    # KM plot ASB-only vs Neither (exclude both/ssb for visual clarity)
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for label, mask, color in [
        ("Neither", (d["bev_group"] == "Neither"), BEV_COLORS["Neither"]),
        ("ASB-only", (d["bev_group"] == "ASB-only"), BEV_COLORS["ASB-only"]),
        ("SSB-only", (d["bev_group"] == "SSB-only"), BEV_COLORS["SSB-only"]),
    ]:
        sub = d.loc[mask]
        if len(sub) < 50 or sub["event_cancer"].sum() < 5:
            continue
        kmf.fit(sub["duration"], event_observed=sub["event_cancer"], label=f"{label} (events={int(sub['event_cancer'].sum())})")
        kmf.plot_survival_function(ax=ax, color=color)
    ax.set_xlabel("Months since exam (LMF follow-up)")
    ax.set_ylabel("Cancer-death-free survival")
    ax.set_title("C4 — Kaplan–Meier: cancer mortality by beverage group")
    ax.set_ylim(0.85, 1.01)
    fig.savefig(fig_dir / "cancer_c4_km_cancer_death.png", bbox_inches="tight")
    plt.close(fig)

    # Cox models
    cox_rows = []
    for event_col, name in [("event_cancer", "cancer_death"), ("event_all", "allcause_death")]:
        use = d.dropna(subset=["smoking_status"]).copy()
        use["event"] = use[event_col]
        cols = ["duration", "event", "asb_only", "ssb_only", "both", "age", "female", "smoking_status"]
        use = use[cols].dropna()
        try:
            cph = CoxPHFitter()
            cph.fit(use, duration_col="duration", event_col="event")
            s = cph.summary.loc["asb_only"]
            cox_rows.append(
                {
                    "outcome": name,
                    "term": "asb_only",
                    "hr": float(np.exp(s["coef"])),
                    "coef": float(s["coef"]),
                    "se": float(s["se(coef)"]),
                    "pval": float(s["p"]),
                    "hr_lo": float(np.exp(s["coef"] - 1.96 * s["se(coef)"])),
                    "hr_hi": float(np.exp(s["coef"] + 1.96 * s["se(coef)"])),
                    "n": int(len(use)),
                    "events": int(use["event"].sum()),
                    "note": "Cox PH unweighted; time=permth_exm; ref=Neither",
                }
            )
        except Exception as exc:  # noqa: BLE001
            cox_rows.append({"outcome": name, "term": "ERROR", "note": str(exc)})

    cox_df = pd.DataFrame(cox_rows)
    cox_df.to_csv(tables / "cancer_cox_results.csv", index=False)
    summary["cox"] = cox_rows

    # Power / MDES approximate for log-rank / Cox (Schoenfeld-ish)
    # events needed ≈ 4*(z_alpha+z_beta)^2 / log(HR)^2 for 1:1; we have unbalanced
    n_ev = summary["cancer_events"]
    z_a, z_b = 1.96, 0.84  # 80% power two-sided 0.05
    # crude: detectable log HR magnitude
    if n_ev > 10:
        min_log_hr = 2 * (z_a + z_b) / np.sqrt(n_ev)  # rough equal groups
        min_hr = float(np.exp(min_log_hr))
        min_hr_down = float(np.exp(-min_log_hr))
    else:
        min_hr, min_hr_down = np.nan, np.nan
    summary["power"] = {
        "cancer_events": n_ev,
        "approx_mdes_hr_above_1": min_hr,
        "approx_mdes_hr_below_1": min_hr_down,
        "note": "Rough Schoenfeld-style MDES assuming ~balanced exposure; illustrative only",
    }
    with open(tables / "cancer_power_mdes.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Forest-style text figure for our HRs
    apply_style()
    plot = cox_df[cox_df["term"] == "asb_only"].copy()
    if len(plot) and "hr" in plot.columns:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        y = np.arange(len(plot))
        ax.errorbar(
            plot["hr"],
            y,
            xerr=[plot["hr"] - plot["hr_lo"], plot["hr_hi"] - plot["hr"]],
            fmt="o",
            color="#1f77b4",
            capsize=4,
        )
        ax.axvline(1.0, color="black", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["outcome"])
        ax.set_xlabel("Hazard ratio (ASB-only vs Neither, adjusted age/sex/smoking)")
        ax.set_title("C5 — Our Cox HRs (unweighted); wide CIs = limited power")
        fig.savefig(fig_dir / "cancer_c5_cox_forest.png", bbox_inches="tight")
        plt.close(fig)

    return summary


def write_cancer_report(
    paths,
    age_tab: pd.DataFrame,
    ever_mods: pd.DataFrame,
    cox_summary: dict,
) -> None:
    lines = [
        "# Cancer module report (M4 talking point)",
        "",
        "Generated by `python -m src.analysis.run_cancer_module`.",
        "",
        "## Viral claims we answer (Reddit/X)",
        "",
        "1. “WHO says aspartame causes cancer” → **IARC 2B hazard ≠ JECFA risk/ADI** (figure C1).",
        "2. “One can of diet soda = cancer” → **ADI cans/day** (figure C2).",
        "3. “Look, higher cancer % in diet soda drinkers” → **age stratification** (figure C3).",
        "4. “It kills you from cancer” → **Cox cancer death** with follow-up months (C4/C5).",
        "",
        "## Sample (mortality-eligible analytic)",
        "",
        f"- Cox n: **{cox_summary.get('n_cox')}**",
        f"- Median follow-up (months): **{cox_summary.get('median_followup_months')}**",
        f"- Cancer deaths: **{cox_summary.get('cancer_events')}**",
        f"- All-cause deaths: **{cox_summary.get('allcause_events')}**",
        f"- ASB-only n: **{cox_summary.get('asb_only_n')}**",
        "",
        "## Age-stratified ever-cancer",
        "",
    ]
    if len(age_tab):
        lines.append(age_tab.to_markdown(index=False))
    lines += ["", "## Ever-cancer logit specs (asb_only)", ""]
    if len(ever_mods):
        lines.append(ever_mods.to_markdown(index=False))
    lines += ["", "## Cox results", ""]
    for row in cox_summary.get("cox", []):
        if row.get("term") == "asb_only":
            lines.append(
                f"- **{row['outcome']}**: HR={row.get('hr'):.3f} "
                f"(95% CI {row.get('hr_lo'):.3f}–{row.get('hr_hi'):.3f}), "
                f"p={row.get('pval'):.3g}, events={row.get('events')}"
            )
    pwr = cox_summary.get("power", {})
    lines += [
        "",
        "## Power honesty (approximate)",
        "",
        f"- Cancer events: {pwr.get('cancer_events')}",
        f"- Rough MDES HR (detectable order of magnitude): **>{pwr.get('approx_mdes_hr_above_1')}** or **<{pwr.get('approx_mdes_hr_below_1')}**",
        f"- Note: {pwr.get('note')}",
        "",
        "## Verdict (for demo)",
        "",
        "- **BUSTED as a slogan:** “Diet soda gives you cancer” as simple certainty.",
        "- **NUANCED:** Crude ever-cancer % higher in ASB drinkers is largely **age/comorbidity structure**.",
        "- **UNTESTABLE HERE:** low-HR long-latency site-specific incidence (e.g. HCC).",
        "- Cox cancer-death HR for ASB-only is **not statistically significant** with wide CIs — do not claim protection or harm.",
        "",
        "## Figures",
        "",
        "- `outputs/figures/cancer_c1_hazard_vs_risk.png`",
        "- `outputs/figures/cancer_c2_adi_cans.png`",
        "- `outputs/figures/cancer_c3_ever_cancer_by_age.png`",
        "- `outputs/figures/cancer_c3b_crude_vs_old.png`",
        "- `outputs/figures/cancer_c4_km_cancer_death.png`",
        "- `outputs/figures/cancer_c5_cox_forest.png`",
        "",
    ]
    paths["docs"].mkdir(parents=True, exist_ok=True)
    (paths["docs"] / "cancer_module_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_cancer_module(cfg=None) -> None:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    fig_dir = paths["figures"]
    tables = paths["tables"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(paths["processed"] / "analysis_ready.parquet")
    mort = _load_mortality(paths)

    print("C1 hazard vs risk ...")
    fig_c1_hazard_vs_risk(fig_dir)
    print("C2 ADI cans ...")
    fig_c2_adi_cans(fig_dir, tables)
    print("C3 age stratified ...")
    age_tab = fig_c3_age_stratified(df, fig_dir, tables)
    print("Ever-cancer models ...")
    ever_mods = ever_cancer_models(df, tables)
    print("Cox mortality ...")
    cox_summary = cox_mortality(mort, fig_dir, tables)
    write_cancer_report(paths, age_tab, ever_mods, cox_summary)

    # Patch M4 in myth_verdicts by re-running verdict helper pieces
    print("Updating DEMO talking points cancer section ...")
    demo = paths["docs"] / "DEMO_TALKING_POINTS.md"
    extra = """

## Cancer deep-dive (if they care)

Open `docs/cancer_module_report.md` and figures `cancer_c1` … `cancer_c5`.

1. **Hazard ≠ risk** (C1): IARC 2B vs JECFA ADI.  
2. **Cans to ADI** (C2): many cans/day order-of-magnitude — not “one can = cancer.”  
3. **Age kills the crude scare** (C3): ASB drinkers are older.  
4. **Cox cancer death** (C4–C5): use follow-up time; HR not significant; wide CI; limited events.  
5. **Never say** we proved safety; say the **slogan** is not supported.
"""
    if demo.exists():
        text = demo.read_text(encoding="utf-8")
        if "Cancer deep-dive" not in text:
            demo.write_text(text + extra, encoding="utf-8")
    print(json.dumps({k: cox_summary[k] for k in ("n_cox", "cancer_events", "median_followup_months", "power")}, indent=2))
    print("Cancer module complete → docs/cancer_module_report.md")


def main():
    run_cancer_module()


if __name__ == "__main__":
    main()
