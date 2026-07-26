"""Inference ladder S0–S7 for cardiometabolic myths + multiverse."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.data.config import get_paths, load_config

warnings.filterwarnings("ignore")

PRIMARY_OUTCOMES = ["bmi", "waist", "hba1c", "sbp_mean", "hdl", "tg"]
BINARY_OUTCOMES = ["obesity", "hba1c_elevated", "cancer_ever"]


def _formula_cols(formula: str, data: pd.DataFrame) -> list[str]:
    """Columns needed for a patsy formula (conservative name match)."""
    needed = []
    for c in data.columns:
        # word-boundary-ish: avoid matching random substrings of long names
        if c in formula.replace("(", " ").replace(")", " ").replace("+", " ").replace("~", " ").split():
            needed.append(c)
        elif f"C({c})" in formula:
            needed.append(c)
    # always keep weight helper if present
    return needed


def _wls(formula: str, data: pd.DataFrame, weights: pd.Series):
    """WLS with sampling weights. Point estimates OK; SEs not design-based (no PSU/strata)."""
    d = data.copy()
    d["_w"] = weights.reindex(d.index)
    cols = _formula_cols(formula, d) + ["_w"]
    d = d.dropna(subset=[c for c in cols if c in d.columns])
    # normalize weights so scale does not confuse interpretation of residual variance
    w = d["_w"].astype(float)
    w = w / w.mean()
    model = smf.wls(formula, data=d, weights=w)
    return model.fit()


def _glm_binomial(formula: str, data: pd.DataFrame, weights: pd.Series | None = None):
    """
    Binomial GLM. NHANES weights are sampling weights, NOT frequency weights.

    Using raw w_mec as freq_weights invents millions of pseudo-observations and
    crushes SEs (p-values print as 0). We either use no weights or normalized
    weights as a rough sensitivity — never raw MEC totals as frequencies.
    """
    d = data.copy()
    cols = _formula_cols(formula, d)
    d = d.dropna(subset=[c for c in cols if c in d.columns])
    if weights is not None:
        w = weights.reindex(d.index).astype(float)
        w = w / w.mean()
        return smf.glm(formula, data=d, family=sm.families.Binomial(), freq_weights=w).fit()
    return smf.glm(formula, data=d, family=sm.families.Binomial()).fit()


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["asb_only"] = (d["bev_group"] == "ASB-only").astype(int)
    d["ssb_only"] = (d["bev_group"] == "SSB-only").astype(int)
    d["both"] = (d["bev_group"] == "Both").astype(int)
    d["asb_vs_ssb"] = np.where(
        d["bev_group"] == "ASB-only",
        1,
        np.where(d["bev_group"] == "SSB-only", 0, np.nan),
    )
    d["log_tg"] = np.log(d["tg"].clip(lower=1))
    d["log_asb_g"] = np.log1p(d["asb_g_d1"])
    return d


def run_models(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    df = pd.read_parquet(paths["processed"] / "analysis_ready.parquet")
    d = _prep(df)
    w = d["w_mec"].fillna(0)
    rows = []

    def add_row(outcome, step, formula, subset=None, note=""):
        data = d if subset is None else d.loc[subset].copy()
        ww = w.loc[data.index]
        try:
            fit = _wls(formula, data, ww)
            # coefficient of interest: asb_only or asb_vs_ssb or asb_g_d1
            for term in ("asb_only", "asb_vs_ssb", "asb_g_d1", "asb_serv_d1", "log_asb_g"):
                if term in fit.params.index:
                    rows.append(
                        {
                            "outcome": outcome,
                            "step": step,
                            "term": term,
                            "coef": float(fit.params[term]),
                            "se": float(fit.bse[term]),
                            "pval": float(fit.pvalues[term]),
                            "n": int(fit.nobs),
                            "formula": formula,
                            "note": note,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "outcome": outcome,
                    "step": step,
                    "term": "ERROR",
                    "coef": np.nan,
                    "se": np.nan,
                    "pval": np.nan,
                    "n": 0,
                    "formula": formula,
                    "note": str(exc),
                }
            )

    base_cov = {
        "S0": "asb_only + ssb_only + both",
        "S1": "asb_only + ssb_only + both + age + female",
        "S2": "asb_only + ssb_only + both + age + female + C(race_eth) + education + pir",
        "S3": "asb_only + ssb_only + both + age + female + C(race_eth) + education + pir + smoking_status + total_kcal_d1",
    }

    for y in PRIMARY_OUTCOMES:
        y_use = "log_tg" if y == "tg" else y
        out_label = "log_tg" if y == "tg" else y
        scale_note = "outcome=log(TG); coef is log-mg/dL scale" if y == "tg" else "outcome=raw scale"
        for step, rhs in base_cov.items():
            add_row(out_label, step, f"{y_use} ~ {rhs}", note=scale_note)
        # S4 + bmi for non-adiposity
        if y not in ("bmi", "waist"):
            add_row(
                out_label,
                "S4",
                f"{y_use} ~ asb_only + ssb_only + both + age + female + C(race_eth) + education + pir + smoking_status + total_kcal_d1 + bmi",
                note=scale_note,
            )
        # S5 exclude known diabetes only (diabetes_sr == 0); do not use != 1 (NaN issue)
        add_row(
            out_label,
            "S5_no_dm",
            f"{y_use} ~ asb_only + ssb_only + both + age + female + C(race_eth) + education + pir + smoking_status + total_kcal_d1",
            subset=d["diabetes_sr"] == 0,
            note=scale_note + "; exclude self-report diabetes (diabetes_sr==0 only)",
        )
        # S6 continuous dose
        add_row(
            out_label,
            "S6_dose",
            f"{y_use} ~ asb_serv_d1 + ssb_serv_d1 + age + female + C(race_eth) + education + pir + smoking_status + total_kcal_d1",
            note=scale_note + ("; TG fasting lab" if y == "tg" else ""),
        )
        # S7 substitution ASB vs SSB
        add_row(
            out_label,
            "S7_sub",
            f"{y_use} ~ asb_vs_ssb + age + female + C(race_eth) + education + pir + smoking_status + total_kcal_d1",
            subset=d["asb_vs_ssb"].notna(),
            note=scale_note + "; ASB-only vs SSB-only",
        )

    # Re-fit TG primary steps on fasting weight where possible
    if "tg" in d.columns and "w_fast" in d.columns:
        d_fast = d.loc[d["w_fast"].fillna(0) > 0].copy()
        w_fast = d_fast["w_fast"]
        for step, rhs in base_cov.items():
            try:
                fit = _wls(f"log_tg ~ {rhs}", d_fast, w_fast)
                if "asb_only" in fit.params.index:
                    rows.append(
                        {
                            "outcome": "log_tg",
                            "step": step + "_fast",
                            "term": "asb_only",
                            "coef": float(fit.params["asb_only"]),
                            "se": float(fit.bse["asb_only"]),
                            "pval": float(fit.pvalues["asb_only"]),
                            "n": int(fit.nobs),
                            "formula": f"log_tg ~ {rhs}",
                            "note": "log-TG; fasting subsample + w_fast normalized",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "outcome": "log_tg",
                        "step": step + "_fast",
                        "term": "ERROR",
                        "coef": np.nan,
                        "se": np.nan,
                        "pval": np.nan,
                        "n": 0,
                        "formula": f"log_tg ~ {rhs}",
                        "note": str(exc),
                    }
                )

    # Binary logistic — normalized sampling weights (NOT raw frequency weights)
    for y in BINARY_OUTCOMES:
        if y not in d.columns:
            continue
        for step, rhs in [("S0", "asb_only + ssb_only + both"), ("S3", base_cov["S3"])]:
            data = d.dropna(subset=[y, "asb_only", "age", "female"])
            try:
                fit = _glm_binomial(f"{y} ~ {rhs}", data, weights=data["w_mec"])
                if "asb_only" in fit.params.index:
                    rows.append(
                        {
                            "outcome": y,
                            "step": step + "_logit",
                            "term": "asb_only",
                            "coef": float(fit.params["asb_only"]),
                            "se": float(fit.bse["asb_only"]),
                            "pval": float(fit.pvalues["asb_only"]),
                            "n": int(fit.nobs),
                            "formula": f"{y} ~ {rhs}",
                            "note": "logit log-OR; sampling weights NORMALIZED (not raw MEC as freq)",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "outcome": y,
                        "step": step + "_logit",
                        "term": "ERROR",
                        "coef": np.nan,
                        "se": np.nan,
                        "pval": np.nan,
                        "n": 0,
                        "formula": f"{y} ~ {rhs}",
                        "note": str(exc),
                    }
                )

    # Mortality logistic among LMF-eligible — normalized weights + unweighted sensitivity
    if "allcause_death" in d.columns:
        m = d[d.get("lmf_eligible", 0) == 1].copy()
        for y in ("allcause_death", "cancer_death"):
            if y not in m.columns:
                continue
            formula = f"{y} ~ asb_only + ssb_only + both + age + female + smoking_status"
            for label, use_w in [("mort_S1_logit", True), ("mort_S1_logit_unwt", False)]:
                try:
                    md = m.dropna(subset=[y, "asb_only", "age", "female"])
                    fit = _glm_binomial(
                        formula,
                        md,
                        weights=md["w_mec"] if use_w else None,
                    )
                    rows.append(
                        {
                            "outcome": y,
                            "step": label,
                            "term": "asb_only",
                            "coef": float(fit.params.get("asb_only", np.nan)),
                            "se": float(fit.bse.get("asb_only", np.nan)),
                            "pval": float(fit.pvalues.get("asb_only", np.nan)),
                            "n": int(fit.nobs),
                            "formula": formula,
                            "note": f"events={int(m[y].fillna(0).sum())}; "
                            + ("normalized w_mec" if use_w else "unweighted"),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "outcome": y,
                            "step": label,
                            "term": "ERROR",
                            "coef": np.nan,
                            "se": np.nan,
                            "pval": np.nan,
                            "n": 0,
                            "formula": formula,
                            "note": str(exc),
                        }
                    )

    res = pd.DataFrame(rows)
    res["ci_low"] = res["coef"] - 1.96 * res["se"]
    res["ci_high"] = res["coef"] + 1.96 * res["se"]
    out_path = paths["tables"] / "model_cardio_ladder.csv"
    res.to_csv(out_path, index=False)

    # Multiverse mini for BMI: exposure defs × steps
    multi = res[res["outcome"] == "bmi"].copy()
    multi.to_csv(paths["tables"] / "multiverse_bmi.csv", index=False)

    # Spec curve figure
    try:
        import matplotlib.pyplot as plt
        from src.analysis.plot_style import apply_style

        apply_style()
        plot = res[(res["outcome"] == "bmi") & (res["term"] == "asb_only")].copy()
        plot = plot.sort_values("coef")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(
            plot["coef"],
            range(len(plot)),
            xerr=1.96 * plot["se"],
            fmt="o",
            color="#1f77b4",
        )
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(range(len(plot)))
        ax.set_yticklabels(plot["step"])
        ax.set_xlabel("WLS coef: ASB-only vs Neither (BMI points)")
        ax.set_title("M7: BMI specification curve (ASB-only term)")
        fig.savefig(paths["figures"] / "myth_m7_spec_curve_bmi.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print("spec curve plot failed", exc)

    print(f"Wrote {out_path} ({len(res)} rows)")
    return res


def main():
    run_models()


if __name__ == "__main__":
    main()
