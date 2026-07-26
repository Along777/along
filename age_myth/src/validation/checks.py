"""Validation checks and coverage summary for processed tables."""
from __future__ import annotations

import pandas as pd

from src.paths import OUTPUTS, PROCESSED, ensure_dirs

FACT = PROCESSED / "life_expectancy_long.csv"
MODELING = PROCESSED / "life_expectancy_modeling.csv"
SOURCES = PROCESSED / "sources.csv"

# Sources expected to be unique at (region_id, year, sex, age, source_id)
UNIQUE_GRAIN_SOURCES = {
    "hmd_summary_public",
    "eurostat_demo_mlexpec",
    "owid_le_longrun",
    "owid_le_age15",
    "owid_le_hmd_unwpp",
    "clio_zijdeman_2015",
}


def validate_fact(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors should block CI; warnings document risks."""
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "region_id",
        "country_region",
        "sex",
        "age",
        "life_expectancy",
        "measure_type",
        "population_type",
        "data_quality_flag",
        "source_id",
        "retrieved_at",
    ]
    for c in required:
        if c not in df.columns:
            errors.append(f"Missing column: {c}")
    if df.empty:
        errors.append("Fact table is empty")
        return errors, warnings

    le = pd.to_numeric(df["life_expectancy"], errors="coerce")
    bad_le = df[le.notna() & ((le <= 0) | (le >= 120))]
    if len(bad_le):
        errors.append(f"life_expectancy out of (0,120): {len(bad_le)} rows")

    if "survival_probability" in df.columns:
        sp = pd.to_numeric(df["survival_probability"], errors="coerce")
        bad_sp = df[sp.notna() & ((sp < 0) | (sp > 1.01))]
        if len(bad_sp):
            errors.append(f"survival_probability out of [0,1]: {len(bad_sp)} rows")

    ages = pd.to_numeric(df["age"], errors="coerce")
    if ages.isna().any():
        errors.append(f"Non-numeric age: {ages.isna().sum()} rows")

    # dual-series
    has0 = set(df.loc[df["age"] == 0, "source_id"].astype(str))
    has_adult = set(df.loc[df["age"].isin([15, 20, 30, 65]), "source_id"].astype(str))
    if not has0:
        errors.append("No age=0 rows found (cannot test Myth A baseline e0)")
    if not has_adult:
        warnings.append("No adult-age e(x) rows found (cannot test conditional adult LE)")

    # fixture leak
    n_fix = (df["source_id"].astype(str) == "hmd_fixture").sum()
    if n_fix:
        warnings.append(
            f"hmd_fixture demo rows present in fact table: {n_fix} — EXCLUDE from modeling"
        )

    # projections
    years = pd.to_numeric(df["year"], errors="coerce")
    n_proj = (years > 2023).sum()
    if n_proj:
        warnings.append(
            f"Rows with year > 2023 (projections / provisional): {n_proj:,} — filter for history"
        )

    # grain uniqueness for sources that should be unique
    grain = ["region_id", "year", "sex", "age", "source_id"]
    for sid in sorted(set(df["source_id"].astype(str)) & UNIQUE_GRAIN_SOURCES):
        sub = df[df["source_id"].astype(str) == sid]
        dups = sub.duplicated(subset=grain, keep=False)
        if dups.any():
            errors.append(
                f"Unexpected duplicate grain for {sid}: {dups.sum()} rows (should be unique)"
            )

    # HLD fan-out (expected warning, not error until Ref-ID fixed)
    hld = df[df["source_id"].astype(str) == "hld"]
    if len(hld):
        vc = hld.groupby(["region_id", "year", "sex", "age"]).size()
        multi = (vc > 1).sum()
        if multi:
            warnings.append(
                f"HLD multi-table grain: {multi:,} keys with >1 table "
                f"(mean={vc.mean():.1f}, max={vc.max()}) — do not naive-average; "
                f"use life_expectancy_modeling (excludes HLD) or hld_median view"
            )

    # sources dim coverage
    if SOURCES.exists():
        src = pd.read_csv(SOURCES)
        missing = set(df["source_id"].astype(str)) - set(src["source_id"].astype(str))
        if missing:
            warnings.append(f"source_id values missing from sources.csv: {sorted(missing)}")

    return errors, warnings


def coverage_report(df: pd.DataFrame, errors: list[str], warnings: list[str]) -> str:
    lines = ["# Coverage summary", ""]
    if df.empty:
        lines.append("Fact table empty.")
        return "\n".join(lines)

    lines.append(f"- **Rows:** {len(df):,}")
    lines.append(
        f"- **Sources:** {df['source_id'].nunique()} — {sorted(df['source_id'].dropna().unique())}"
    )
    lines.append(f"- **Regions:** {df['region_id'].nunique()}")
    ages_all = sorted(int(a) for a in pd.to_numeric(df["age"], errors="coerce").dropna().unique())
    lines.append(f"- **Ages:** {ages_all}")
    years = pd.to_numeric(df["year"], errors="coerce")
    if years.notna().any():
        lines.append(f"- **Year range:** {int(years.min())}–{int(years.max())}")
    lines.append("")
    lines.append("## Rows by source_id")
    lines.append("")
    lines.append("| source_id | rows | ages | year min | year max |")
    lines.append("|-----------|------|------|----------|----------|")
    for sid, g in df.groupby("source_id"):
        ys = pd.to_numeric(g["year"], errors="coerce")
        ages = sorted(int(a) for a in pd.to_numeric(g["age"], errors="coerce").dropna().unique())
        ymin = int(ys.min()) if ys.notna().any() else ""
        ymax = int(ys.max()) if ys.notna().any() else ""
        lines.append(f"| {sid} | {len(g):,} | {ages} | {ymin} | {ymax} |")
    lines.append("")
    lines.append("## Dual-myth readiness")
    lines.append("")
    e0 = df[df["age"] == 0]
    e15 = df[df["age"] == 15]
    e65 = df[df["age"] == 65]
    lines.append(f"- e0 rows: **{len(e0):,}**")
    lines.append(f"- e15 rows: **{len(e15):,}**")
    lines.append(f"- e65 rows: **{len(e65):,}**")
    imr_n = df["infant_mortality_rate"].notna().sum() if "infant_mortality_rate" in df.columns else 0
    lines.append(f"- IMR non-null rows: **{imr_n:,}**")
    lines.append(
        f"- Quality flags present: {sorted(df['data_quality_flag'].dropna().unique())}"
    )
    lines.append("")
    lines.append("## Modeling recommendation")
    lines.append("")
    lines.append(
        "- Prefer `data/processed/life_expectancy_modeling.parquet` "
        "(excludes fixture, projections, raw HLD fan-out)."
    )
    lines.append(
        "- Gold long-run: `source_id == 'hmd_summary_public'` (e0/e65/e80 + IMR + S→65)."
    )
    lines.append(
        "- EU multi-age: `source_id == 'eurostat_demo_mlexpec'` (ISO2 geos)."
    )
    lines.append("")
    lines.append("## Validation errors")
    lines.append("")
    if errors:
        for i in errors:
            lines.append(f"- **ERROR:** {i}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Validation warnings")
    lines.append("")
    if warnings:
        for i in warnings:
            lines.append(f"- **WARN:** {i}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    if not FACT.exists():
        print(f"Missing {FACT}. Run loaders first.")
        raise SystemExit(1)
    df = pd.read_csv(FACT, low_memory=False)
    errors, warnings = validate_fact(df)
    report = coverage_report(df, errors, warnings)
    out = OUTPUTS / "coverage_summary.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")
    if MODELING.exists():
        print(f"Modeling view present: {MODELING}")
    else:
        print("Modeling view not built yet — run: python -m src.cleaning.build_modeling_view")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
