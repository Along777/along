"""Create empty processed table templates, seed sources, methodology notes, literature benchmarks."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from src.paths import PROCESSED, RAW, ensure_dirs

FACT_COLUMNS = [
    "region_id",
    "country_region",
    "year",
    "period_start",
    "period_end",
    "sex",
    "age",
    "life_expectancy",
    "survival_probability",
    "infant_mortality_rate",
    "measure_type",
    "population_type",
    "table_type",
    "data_quality_flag",
    "source_id",
    "notes",
    "retrieved_at",
]

SOURCES_COLUMNS = [
    "source_id",
    "name",
    "citation",
    "url",
    "license",
    "access_notes",
    "version_or_retrieved",
]

REGIONS_COLUMNS = [
    "region_id",
    "name",
    "iso3",
    "hmd_code",
    "hld_name",
    "region_type",
    "continent",
    "coverage_notes",
]

NOTES_COLUMNS = [
    "note_id",
    "topic",
    "applies_to_source_id",
    "applies_to_region_id",
    "text",
    "severity",
]

LIT_COLUMNS = FACT_COLUMNS + ["paper_citation"]


def _write_csv(path: Path, columns: list[str], rows: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        if rows:
            for r in rows:
                w.writerow({c: r.get(c, "") for c in columns})


def seed_sources() -> list[dict]:
    today = date.today().isoformat()
    return [
        {
            "source_id": "owid_le_longrun",
            "name": "OWID Life expectancy (long-run)",
            "citation": "Our World in Data based on HMD, Zijdeman et al., Riley, UN WPP (compilation).",
            "url": "https://ourworldindata.org/grapher/life-expectancy",
            "license": "CC BY; cite original providers",
            "access_notes": "Open CSV API",
            "version_or_retrieved": today,
        },
        {
            "source_id": "owid_le_age15",
            "name": "OWID Life expectancy at age 15",
            "citation": "Human Mortality Database; UN WPP — with processing by Our World in Data.",
            "url": "https://ourworldindata.org/grapher/life-expectancy-at-age-15",
            "license": "CC BY; cite original providers",
            "access_notes": "Open CSV API",
            "version_or_retrieved": today,
        },
        {
            "source_id": "owid_le_hmd_unwpp",
            "name": "OWID Life expectancy HMD + UN WPP",
            "citation": "Human Mortality Database; UN World Population Prospects — OWID processing.",
            "url": "https://ourworldindata.org/grapher/life-expectancy-hmd-unwpp",
            "license": "CC BY; cite original providers",
            "access_notes": "Open CSV API",
            "version_or_retrieved": today,
        },
        {
            "source_id": "hld",
            "name": "Human Life-Table Database",
            "citation": "Human Life-Table Database. Max Planck Institute for Demographic Research and others. https://www.lifetable.de/",
            "url": "https://www.lifetable.de/",
            "license": "See HLD/HMD terms; free access, no registration",
            "access_notes": "Bulk zip / country files; methods vary by original table",
            "version_or_retrieved": today,
        },
        {
            "source_id": "hmd_v6",
            "name": "Human Mortality Database",
            "citation": "HMD. Human Mortality Database. Max Planck Institute for Demographic Research (Germany), University of California, Berkeley (USA), and French Institute for Demographic Studies (France). Available at www.mortality.org.",
            "url": "https://www.mortality.org/",
            "license": "Free with registration; accept HMD user agreement",
            "access_notes": "Set HMD_USER and HMD_PASSWORD env vars",
            "version_or_retrieved": today,
        },
        {
            "source_id": "clio_zijdeman_2015",
            "name": "Clio-Infra Life Expectancy at Birth (Zijdeman)",
            "citation": "Zijdeman, R. and Ribeira da Silva, F. (2015). Life Expectancy at Birth (Total). http://hdl.handle.net/10622/LKYT53",
            "url": "http://hdl.handle.net/10622/LKYT53",
            "license": "See Clio-Infra / IISH terms",
            "access_notes": "Historical e0 compilation",
            "version_or_retrieved": today,
        },
        {
            "source_id": "gurven_kaplan_2007",
            "name": "Gurven & Kaplan (2007) hunter-gatherer longevity",
            "citation": "Gurven, M. & Kaplan, H. (2007). Longevity among hunter-gatherers: A cross-cultural examination. Population and Development Review, 33(2), 321–365.",
            "url": "https://doi.org/10.1111/j.1728-4457.2007.00171.x",
            "license": "Literature extract for research citation",
            "access_notes": "Curated table values; not national VR",
            "version_or_retrieved": today,
        },
        {
            "source_id": "hmd_fixture",
            "name": "HMD-format demo fixture",
            "citation": "Synthetic sample in HMD life-table layout for pipeline tests (not real HMD data).",
            "url": "",
            "license": "Project internal",
            "access_notes": "data/raw/fixtures/ — EXCLUDE from modeling",
            "version_or_retrieved": today,
        },
        {
            "source_id": "hmd_summary_public",
            "name": "HMD Public Summary Indicators",
            "citation": "HMD. Human Mortality Database. Public summary indicators (IMR; survival 0–65; e0/e65/e80). Available at www.mortality.org (no registration).",
            "url": "https://www.mortality.org/",
            "license": "Free; cite HMD",
            "access_notes": "Public Excel summaries under /Public/HMD_summary/",
            "version_or_retrieved": today,
        },
        {
            "source_id": "owid_imr",
            "name": "OWID Infant mortality",
            "citation": "Our World in Data infant mortality series (underlying sources vary).",
            "url": "https://ourworldindata.org/grapher/infant-mortality",
            "license": "CC BY; cite original providers",
            "access_notes": "Open CSV; joined onto OWID e0 rows where keys match",
            "version_or_retrieved": today,
        },
        {
            "source_id": "owid_u5mr",
            "name": "OWID Child (under-5) mortality",
            "citation": "Our World in Data child mortality series.",
            "url": "https://ourworldindata.org/grapher/child-mortality",
            "license": "CC BY; cite original providers",
            "access_notes": "Stored in interim owid_mortality_long.csv",
            "version_or_retrieved": today,
        },
        {
            "source_id": "eurostat_demo_mlexpec",
            "name": "Eurostat life expectancy by age and sex",
            "citation": "Eurostat dataset demo_mlexpec — Life expectancy by age and sex.",
            "url": "https://ec.europa.eu/eurostat/databrowser/product/view/demo_mlexpec",
            "license": "Eurostat free reuse with attribution",
            "access_notes": "SDMX-CSV bulk download",
            "version_or_retrieved": today,
        },
    ]


def seed_regions() -> list[dict]:
    return [
        {
            "region_id": "SWE",
            "name": "Sweden",
            "iso3": "SWE",
            "hmd_code": "SWE",
            "hld_name": "Sweden",
            "region_type": "country",
            "continent": "Europe",
            "coverage_notes": "HMD period tables from 1751",
        },
        {
            "region_id": "GBRTENW",
            "name": "England and Wales",
            "iso3": "GBR",
            "hmd_code": "GBRTENW",
            "hld_name": "England and Wales",
            "region_type": "country",
            "continent": "Europe",
            "coverage_notes": "HMD England & Wales; Cambridge Group for earlier periods",
        },
        {
            "region_id": "FRATNP",
            "name": "France",
            "iso3": "FRA",
            "hmd_code": "FRATNP",
            "hld_name": "France",
            "region_type": "country",
            "continent": "Europe",
            "coverage_notes": "Long HMD series; used in OWID multi-age examples",
        },
        {
            "region_id": "JPN",
            "name": "Japan",
            "iso3": "JPN",
            "hmd_code": "JPN",
            "hld_name": "Japan",
            "region_type": "country",
            "continent": "Asia",
            "coverage_notes": "HMD modern series",
        },
        {
            "region_id": "OWID_WORLD",
            "name": "World",
            "iso3": "",
            "hmd_code": "",
            "hld_name": "",
            "region_type": "aggregate",
            "continent": "World",
            "coverage_notes": "OWID global aggregates",
        },
        {
            "region_id": "FORAGER_COMPOSITE",
            "name": "Hunter-gatherer composite (Gurven & Kaplan)",
            "iso3": "",
            "hmd_code": "",
            "hld_name": "",
            "region_type": "population_group",
            "continent": "Multi",
            "coverage_notes": "Anthropological samples; not national VR",
        },
    ]


def seed_notes() -> list[dict]:
    return [
        {
            "note_id": "dual_myth",
            "topic": "dual_myth",
            "applies_to_source_id": "",
            "applies_to_region_id": "",
            "text": (
                "Myth A: e0~30 does not mean adults died at 30. "
                "Myth B: surviving childhood did not yield modern adult e(x). "
                "Always compare multi-age e(x) over time."
            ),
            "severity": "info",
        },
        {
            "note_id": "owid_stitch",
            "topic": "source_methods",
            "applies_to_source_id": "owid_le_longrun",
            "applies_to_region_id": "",
            "text": "OWID long-run e0 stitches HMD, Zijdeman/Clio-Infra, Riley, and UN WPP. Not a single method.",
            "severity": "caveat",
        },
        {
            "note_id": "hld_hetero",
            "topic": "source_methods",
            "applies_to_source_id": "hld",
            "applies_to_region_id": "",
            "text": "HLD aggregates published life tables with heterogeneous construction methods. Prefer HMD for gold-standard national series when both exist.",
            "severity": "caveat",
        },
        {
            "note_id": "hmd_registration",
            "topic": "access",
            "applies_to_source_id": "hmd_v6",
            "applies_to_region_id": "",
            "text": "HMD requires free registration. Full country series need HMD_USER/HMD_PASSWORD.",
            "severity": "info",
        },
        {
            "note_id": "forager_modal",
            "topic": "interpretation",
            "applies_to_source_id": "gurven_kaplan_2007",
            "applies_to_region_id": "FORAGER_COMPOSITE",
            "text": "Modal adult lifespan ~68-78 is not e0. Gurven & Kaplan e0 for traditional hunter-gatherers is typically low 20s to mid 30s.",
            "severity": "limitation",
        },
        {
            "note_id": "no_paleodemography_in_fact",
            "topic": "exclusions",
            "applies_to_source_id": "",
            "applies_to_region_id": "",
            "text": "Mean age at death from cemeteries is not stored as life expectancy in the main fact table.",
            "severity": "limitation",
        },
    ]


def seed_literature_benchmarks() -> list[dict]:
    """Approximate values from Gurven & Kaplan (2007) summary ranges for illustration.

    These are curated educational benchmarks, not a full digitization of every group.
    See paper tables for group-level detail.
    """
    today = date.today().isoformat()
    paper = "Gurven & Kaplan (2007) PDR; curated composite/ranges for Phase 1 seed"
    base = {
        "period_start": 1950,
        "period_end": 2000,
        "year": "",
        "sex": "both",
        "measure_type": "period",
        "population_type": "forager_horticultural",
        "table_type": "literature_summary",
        "data_quality_flag": "literature_extract",
        "source_id": "gurven_kaplan_2007",
        "retrieved_at": today,
        "paper_citation": paper,
        "infant_mortality_rate": "",
    }
    return [
        {
            **base,
            "region_id": "FORAGER_HG_TRAD",
            "country_region": "Traditional hunter-gatherers (composite range mid)",
            "age": 0,
            "life_expectancy": 31.0,
            "survival_probability": 1.0,
            "notes": "e0 typically ~21-37 across traditional HG groups; mid-range illustrative value",
        },
        {
            **base,
            "region_id": "FORAGER_HG_TRAD",
            "country_region": "Traditional hunter-gatherers (composite range mid)",
            "age": 15,
            "life_expectancy": 36.0,
            "survival_probability": "",
            "notes": "Illustrative remaining years at age 15; see paper for group-level tables",
        },
        {
            **base,
            "region_id": "FORAGER_HG_TRAD",
            "country_region": "Traditional hunter-gatherers (composite range mid)",
            "age": 45,
            "life_expectancy": 20.7,
            "survival_probability": 0.35,
            "notes": "e(45)~20.7; survival to 45 often ~0.26-0.43 (illustrative mid 0.35)",
        },
    ]


def write_gurven_raw_copy(rows: list[dict]) -> None:
    path = RAW / "gurven_kaplan" / "gurven_kaplan_seed.csv"
    _write_csv(path, LIT_COLUMNS, rows)


def main() -> None:
    ensure_dirs()
    fact_path = PROCESSED / "life_expectancy_long.csv"
    # Do not wipe an existing populated fact table
    if not fact_path.exists() or fact_path.stat().st_size < 100:
        _write_csv(fact_path, FACT_COLUMNS, [])
        print(f"  life_expectancy_long: header only (new)")
    else:
        print(f"  life_expectancy_long: left existing file intact")
    _write_csv(PROCESSED / "sources.csv", SOURCES_COLUMNS, seed_sources())
    _write_csv(PROCESSED / "countries_regions.csv", REGIONS_COLUMNS, seed_regions())
    _write_csv(PROCESSED / "methodology_notes.csv", NOTES_COLUMNS, seed_notes())
    lit = seed_literature_benchmarks()
    _write_csv(PROCESSED / "literature_benchmarks.csv", LIT_COLUMNS, lit)
    write_gurven_raw_copy(lit)
    print(f"Wrote templates and seeds under {PROCESSED}")
    print(f"  sources: {len(seed_sources())} rows")
    print(f"  regions: {len(seed_regions())} rows")
    print(f"  methodology_notes: {len(seed_notes())} rows")
    print(f"  literature_benchmarks: {len(lit)} rows")


if __name__ == "__main__":
    main()
