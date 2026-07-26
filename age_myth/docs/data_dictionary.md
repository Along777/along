# Data Dictionary

Analysis-ready schema for the historical life expectancy database. All measures are stored in **long** (tidy) form unless noted.

## Core concepts

### Life expectancy at exact age \(x\), \(e(x)\)

Average remaining years of life for a person who has already reached exact age \(x\), under a given schedule of age-specific mortality rates.

- \(e(0)\) or \(e_0\): life expectancy **at birth**
- \(e(15)\): remaining years expected at age 15 (conditional on surviving to 15)

**What it is not:** the mean age of skeletons in a cemetery, the average age of famous people, or “when most people died.”

### Period vs cohort

| Type | Meaning |
|------|---------|
| **Period** | Synthetic cohort: applies one calendar year’s (or period’s) age-specific rates to a hypothetical life course |
| **Cohort** | Follows people born in the same year(s) through their actual lifetimes |

Phase 1 loads are almost entirely **period** measures.

### Survival probability

When derived from a complete life table with radix \(l(0)\):

\[
S(x) = \frac{l(x)}{l(0)}
\]

Interpretation: fraction of a birth cohort expected to survive to exact age \(x\) under the table’s rates.

---

## Table: `life_expectancy_long`

**Grain:** one row per  
`(region_id, year|period, sex, age, measure_type, population_type, source_id)`.

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `region_id` | string | yes | Stable ID (ISO3 when possible, else slug) |
| `country_region` | string | yes | Display name |
| `year` | int | conditional | Calendar year if single-year table |
| `period_start` | int | conditional | Inclusive start of multi-year period |
| `period_end` | int | conditional | Inclusive end of multi-year period |
| `sex` | enum | yes | `male`, `female`, `both` |
| `age` | int | yes | Exact age \(x\) for \(e(x)\) (0, 15, 20, …) |
| `life_expectancy` | float | yes | Remaining years at age \(x\) |
| `survival_probability` | float | no | \(l(x)/l(0)\) when available |
| `infant_mortality_rate` | float | no | Deaths under age 1 per 1,000 live births |
| `measure_type` | enum | yes | `period` or `cohort` |
| `population_type` | enum | yes | See below |
| `table_type` | string | no | e.g. `hmd_1x1`, `hld_abridged`, `owid_compiled` |
| `data_quality_flag` | string | yes | See flags |
| `source_id` | string | yes | FK → `sources.source_id` |
| `notes` | string | no | Row-level caveats |
| `retrieved_at` | date | yes | ISO date of download/processing |

At least one of `year` or (`period_start`, `period_end`) must be non-null.

### `population_type`

| Value | Use |
|-------|-----|
| `national` | National (or HMD “country”) vital-registration / reconstructed national series |
| `subnational` | Province, city, parish set |
| `elite` | Nobility, scholars, genealogical elites (ages at death / adult lifespan) |
| `forager_horticultural` | Anthropological forager / horticultural populations |
| `model_table` | Coale–Demeny, UN model tables (synthetic, not observed) |

### `data_quality_flag` (controlled vocabulary)

| Flag | Meaning |
|------|---------|
| `hmd_complete` | HMD uniform methods, high-quality VR |
| `hld_published_table` | HLD collected published table; methods vary by original author |
| `owid_stitched` | OWID compilation across HMD / Zijdeman / Riley / UN WPP |
| `clio_historical` | Clio-Infra / Zijdeman historical e₀ compilation |
| `literature_extract` | Hand-curated from a published paper/table |
| `reconstructed` | Family reconstitution / model-based reconstruction |
| `sparse_estimate` | Few observations or wide uncertainty |
| `fixture_demo` | Synthetic/demo HMD-format fixture for pipeline tests |
| `official_stats` | National/international statistical office series (e.g. Eurostat) |

---

## Table: `sources`

| Column | Type | Description |
|--------|------|-------------|
| `source_id` | string (PK) | e.g. `owid_le_longrun`, `hmd_v6`, `hld` |
| `name` | string | Short name |
| `citation` | string | Bibliographic citation |
| `url` | string | Canonical URL |
| `license` | string | License / reuse note |
| `access_notes` | string | Registration, bulk zip, etc. |
| `version_or_retrieved` | string | Version or retrieval date |

---

## Table: `countries_regions`

| Column | Type | Description |
|--------|------|-------------|
| `region_id` | string (PK) | |
| `name` | string | |
| `iso3` | string | ISO 3166-1 alpha-3 when applicable |
| `hmd_code` | string | HMD country code (e.g. `SWE`) |
| `hld_name` | string | Name as used in HLD files |
| `region_type` | enum | `country`, `subnational`, `aggregate`, `population_group` |
| `continent` | string | |
| `coverage_notes` | string | |

---

## Table: `methodology_notes`

| Column | Type | Description |
|--------|------|-------------|
| `note_id` | string (PK) | |
| `topic` | string | e.g. `dual_myth`, `infant_mortality`, `hld_methods` |
| `applies_to_source_id` | string | optional |
| `applies_to_region_id` | string | optional |
| `text` | string | Note body |
| `severity` | enum | `info`, `caveat`, `limitation` |

---

## Table: `literature_benchmarks` (optional seed)

Curated non-VR benchmarks (forager groups, early life tables) with the same conceptual columns as the fact table plus `paper_citation`. Always `data_quality_flag = literature_extract`.

---

## Units and valid ranges (validation)

| Field | Unit | Soft range |
|-------|------|------------|
| `life_expectancy` | years | (0, 120) |
| `survival_probability` | probability | [0, 1] |
| `infant_mortality_rate` | per 1,000 births | [0, 1000) |
| `age` | years | [0, 110] |

---

## Join keys

```
life_expectancy_long.source_id  → sources.source_id
life_expectancy_long.region_id  → countries_regions.region_id
```

Do **not** average or stack rows from different `source_id` values without an explicit concordance step.
