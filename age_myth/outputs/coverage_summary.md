# Coverage summary

- **Rows:** 578,032
- **Sources:** 9 — ['clio_zijdeman_2015', 'eurostat_demo_mlexpec', 'gurven_kaplan_2007', 'hld', 'hmd_fixture', 'hmd_summary_public', 'owid_le_age15', 'owid_le_hmd_unwpp', 'owid_le_longrun']
- **Regions:** 525
- **Ages:** [0, 1, 5, 15, 20, 30, 45, 50, 65, 80]
- **Year range:** 1543–2100

## Rows by source_id

| source_id | rows | ages | year min | year max |
|-----------|------|------|----------|----------|
| clio_zijdeman_2015 | 12,863 | [0] | 1543 | 2012 |
| eurostat_demo_mlexpec | 53,652 | [0, 1, 5, 15, 20, 30, 50, 65, 80] | 1960 | 2024 |
| gurven_kaplan_2007 | 3 | [0, 15, 45] |  |  |
| hld | 383,961 | [0, 1, 5, 15, 20, 30, 50, 65] | 1751 | 2024 |
| hmd_fixture | 16 | [0, 1, 5, 15, 20, 30, 50, 65] | 1800 | 1900 |
| hmd_summary_public | 44,883 | [0, 65, 80] | 1751 | 2025 |
| owid_le_age15 | 40,285 | [15] | 1751 | 2100 |
| owid_le_hmd_unwpp | 20,804 | [0] | 1751 | 2023 |
| owid_le_longrun | 21,565 | [0] | 1543 | 2023 |

## Dual-myth readiness

- e0 rows: **124,232**
- e15 rows: **94,265**
- e65 rows: **68,926**
- IMR non-null rows: **71,361**
- Quality flags present: ['clio_historical', 'fixture_demo', 'hld_published_table', 'hmd_complete', 'literature_extract', 'official_stats', 'owid_stitched']

## Modeling recommendation

- Prefer `data/processed/life_expectancy_modeling.parquet` (excludes fixture, projections, raw HLD fan-out).
- Gold long-run: `source_id == 'hmd_summary_public'` (e0/e65/e80 + IMR + S→65).
- EU multi-age: `source_id == 'eurostat_demo_mlexpec'` (ISO2 geos).

## Validation errors

- None

## Validation warnings

- **WARN:** hmd_fixture demo rows present in fact table: 16 — EXCLUDE from modeling
- **WARN:** Rows with year > 2023 (projections / provisional): 22,745 — filter for history
- **WARN:** HLD multi-table grain: 43,902 keys with >1 table (mean=6.6, max=114) — do not naive-average; use life_expectancy_modeling (excludes HLD) or hld_median view
