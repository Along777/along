# Modeling view summary

- Built with `year <= 2023`, excluded sources: `['gurven_kaplan_2007', 'hmd_fixture']`.
- Main modeling rows: **167,443**
- Sources: ['clio_zijdeman_2015', 'eurostat_demo_mlexpec', 'hmd_summary_public', 'owid_le_age15', 'owid_le_hmd_unwpp', 'owid_le_longrun']
- Ages: [0, 1, 5, 15, 20, 30, 50, 65, 80]

## Rows by source

| source_id | rows |
|-----------|------|
| eurostat_demo_mlexpec | 46,713 |
| hmd_summary_public | 44,694 |
| owid_le_longrun | 21,565 |
| owid_le_age15 | 20,804 |
| owid_le_hmd_unwpp | 20,804 |
| clio_zijdeman_2015 | 12,863 |

- Optional HLD median-collapsed rows: **57,665** (max n_tables=114)

## Safe defaults

- Prefer `source_id == 'hmd_summary_public'` for long-run e0/e65/e80 + IMR.
- Prefer `source_id == 'eurostat_demo_mlexpec'` for EU multi-age (ISO2 geos only).
- Use `hld_median` only as sensitivity; inspect `n_tables`.
