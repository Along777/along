# Modeling Handoff

Short contract for analysts after DE review #2.

## Start here (preferred)

```python
import pandas as pd

# Pre-filtered analysis table (recommended)
df = pd.read_parquet("data/processed/life_expectancy_modeling.parquet")

hmd = df[df["source_id"] == "hmd_summary_public"]   # e0/e65/e80 + IMR + S→65
eu  = df[df["source_id"] == "eurostat_demo_mlexpec"]  # multi-age EU, ISO2 only
```

Build/refresh:

```powershell
python -m src.cleaning.build_modeling_view
python -m src.validation.checks
```

### What the modeling view already does

| Filter | Applied? |
|--------|----------|
| Drop `hmd_fixture`, `gurven_kaplan_2007` | Yes |
| Drop year > 2023 (projections) | Yes |
| Drop raw multi-table **HLD** | Yes (default) |
| Eurostat: keep ISO2 geos only | Yes |
| Adds `modeling_tier`, `is_projection` | Yes |

Optional HLD sensitivity (median-collapsed, with `n_tables`):

- `data/processed/life_expectancy_modeling_hld_median.parquet`

## Full fact table (warehouse)

- `data/processed/life_expectancy_long.parquet` — **all** sources including HLD fan-out, fixture, projections  
- Companion: `data/processed/sources.csv`  
- Mortality helper: `data/interim/owid_mortality_long.csv` (IMR + U5MR)  
- **Do not** naive-average the full fact table.

## Sources

| source_id | In modeling view? | Notes |
|-----------|-------------------|--------|
| `hmd_summary_public` | **Yes — gold** | e0, e65, e80; IMR; S(0→65); unique grain; no login |
| `eurostat_demo_mlexpec` | **Yes — gold EU** | Ages 0–80; modern; ISO2 only in modeling view |
| `owid_le_*` | Yes | Stitched; e0/e15 split across source_ids |
| `clio_zijdeman_2015` | Yes | Deep historical e0 |
| raw `hld` | No (default) | Use `hld_median` sensitivity file |
| `hmd_fixture` | No | Fake |
| `gurven_kaplan_2007` | No | Literature only |

## Dual-myth analyses

| Claim | Recommended series |
|-------|-------------------|
| Myth A: e0 ≠ adult death age | HMD summary e0 vs e65 + IMR; or Eurostat e0 vs e15 |
| Myth B: adult LE also rose | HMD summary e65/e80 over time; Eurostat e15/e50/e65 |
| Infant drag | `infant_mortality_rate` on HMD summary e0 (complete) |

## Region codes

| Family | Example Sweden |
|--------|----------------|
| HMD summary / HLD | `SWE` |
| Eurostat | `SE` |
| Clio | `SWEDEN` |

Do not join `countries_regions.csv` as a complete dimension (seed only).

## Known open issues

1. HLD raw grain still multi-table until Ref-ID retained in loader.  
2. Full HMD 1x1 (continuous e15/e20 from 1751) needs registration.  
3. Region concordance table not built.  

## Full refresh

```powershell
python scripts/run_phase1.py
python -m src.cleaning.build_modeling_view
python -m src.validation.checks
```
