# Data Acquisition Plan

Priority order for Phase 1. Prefer reproducible scripts over manual clicks; log every download.

## Priority 1 — Our World in Data (open)

| Series | URL pattern | Age | Role |
|--------|-------------|-----|------|
| Long-run life expectancy | `https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=false` | 0 | Global / long-run e₀ (stitched) |
| Life expectancy at age 15 | `https://ourworldindata.org/grapher/life-expectancy-at-age-15.csv?v=1&csvType=full&useColumnShortNames=false` | 15 | Adult conditional LE |
| HMD–UN WPP e₀ | `https://ourworldindata.org/grapher/life-expectancy-hmd-unwpp.csv?v=1&csvType=full&useColumnShortNames=false` | 0 | Cleaner modern VR-based e₀ |

**Auth:** none. **User-Agent:** set a descriptive UA (OWID requests this).

**Script:** `python -m src.acquisition.download_owid`  
**Raw dir:** `data/raw/owid/`  
**Log:** `data/raw/owid/download_log.json`

**Caveat:** Long-run series combines HMD, Zijdeman/Clio-Infra, Riley regional, UN WPP. Flag `owid_stitched`.

---

## Priority 2 — Human Life-Table Database (open, no login)

| Item | Detail |
|------|--------|
| Site | https://www.lifetable.de/ |
| Bulk | https://www.lifetable.de/File/GetDocument/data/hld.zip (path may change; script probes) |
| Coverage | ~15,000 tables, ~142 countries/areas |
| Auth | **None** |

**Script:** `python -m src.acquisition.download_hld`  
**Raw dir:** `data/raw/hld/`  
**Parser:** `python -m src.cleaning.hld_to_long`

**Caveat:** Methods vary by original table. Flag `hld_published_table`. Prefer HMD when both exist for the same country-year for analytical “gold” series.

**Bulk format (observed 2026):** `hld.zip` contains a single member named `res` (~214 MB uncompressed CSV) with columns:

`Country, Region, Residence, Ethnicity, SocDem, Version, Ref-ID, Year1, Year2, TypeLT, Sex, Age, AgeInt, m(x), q(x), l(x), d(x), L(x), T(x), e(x), e(x)Orig`

- `Sex`: 1 = male, 2 = female  
- Parser keeps ages `{0,1,5,15,20,30,50,65}` and years 1500–2100

---

## Priority 2b — HMD Public Summary Indicators (no login) **NEW**

| File | URL |
|------|-----|
| e0 / e65 / e80 | `https://www.mortality.org/File/GetDocument/Public/HMD_summary/hmd_summary_ex_0_65_80.xlsx` |
| Infant mortality | `.../hmd_summary_IMR.xlsx` |
| Survival birth→65 | `.../hmd_summary_px_0_to_65.xlsx` |

**Script:** `python -m src.acquisition.download_hmd_summary`  
**Clean:** `python -m src.cleaning.hmd_summary_to_long`  
**source_id:** `hmd_summary_public` · **flag:** `hmd_complete`

Wide Excel by sex sheet; melt to long. IMR joined onto e0; survival onto age-65 rows.

## Priority 2c — OWID infant & child mortality **NEW**

| Series | CSV |
|--------|-----|
| Infant mortality | `https://ourworldindata.org/grapher/infant-mortality.csv?v=1&csvType=full&useColumnShortNames=false` |
| Child mortality | `https://ourworldindata.org/grapher/child-mortality.csv?v=1&csvType=full&useColumnShortNames=false` |

Included in `download_owid.py`. Join: `python -m src.cleaning.owid_mortality_to_long`  
Also writes `data/interim/owid_mortality_long.csv`.

## Priority 2d — Eurostat demo_mlexpec **NEW**

Life expectancy by age and sex (EU+). SDMX-CSV bulk.

**Script:** `python -m src.acquisition.download_eurostat`  
**Clean:** `python -m src.cleaning.eurostat_to_long`  
**source_id:** `eurostat_demo_mlexpec`

## Priority 3 — Human Mortality Database full tables (registration)

| Item | Detail |
|------|--------|
| Site | https://www.mortality.org/ |
| Access | Free registration; accept user agreement |
| Priority countries | SWE (1751+), GBRTENW (England & Wales), FRATNP (France), JPN |
| Files | Period life tables `bltper_1x1.txt`, `fltper_1x1.txt`, `mltper_1x1.txt` |
| Columns used | `Year`, `Age`, `mx`, `qx`, `ax`, `lx`, `dx`, `Lx`, `Tx`, `ex` |

**Credentials (env only):**

```powershell
$env:HMD_USER = "you@example.com"
$env:HMD_PASSWORD = "your-password"
```

**Script:** `python -m src.acquisition.download_hmd`  
**Without credentials:** `python -m src.cleaning.hmd_life_table --fixture` uses `data/raw/fixtures/hmd_sweden_sample.txt`.

**Zipped alternatives:** https://www.mortality.org/Data/ZippedDataFiles (by country / by statistic).

---

## Priority 4 — Clio-Infra / Zijdeman et al.

| Item | Detail |
|------|--------|
| Dataset | Life Expectancy at Birth (Total), Zijdeman & Ribeira da Silva (2015) |
| Handle | http://hdl.handle.net/10622/LKYT53 |
| Also | clio-infra.eu indicators; IISH Dataverse |

**Script:** `python -m src.acquisition.download_clio` (best-effort URL probe)  
**If download fails:** document path for manual drop into `data/raw/clio_infra/`.

---

## Priority 5 — Literature seed (Gurven & Kaplan 2007)

| Item | Detail |
|------|--------|
| Paper | Gurven, M. & Kaplan, H. (2007). Longevity among hunter-gatherers. *Population and Development Review* |
| Content | e₀, survival to adulthood, e(45) for forager / horticultural groups |
| Script | Curated CSV under `data/raw/gurven_kaplan/` + load into `literature_benchmarks` |

**Not** national vital registration. `population_type = forager_horticultural`.

---

## Priority 6 — Deferred (documented only in Phase 1)

| Source | Notes |
|--------|-------|
| Cambridge Group / Wrigley & Schofield | English historical demography; extract published tables in Phase 2 |
| LAMBdA | LatAm adjusted life tables; free registration |
| Cummins European nobility | Adult ages at death 800–1800; elite series |
| CBDB Chinese notables | Elite longevity comparisons |
| Coale–Demeny / UN MLT | Model tables only with `population_type = model_table` |

---

## Download hygiene

1. Never commit credentials or multi-GB zips if gitignored.  
2. Always write `download_log.json` with URL, timestamp, bytes, sha256 when feasible.  
3. Keep raw files immutable; transform only into `interim/` and `processed/`.  
4. Record `retrieved_at` on every processed row.
