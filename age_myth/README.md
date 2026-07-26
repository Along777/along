# age_myth

**Life expectancy was 35. Almost nobody died at 35.**

When life expectancy at birth was under 40, people already 65 still expected about **75**. Only about **27%** of births got there. Median infant mortality was about **196 per 1,000**. Adults also gained about **+8.5 years** of remaining life at 65 from pre-1900 to post-2000.

**Start here:** open `index.html` from this folder (charts load from `outputs/figures/`).

## What this repo is

Open-data pipeline and mythbust on historical life expectancy.

| Myth | Claim | Result |
|------|-------|--------|
| A | "LE was 30-35, so adults died at 30-35" | False |
| B | "Only infants improved; adults always lived modern lengths" | False |

Primary: **Human Mortality Database** public summaries (birth LE, LE at 65, infant mortality, survival to 65).  
Ladder: **Human Life-Table Database**. Context: OWID, Clio, Eurostat.

## Quick start

```powershell
cd age_myth
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pull + clean (needs network; HLD zip is large)
python scripts/run_phase1.py

# Full analysis pack + figures + claim gates
python -m src.analysis.run_final_agrade
python -m pytest -q
```

## Key outputs

| Path | What |
|------|------|
| `index.html` | Public article (start here) |
| `outputs/reports/FINAL_A_GRADE_REPORT.md` | Full ship report |
| `outputs/reports/EDA_AND_MODEL_SUMMARY.md` | EDA + models |
| `outputs/bulletproof/final_claims.json` | Locked claim numbers |
| `outputs/figures/` | Chart pack |

## Layout

```text
age_myth/
  index.html          public article
  src/                acquisition, cleaning, analysis
  data/raw|interim|processed
  outputs/figures|reports|bulletproof|tables
  docs/               methodology + handoff notes
  tests/
```

Large raw downloads (HLD zip, full HMD login extracts) are gitignored. Re-run the acquisition scripts after clone.

## Scope (say this when sharing)

- High-quality vital-registration populations in open life tables, not every past society
- Period rates for a year, not one person's full lived life
- Always pair "age if already 65" with infant death or survival to 65

## License / data

Code is yours to use in this project. Source data remain under their own terms (HMD, HLD, OWID, Eurostat, Clio). Respect provider attribution and access rules.
