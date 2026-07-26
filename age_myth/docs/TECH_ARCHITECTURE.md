# Technical architecture (engineering view)

## Canonical ship path

```powershell
# from repo root
python -m src.analysis.run_final_agrade
# dev only if figures flaky:
python -m src.analysis.run_final_agrade --soft

pytest -q
```

**Single source of ship numbers:** `outputs/bulletproof/final_claims.json`  
**Hard invariants:** `outputs/bulletproof/claim_gate_results.json` (`all_pass`)  
**Human report:** `outputs/reports/FINAL_A_GRADE_REPORT.md`

## Data flow

```
data/raw/*
  → src/acquisition/* + src/cleaning/*
  → data/processed/life_expectancy_long.*
  → src/cleaning/build_modeling_view.py
  → data/processed/life_expectancy_modeling.*
  → src/analysis/panels.py  →  data/processed/analysis/hmd_summary_wide_*.parquet
  → src/analysis/bulletproof_suite.py  → claims + gates
  → src/analysis/final_agrade.py       → equal-country, strict band, scorecard
  → figures + FINAL report
```

## Module map

| Module | Role | Status |
|--------|------|--------|
| `populations.py` | De-dupe / allowlist | **canonical filter** |
| `panels.py` | Wide HMD panel | active |
| `ladder.py` | Multi-age expected ages | active |
| `claim_registry.py` | Gate thresholds | active |
| `bulletproof_suite.py` | Primary claims + exit codes | **canonical science gates** |
| `final_agrade.py` | Equal-country, strict band, scorecard | ship layer |
| `myth_tests.py` / `peer_hardening.py` | Historical iterations | prefer not to run for ship |
| `run_final_agrade.py` | Orchestrator | **only entry point to demo** |

## What gates protect

If someone breaks de-dupe, invents low age|65 when e0 is low, or breaks Sweden 1800 anchor, `bulletproof_suite` should fail (`SystemExit 2`).

Gates encode **our** scientific thresholds—they are not external truth. They prevent silent regression.

## Known engineering limits (own these)

1. Full suite needs built modeling data on disk (not pure unit-testable end-to-end without fixtures).  
2. Bootstrap is stochastic with fixed seeds per module (good enough; not cryptographic).  
3. Scorecard is an **internal checklist**, not journal peer review.  
4. Must run from repo root so `src.paths` resolve (or install package editable later).

## Tests

```powershell
pytest -q
```

Live panel tests skip if parquet panels are missing.
