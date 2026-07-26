# Analysis notebooks

Prefer the scripted pipeline (reproducible):

```powershell
cd ..
python -m src.analysis.run_analysis
```

Outputs:

- `outputs/reports/age_myth_findings.md`
- `outputs/figures/*.png`
- `outputs/tables/*.csv` / `myth_tests_summary.json`
- `data/processed/analysis/hmd_summary_wide_both.parquet`

Interactive follow-up: load the wide panel and summary JSON in a notebook of your choice.
