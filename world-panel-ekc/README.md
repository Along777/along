# Does Growth Have to Cost the Planet?

**A world-panel econometrics project — 217 economies × 30 years — built
end-to-end by AI (Claude), directed only through prompts.**

> **The experiment:** can a modern AI carry a real, rigorous data-science
> project by itself? Every artifact in this repo — the data pipeline, the
> models, the peer review, the stakeholder defense, and the write-up — was
> written by Claude. The human contribution was *direction* (plain-English
> prompts), not code: **zero lines were hand-written**, and the AI caught
> three of its own bugs along the way. The climate finding below is the
> analysis; the project itself is the proof of concept.

📊 **[Read the interactive write-up](https://along777.github.io/along/world-panel-ekc/)** —
built for both technical and non-technical readers, with plain-English
explanations alongside every modeling decision.

## The finding in three bullets

- The famous **Environmental Kuznets Curve** ("grow now, clean up later") is
  **not in the data** — its inverted-U flips sign under within-country fixed
  effects and never yields an identifiable turning point (2% of 500 bootstrap
  resamples).
- **Growth still buys carbon**: every 1% of GDP growth brings ~0.5–0.7% more
  CO2 — a *tighter* coupling than in the 1990s.
- What changed is the **autonomous drift**: at zero growth, emissions now fall
  ~1%/yr globally and ~2.7%/yr in rich countries (vs. *rising* 0.8%/yr in the
  late '90s). **Two-speed decarbonization**: technology and the energy mix are
  doing the cleanup — not income.

## What's here

| Path | What it is |
|---|---|
| `index.html` | The interactive project write-up (self-contained) |
| `FINAL_SUMMARY.md` | Full technical narrative, end to end |
| `notebooks/build_panel.py` | Stage 1 — fetch, clean, audit, feature-engineer the panel |
| `notebooks/eda.py` | Stage 2 — exploratory analysis |
| `notebooks/topic_models.py` | Stage 3 — four candidate topics as panel FE regressions |
| `notebooks/run_ekc_pipeline.py` + `notebooks/ekc_pipeline/` | Stage 4 — the EKC pipeline (spec ladder, diagnostics, bootstrap, ML benchmark) |
| `notebooks/stakeholder_stress_tests.py` | Stage 5 — red team: stress tests + the star result |
| `notebooks/output/` | Reports, figures, and result tables from every stage |

Key reports: [`EKC_REPORT.md`](notebooks/output/ekc/EKC_REPORT.md) (the model)
and [`STAKEHOLDER_BRIEF.md`](notebooks/output/ekc/STAKEHOLDER_BRIEF.md)
(the 8-objection defense).

## Run it yourself

```bash
pip install -r requirements.txt
cd notebooks
python build_panel.py                  # rebuilds the raw panel from the World Bank API
python eda.py
python topic_models.py
python run_ekc_pipeline.py
python stakeholder_stress_tests.py
```

The raw panel (~17MB) and API cache are not committed — `build_panel.py`
regenerates them from the World Bank's public API (responses are cached
locally, so re-runs don't re-hit the servers).

## How it was built (the point of the project)

Built entirely by **Claude**, directed only through plain-English prompts — no
hand-written code. The method that made it trustworthy was steering the AI
through a **rotating-role workflow**:

**builder → data auditor → reviewer-on-another-team → red team**

Each role was told to attack the previous one's work, and each surfaced
problems the last was blind to — including a naive "carry-forward" baseline
that beat both models on the easy prediction task (forcing an honest reframe),
and the COVID sensitivity that became the headline result. The AI also caught
three of its own bugs: a fixed-effects/trend collinearity, a catastrophic
intercept-alignment error (R² of −31), and a silently-blank set of EDA charts.

All findings are observational associations, not causal effects; limitations
are stated in every report. (The exact prompt playbook is kept private — the
*method* above is the shareable part.)
