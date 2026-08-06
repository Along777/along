# Aaron Long — Data Scientist

**Live site: [along777.github.io/along](https://along777.github.io/along/)** · [LinkedIn](https://www.linkedin.com/in/aaron-l-403644a7/) · [Résumé (PDF)](https://along777.github.io/along/aaron-long-resume.pdf) · Orlando, Florida

This repository is my portfolio. It holds the site itself plus the full source for every project on it: code, figures, and write-ups.

---

## About me

I started close to the markets, working as a financial analyst and day trader after earning my B.S. in Applied Statistics and B.A. in Economics from UC Davis. In 2019 I moved to Central Florida and into healthcare, working as an analyst for hospitals and clinics, where I learned how to turn messy operational data into decisions people actually trust.

Today I'm a **Data Science Advisor at Cigna**, building end-to-end ML pipelines, partnering with stakeholders, and putting AI agents into real workflows. I'm the AI lead for my team and run our internal training on AI-assisted development. Finance, then healthcare, now insurance: the industries keep changing, but the throughline stays the same, rigorous analytics that drive clear, practical decisions.

Since the first ChatGPT models in 2022 I've worked hands-on with generative AI as it evolved, and today I build across the whole landscape: Claude, Grok, and local open models on my own hardware. The projects below are where I put that to the test.

---

## AI-powered projects (2026)

Each of these is an end-to-end study built with AI as the engineering partner, published as a hand-written article rather than a notebook dump. Every one uses public data and states its own limits.

| Project | What it asks | Built with |
|---|---|---|
| **[Return to Fire](https://along777.github.io/along/wildfire-return/)** | The 2017 Tubbs Fire took my family's home, and fires became the first subject of my career in 2020. Six years later I rebuilt that work against 2.3M fires and modern methods. Five rounds, four exhibits, including a published claim I had to retract and rebuild. | Claude Fable 5 + Opus 5 |
| **[A Tale of Two AI Minds: Heat, Crime, and Chicago](https://along777.github.io/along/heat-and-crime/)** | Does heat drive crime? 2.76M Chicago reports joined to weather: a day 10°F hotter than others in the same month shows about 5.6% more reported violent crime. | Claude Fable 5 + Grok 4.5 |
| **[Do UFOs Follow the News Cycle?](https://along777.github.io/along/ufo-news-cycle/)** | 618,316 sighting reports, two databases, 384 SpaceX launches, one adversarial audit. Fireworks and Starlink move reports; congressional hearings do not. | Claude |
| **[Does Growth Have to Cost the Planet?](https://along777.github.io/along/world-panel-ekc/)** | The environmental Kuznets curve tested across 217 economies and 30 years. | Claude |
| **[Life expectancy was 35. Almost nobody died at 35.](https://along777.github.io/along/age_myth/)** | The classic age myth, tested. When life expectancy at birth was under 40, people who reached 65 still expected about 75. Infant mortality crushed the average. | Grok 4.5 |
| **[Diet soda did not give you cancer](https://along777.github.io/along/diet-soda-analysis/)** | NHANES on the diet soda myths. Who drinks it writes most of the scare: the cancer signal is not significant, and the blood-sugar gap is mostly selection. | Grok 4.5 |
| **[Local RAG over a 148-Book Library](https://along777.github.io/along/rag_books/)** | 148 technical PDFs, about 48,800 pages, turned into a citation-grounded chatbot running entirely on a laptop. No cloud, no API cost. | Python, LangGraph, Ollama |

### How I work with AI

These projects are also an experiment in **which model for which job**. Claude Fable 5 is strong at greenfield velocity: a super-prompt to a working notebook in one sitting. Grok 4.5 is strong at the second pass, the audit and the attack. Heat and Crime is a deliberate handoff between the two, and it documents why, including the practical constraint that Claude Max rate-limits on a five-hour cycle.

The honesty tooling matters more than the model choice. *Return to Fire* ships a `verify_claims.py` that machine-checks **130 numeric claims** across its pages, enforces a byte-frozen copy of the original first-shot output, and fails the build if a retired claim ever reappears. One published calibration claim did not survive that gate, so it was retracted, rebuilt, and re-verified in public.

---

## Working papers

AI-assisted research beyond code. Four working papers in four fields — physics, economics/politics, economics advocacy, and art history — each written with AI, built by one Python pipeline (every paper is a program that prints its own PDF), and put through a commissioned hostile review (attack → defense → verdict) before revision. Drafts, not peer-reviewed publications, and labeled as such.

**Browse:** [along777.github.io/ai-working-papers](https://along777.github.io/ai-working-papers/) · [source repo](https://github.com/Along777/ai-working-papers)

---

## Foundational projects (2019–2021)

The classical work that built the fundamentals: Python, R, EDA, and traditional ML. These stay online unedited. *Return to Fire* revisits the wildfire trilogy honestly rather than quietly replacing it.

| Project | Stack |
|---|---|
| [Wildfire Prediction with Random Forest](https://nbviewer.jupyter.org/github/Along777/along/blob/master/actualProjects/randomforest.ipynb) | Python, scikit-learn, SQLite |
| [Wildfire Analysis I](https://along777.github.io/along/projects/wildfiresp1.html) and [II](https://along777.github.io/along/projects/wildfiresp2.html) | R, ggplot2, leaflet |
| [Bitcoin Price Regression](https://nbviewer.jupyter.org/github/Along777/along/blob/master/actualProjects/BitcoinRegression.ipynb) | Python, statsmodels |
| [PDF Parsing with Regex → SQL Server](https://nbviewer.jupyter.org/github/Along777/along/blob/master/actualProjects/Regex.ipynb) | Python, pdfplumber, pyodbc |
| [YouTube Trending Videos EDA](https://nbviewer.jupyter.org/github/Along777/along/blob/master/actualProjects/Youtube.ipynb) | Python, pandas, seaborn |
| [Python for Pivot Tables](https://nbviewer.jupyter.org/github/Along777/along/blob/master/actualProjects/PivotTables.ipynb) | Python, pandas |
| [Flex Dashboard, US Accidents](https://along777.github.io/along/projects/flexdashboard.html) | R, flexdashboard, highcharter |
| [Stock Analysis with Highcharter](https://along777.github.io/along/projects/highstocks.html) | R, highcharter, quantmod |

---

## Repository layout

```
index.html              the portfolio site (single file, Tailwind via CDN)
aaron-long-resume.pdf    résumé
wildfire-return/         Return to Fire: 4 pages, verifier, full pipeline
heat-and-crime/          Heat and Crime: article + code + cached data
age_myth/                Age myth: article, src/, tests
diet-soda-analysis/      Diet soda: article, src/, docs
ufo-news-cycle/          UFO article
world-panel-ekc/         EKC project + notebooks
rag_books/               Local RAG chatbot (code only, bring your own PDFs)
legacy/                  the original Bootstrap portfolio, preserved
projects/, actualProjects/, images/   assets for the older work
```

### Running a project

Each project folder has its own README with setup steps. The pattern is the same:

```bash
cd <project>
pip install -r requirements.txt
python <the run script named in that README>
```

**Bulk data is intentionally not committed.** Raw downloads and built datasets are gitignored and rebuilt by each project's pipeline, so the repo stays code, figures, and write-ups. Where a cache is small enough to ship, it is hash-locked so a rebuild can be proven identical.

---

## Contact

**[along1929@gmail.com](mailto:along1929@gmail.com)** · [LinkedIn](https://www.linkedin.com/in/aaron-l-403644a7/) · [github.com/Along777](https://github.com/Along777)
