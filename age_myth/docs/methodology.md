# Methodology and Research Framing

## The research problem

A common claim is that “people in the past only lived to 30–35 years.” That figure usually refers to **period life expectancy at birth** (\(e_0\)) in high-mortality historical populations. Interpreting it as “adults typically died at 30–35” is a category error.

A common overcorrection claims that “if you survived childhood, you lived into your 70s just like today.” That is also too strong: **adult** remaining life expectancy rose substantially over the last two centuries as well.

This project builds a database that lets both claims be tested with the right measures.

## Dual-myth framing

### Myth A — “People only lived to ~30”

| | |
|--|--|
| **Usually based on** | \(e_0\) or crude mean age at death |
| **Why misleading** | High infant and child mortality pull the average down even when many adults reach 50–70 |
| **Better evidence** | \(e(15)\), \(e(20)\), \(e(30)\); survival \(l(15)/l(0)\); age-at-death distributions conditional on adulthood |

### Myth B — “Survivors lived modern adult lifespans”

| | |
|--|--|
| **Usually based on** | Anecdotes; forager modal adult ages; stripping only infant mortality once |
| **Why misleading** | Historical adult mortality was still high from infection, violence, maternal mortality, etc. Period \(e(x)\) for adults has risen a great deal (e.g. OWID France multi-age charts) |
| **Better evidence** | Long series of \(e(15)\), \(e(50)\), \(e(65)\) for the **same** populations over time (HMD/HLD) |

**Design rule:** Prefer tables that supply **multiple ages** for the same place-year, not a single “adjusted life expectancy.”

## What “life expectancy at age \(x\)” means

For a **period** life table in year \(t\):

1. Observe age-specific death rates in year \(t\).
2. Convert rates to survival probabilities.
3. Compute the expected remaining lifetime of a hypothetical person aged \(x\) who faces those rates for the rest of life.

It does **not** mean the average age at death of people who died in year \(t\), nor the average age of people alive in year \(t\).

## Infant and child mortality

In pre-modern and early modern settings, deaths under age 5 often account for a large share of the gap between historical and modern \(e_0\). That does **not** imply adult mortality was modern. The database stores:

- \(e_0\) (sensitive to early mortality)
- \(e(x)\) for \(x \ge 15\) (conditional on survival to \(x\))
- Survival ratios when life tables provide \(l(x)\)
- Infant mortality rates when sources publish them

## Source quality hierarchy (research use)

1. **HMD** — uniform methods, high-quality vital registration; best for deep European/high-income series (Sweden from 1751, etc.).
2. **HLD** — large collection of published life tables worldwide; methods **heterogeneous**; excellent coverage expansion with explicit quality flags.
3. **OWID compilations** — convenient global e₀ and selected age charts; **stitched** across providers—always keep `source_id` and `owid_stitched` flag.
4. **Clio-Infra / Zijdeman** — historical e₀ estimates feeding many long-run charts.
5. **Cambridge Group / Wrigley et al.** — English parish reconstitution / national reconstructions (document carefully).
6. **Anthropological forager tables (Gurven & Kaplan)** — not national VR; answer evolutionary/lifestyle questions; `population_type = forager_horticultural`.
7. **Elite genealogies (Cummins, CBDB)** — adult ages at death for elites; not period e₀ for the general population.
8. **Model life tables** — synthetic; never label as observed national data.
9. **Paleodemography / cemetery means** — generally **excluded** from the main fact table (selection and age-estimation biases).

## Assumptions (Phase 1)

1. Period measures are the default unless a cohort table is explicitly loaded.
2. Sex `both` is stored only when the source provides it; we do not invent mid-sex averages without documentation.
3. Rows from different sources for the same country-year are **not** averaged in the loader; analysts choose or model concordance later.
4. HLD tables may use abridged ages; extraction targets exact ages 0, 15, 20, 30, 50, 65 when present, else nearest documented age with a note.
5. “England” vs “England & Wales” vs “UK” are distinct `region_id`s when sources differ.

## Known limitations

- Pre-1800 series are often reconstructions with wider uncertainty.
- Global e₀ before ~1950 is sparse and uneven.
- Viral forager “lived to 78” claims often cite **modal adult lifespan**, not \(e_0\).
- Registration completeness and age misreporting bias some national series (esp. older ages, some regions).

## Citation practice

When publishing results from this database, cite **both** this project’s processing version and the **original** data providers (HMD, HLD, OWID, etc.) per their terms.
