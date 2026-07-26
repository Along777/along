# Community Source Map (Reddit + X triage)

Notes from public discussion of the “lived to 30” claim. This is **not** peer review; it guides which sources people cite and where discourse goes wrong.

## Venues surveyed

- Reddit: r/AskHistorians, r/badhistory, r/dataisbeautiful, r/todayilearned, r/history, r/AskAnthropology  
- X: demography/data accounts, OWID, viral wellness/history posts  

## What high-quality threads emphasize

| Point | Typical sources named |
|-------|----------------------|
| e₀ ≠ adult death age | HMD, OWID explainers, AskHistorians answers |
| Use e(15)/e(20) | HMD life tables; OWID age-15 charts |
| England deep history | Wrigley & Schofield; Cambridge Group reconstitutions |
| Sweden long VR series | HMD Sweden from 1751 |
| Adult LE also rose | OWID multi-age France charts; r/badhistory “not 60–70 forever” |
| Foragers | Gurven & Kaplan (2007)—often misquoted online |

## Source reliability triage (community-informed)

| Source / claim type | Reliability for this DB | Notes |
|---------------------|-------------------------|-------|
| HMD full life tables | ★★★★★ | Gold standard; login |
| HLD published tables | ★★★★ | Broad coverage; methods vary |
| OWID stitched e₀ | ★★★★ (for overview) | Document stitching |
| Wrigley et al. England | ★★★★ | Scholarly reconstruction |
| Gurven & Kaplan tables | ★★★ (for foragers) | Not national; careful use |
| Elite ages at death | ★★★ (for elites only) | Selection bias |
| “Famous people lived to 80” lists | ★ | Anecdote |
| Cemetery mean ages as e₀ | ★ | Paleodemography ≠ period LE |
| “Survive childhood → 75 like today” | ★★ (claim) | Often false as absolute |

## Dual-myth as community pattern

1. **Viral oversimplification (Myth A):** e₀ ≈ 30 → “everyone died young.”  
2. **Reactive oversimplification (Myth B):** infant mortality only → “adults always lived modern lengths.”  
3. **Careful middle (target of this DB):** high child mortality **and** elevated adult mortality historically; measure both.

## X-specific notes

- OWID posts multi-age period LE (e.g. France ages 10 and 65 over time) specifically against the “only child mortality improved” story.  
- Wellness accounts often cite hunter-gatherer modal adult lifespan (~68–78) as if it were historical e₀.  
- Early life tables (Halley 1693 Breslau) appear in history-of-science posts: useful curated benchmark, not national modern VR.

## Implications for Phase 1 loaders

1. Always load **age 0 and adult ages** when available.  
2. Prefer HMD/HLD over anecdote tables.  
3. Tag forager/elite rows with `population_type`.  
4. Never collapse Myth A testing into a single infant-mortality-adjusted number without keeping raw multi-age series.
