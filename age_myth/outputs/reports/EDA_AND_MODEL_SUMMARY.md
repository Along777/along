# EDA and Model Findings — Double-Checked Summary

**Verified:** 2026-07-26  
**Sources:** `outputs/bulletproof/final_claims.json`, `claims.json`, live HMD wide panel  
**Quality gates:** `all_pass=True` · **pytest:** 7 passed  

**Data primary:** De-duplicated HMD public summary (both sexes)  
**Data supporting mid-adult ages:** HLD gold (`n_tables==1`) + HLD median  

---

## Executive summary

| Myth | Finding | Verdict |
|------|---------|---------|
| **A.** “e0 ≈ 30–35 means adults died at 30–35” | When e0 &lt; 40: median age **if alive at 65** ≈ **75.4**, but only **~27%** survive to 65; IMR ≈ **196**/1000. At age **15**, expected age already **~56–58**. | **Rejected** |
| **B.** “Only infant mortality improved; adults always lived modern lengths” | Remaining LE at 65 rose **+8.5 years** (pre-1900 → post-2000; 12 countries; CI ~7.7–9.1). | **Rejected** |

**One careful sentence:** Historical period e0 near 30–40 reflects **early-life mortality**, not typical adult ages at death; conditional ages at 15 and 65 sit far above 35, and adult remaining LE still rose substantially into the modern era.

**Scope (always):** High-quality vital-registration populations (HMD-like), **period** life tables—not all past humans, not cohort lifespans.

---

## 1. EDA — what the data look like

### Coverage

| Item | Value |
|------|------:|
| HMD wide panel (de-duped, both sexes) | **4,277** country-years |
| Regions | **44** |
| Year span | **1751–2023** |
| Longest series | **Sweden** (1751–2023, 273 years) |
| France total | from 1816 |
| England & Wales total | from 1841 |
| USA | from **1933** only |
| Low-e0 myth sample (e0 &lt; 40) | **211** years, **11** countries |

**EDA takeaway:** The “myth zone” (e0 in the 30s) is mostly **19th-century (and earlier Swedish) VR Europe**, not a balanced global sample.

### Sweden 1800 anchor

| Metric | Value |
|--------|------:|
| e0 | **32.2** |
| IMR (per 1,000) | **227** |
| Expected age if alive at 65 | **73.7** |
| Survival birth → 65 | **0.21 (21%)** |

### Patterns seen in charts

| Pattern | Figures |
|---------|---------|
| e0 low while age\|65 stays high | D1, FA storyboard |
| Low e0 → age\|65 clusters ~75 | F2, FA1 |
| e0 tracks IMR (strong negative) | D7, F3 |
| Deaths concentrated early (composition) | D6 |
| Full age ladder when e0 &lt; 40 | D2, FA3 |
| Adult e65 rose over time | D10 |
| Year-weight ≈ equal-country | FA1 |
| Strict e0 ∈ [30, 35] same story | FA2 |

---

## 2. Myth A — detailed findings (verified)

### Primary: e0 &lt; 40 (de-duplicated)

| Estimand | Value |
|----------|------:|
| n country-years | **211** |
| n countries | **11** |
| Median e0 | **37.0** |
| Median IMR | **195.9** per 1,000 |
| Median S(0→65) | **0.270** |
| Median expected age if alive at 65 | **75.39** |
| Share age\|65 ≥ 70 | **100%** (both sexes combined) |
| Equal-country median of medians | **75.37** |
| Country medians range | **74.78 – 76.08** |

**Equal-country check:** Year-weighting does **not** invent the result (Sweden has many years, but every country median is ~75).

### Strict myth band: 30 ≤ e0 ≤ 35

| Estimand | Value |
|----------|------:|
| n | **45** country-years, **8** countries |
| Median e0 | 33.5 |
| Median IMR | **213** |
| Median S(0→65) | **0.22** |
| Median age\|65 | **75.07** |
| Share ≥ 70 | **100%** |
| Min age\|65 in band | 73.63 |

Even in the exact “lived to 30–35” window, conditional age\|65 stays ~75; early mortality is **worse** (higher IMR, lower S65).

### Sex-specific

| Sex | n | Median age\|65 | Share ≥70 | Min age\|65 |
|-----|--:|---------------:|----------:|------------:|
| Female | 150 | 75.37 | **99.3%** | **69.88** |
| Male | 295 | 75.31 | **100%** | 70.3 |

**Female floor — Iceland 1843:** e0=28.4, IMR=321, S(0→65)=**0.067**, age\|65=**69.9**.  
Still ~70, not 30. Do **not** claim “100% of female years ≥70” without this footnote.

### Mid-adult ladder (HLD) — when e0 &lt; 40

Median **expected age if alive at x** (= x + e(x)):

| Age x | Gold (n_tables=1) | Median (multi-table) |
|------:|------------------:|---------------------:|
| 0 | ~35 | ~37 |
| 5 | ~52 | ~53 |
| **15** | **~56** | **~58** |
| 20 | ~57 | ~59 |
| **30** | **~60** | **~62** |
| 50 | ~68 | ~69 |
| **65** | **~75** | **~75** |

By age **15**, conditional expected age is already mid-50s.  
HLD is **supporting** (heterogeneous methods); HMD summary is **primary** for e0 / e65 / IMR / S65.

### Correct reading of Myth A

| Say this | Not this |
|----------|----------|
| Conditional on age 65, expected age ~75 | “They lived to 75” |
| Only ~27% survived to 65 | “Most people reached old age” |
| High IMR drives low e0 | “Adults dropped dead at 30” |

---

## 3. Myth B — detailed findings (verified)

| Estimand | Value |
|----------|------:|
| Mean Δ remaining LE at 65 (post-2000 − pre-1900) | **+8.45 years** |
| 95% CI (country bootstrap) | **7.74 – 9.14** |
| Median Δe65 | +8.44 |
| Range | +6.4 to +10.7 |
| n countries (de-duplicated) | **12** |

Adult remaining longevity **rose** ~8–9 years. Rejects “only infants improved.”

---

## 4. Model / association findings (verified)

**Goal:** Describe co-movement of e0 and IMR—not predict or causal ID.

| Model | R² | β(IMR) | Meaning |
|-------|---:|-------:|---------|
| M0 pooled e0 ~ IMR | **0.92** | −0.21 | Strong raw link |
| M1 within-country | **0.91** | −0.21 | Not only between-country |
| M2 within + year | **0.95** | −0.12 | Still negative with linear time |
| M3 first difference Δe0 ~ ΔIMR | **0.51** | −0.11 | r(Δe0, ΔIMR) ≈ **−0.71** |

| Correlation | Value |
|-------------|------:|
| Median within-country corr(e0, IMR) | **−0.94** |

### Model caveats (double-checked)

1. **Not ML overfit** — simple specs; high R² is expected because e0 and IMR come from the **same period mortality schedule**.  
2. **Not causal** — shared drivers (sanitation, nutrition, medicine).  
3. **IMR collinear with year** — prefer M2/M3 language over “the IMR effect is −0.12.”  
4. **Do not present FE R² as the hero metric.**

---

## 5. Joint conclusion (EDA + stats + models)

```
Myth quote uses period e0 ~ 30–35.
EDA shows those years have extreme IMR and low survival to 15/65.
Conditional expected ages at 15 and 65 are mid-50s to mid-70s.
Associations: e0 moves tightly with IMR (descriptive).
Adults: e65 still rose ~8.5 years into the 21st century.

→ Reject “died at 30.”
→ Reject “only child survival improved.”
```

---

## 6. Limits (attach to every presentation)

1. **VR populations** (HMD-like), not global premodern humanity.  
2. **Period** life expectancy ≠ lived **cohort** lifespan.  
3. Conditional ages require **surviving to that age**.  
4. HLD mid-adult tables are **heterogeneous**; gold coverage incomplete for some countries.  
5. Internal scorecard letter is an **engineering checklist**, not external peer review.

---

## 7. Reproduce

```powershell
cd age_myth
python -m src.analysis.run_final_agrade
python -m pytest -q
```

| Artifact | Path |
|----------|------|
| This summary | `outputs/reports/EDA_AND_MODEL_SUMMARY.md` |
| Full A-grade report | `outputs/reports/FINAL_A_GRADE_REPORT.md` |
| Stakeholder defense | `docs/STAKEHOLDER_DEFENSE.md` |
| HTML board | `outputs/reports/myth_bust_board.html` |
| Claims JSON | `outputs/bulletproof/final_claims.json` |
