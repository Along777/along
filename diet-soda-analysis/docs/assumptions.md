# Analytic assumptions & known limitations

## Survey design

- Multi-cycle weight: `w_mec = WTMEC2YR / 4` (four 2-year cycles 2011–2018).
- Fasting: `w_fast = WTSAF2YR / 4` when used.
- **PSU (`SDMVPSU`) and strata (`SDMVSTRA`) are stored but not yet used for variance estimation.**
- Therefore:
  - **Coefficients** from weighted least squares are reasonable for descriptive weighted associations.
  - **Standard errors and p-values are approximate** (typically too small vs full design-based variance).
  - Binary GLMs use **normalized** sampling weights (`w / mean(w)`), never raw MEC totals as frequency weights.

### Bug fixed (2026-07-26)

Using `freq_weights=w_mec` (raw) in binomial GLM treated each adult as ~10⁴ observations, producing p-values of 0 and meaningless CIs for cancer/mortality logits. Fixed by normalizing weights.

## Exposure

- Primary: WWEIA category **7102** (diet soft drinks) vs **7202** (soft drinks), Day-1 24h recall.
- `bev_group` is exclusive on **soda type** only (other beverages allowed).

## Sample

- Adults ≥20, not pregnant (`RIDEXPRG != 1`), MEC weight > 0, Day-1 exposure row, `DR1DRSTZ == 1`.
- Special NHANES missing codes (education 7/9, sedentary 7777/9999) set to NaN.

## Causal language

Cross-sectional NHANES + single-day diet **cannot** establish that diet soda causes obesity, diabetes, or cancer. Myth verdicts are about **claims vs patterns in public data**, not RCT truth.
