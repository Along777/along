# Peer Review Response

| Issue | Severity | Response |
|-------|----------|----------|
| FE R²=0.97 looks overfit | High (comms) | Added model stack M0–M3; within R²; first differences; LOCO β; FE demoted to sensitivity. High R² expected (shared mortality schedule). |
| Bootstrap CI too tight | High | Cluster bootstrap by country for Myth A/B. |
| France/UK TOTAL+CIVILIAN double count | Medium | De-duplication filter drops CIVILIAN/East/West/ethnic subseries. |
| Age 65 only for “adults” | High (claim scope) | Added HLD e15/e30 mid-adult analysis + figures G2. |
| Correlation vs time collinearity | Medium | Report corr(IMR,year); first-difference model M3; within+year M2. |
| Not enough charts | Medium | Peer pack G1–G10 + survival composition + coverage honesty. |
| Overclaim “the past” globally | Medium | Findings scoped to HMD-like VR populations; G10 coverage heatmap. |
| Concordance r≈1 independent validation | Low | Note shared HMD lineage with OWID for modern e0. |

**Verdict after remediation:** Myth-busting claims remain **directionally correct and stronger under de-dupe + mid-adult checks**. Statistical presentation is peer-safe if FE R² is not headlined.
