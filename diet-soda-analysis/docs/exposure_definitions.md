# Exposure definitions

Primary labels use **USDA WWEIA Food Categories** joined to Day-1 Individual Foods (DR1IFF).

| Code | Definition | Role |
|------|------------|------|
| E1 | WWEIA **7102** diet soft drinks any Day-1 (`asb_any_d1`) | **Primary** |
| E1b | WWEIA **7202** soft drinks any Day-1 (`ssb_any_d1`) | Primary contrast |
| E2 | Broad diet beverages 7102+7104+7106 | Sensitivity |
| E3 | FNDDS description keywords for diet soft drinks | Sensitivity |
| E4 | Continuous grams / servings | Dose |
| E5 | `bev_group` ASB-only vs SSB-only among soft drinkers | Substitution |

Soda-type exclusivity for `bev_group` allows other beverages (water, coffee, etc.).

## Analytic sample counts (unweighted)

| Group | n |
|-------|---|
| Neither | 11558 |
| SSB-only | 5934 |
| ASB-only | 1744 |
| Both | 148 |