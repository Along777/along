"""Machine-readable claim definitions and pass thresholds."""
from __future__ import annotations

CLAIMS = {
    "A_birth_e0_not_adult_death_age": {
        "description": "When e0<40, expected age if alive at 65 is far above 30-35",
        "gates": {
            "median_exp65_ge": 70.0,
            "share_exp65_ge_70_ge": 0.95,
            "median_s_to_65_lt": 0.40,
            "median_imr_gt": 100.0,
            "min_n_country_years": 50,
        },
    },
    "A2_mid_adult": {
        "description": "When e0<40, expected age if alive at 15/30 still >> 35",
        "gates": {
            "median_exp15_ge": 50.0,
            "median_exp30_ge": 55.0,
            "share_exp15_ge_45_ge": 0.90,
            "min_n_low_e0": 30,
        },
        "prefer_hld_quality": "gold",  # fall back to median if gold thin
    },
    "B_adults_improved": {
        "description": "Remaining LE at 65 rose from pre-1900 to post-2000",
        "gates": {
            "mean_delta_e65_gt": 5.0,
            "ci_low_gt": 0.0,
            "min_countries": 8,
        },
    },
    "C_infant_mechanism": {
        "description": "e0 co-moves with IMR (associational)",
        "gates": {
            "median_within_corr_lt": -0.7,
            "first_diff_beta_imr_lt": 0.0,
        },
    },
    "S_sweden_1800_snapshot": {
        "description": "Sweden 1800 sanity anchor",
        "gates": {
            "year": 1800,
            "e0_tol": 2.0,
            "e0_target": 32.19,
            "exp65_tol": 2.0,
            "exp65_target": 73.73,
            "imr_tol": 30.0,
            "imr_target": 227.06,
        },
    },
}

SCOPE = {
    "populations": "High-quality vital registration (HMD public summary); HLD for mid-adult ages",
    "not_claiming": [
        "Global premodern humanity outside VR data",
        "Cohort (lived) lifespan equal to period e0",
        "Causal effect of IMR on e0 from regressions",
        "Everyone reached age 65 historically",
    ],
    "measure_notes": [
        "expected_age_if_alive_at_x = x + period remaining LE e(x)",
        "S(0->65) is required co-evidence for Claim A",
        "Period life tables apply one year age-specific rates",
    ],
}
