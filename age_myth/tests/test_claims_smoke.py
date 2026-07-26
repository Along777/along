"""Smoke tests: de-dupe, claim structure, optional live panel checks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dedupe_drops_civilian_and_splits():
    from src.analysis.populations import dedupe_hmd, is_duplicate_subseries

    assert is_duplicate_subseries("FRANCE:_CIVILIAN_POPULATION")
    assert is_duplicate_subseries("GERMANY:_EAST_GERMANY")
    assert is_duplicate_subseries("NEW_ZEALAND:_MAORI")
    assert not is_duplicate_subseries("SWE")
    assert not is_duplicate_subseries("FRANCE:_TOTAL_POPULATION")

    df = pd.DataFrame(
        {
            "region_id": [
                "SWE",
                "FRANCE:_CIVILIAN_POPULATION",
                "FRANCE:_TOTAL_POPULATION",
                "UK:_ENGLAND_&_WALES_CIVILIAN_POPULATION",
            ]
        }
    )
    out = dedupe_hmd(df)
    assert set(out["region_id"]) == {"SWE", "FRANCE:_TOTAL_POPULATION"}


def test_claim_registry_has_required_gates():
    from src.analysis.claim_registry import CLAIMS, SCOPE

    assert "A_birth_e0_not_adult_death_age" in CLAIMS
    assert "B_adults_improved" in CLAIMS
    assert "not_claiming" in SCOPE
    assert CLAIMS["A_birth_e0_not_adult_death_age"]["gates"]["median_exp65_ge"] >= 70


def test_evaluate_gates_structure_with_fixture_claims():
    """Unit-test gate evaluator with synthetic claim payloads (no big data)."""
    from src.analysis.bulletproof_suite import evaluate_gates

    fake = {
        "claim_A": {
            "n_country_years": 100,
            "median_s_to_65": 0.25,
            "median_imr": 180.0,
            "share_exp65_ge_70": 1.0,
            "exp_death_65": {"median": 75.0},
        },
        "claim_A2": {
            "ages": {
                "15": {
                    "median": 56.0,
                    "share_ge_45": 0.97,
                    "n_rows": 40,
                },
                "30": {"median": 60.0},
            }
        },
        "claim_B": {
            "delta_e65": {"mean": 8.0, "ci_low": 7.0, "n": 12},
        },
        "claim_C": {
            "median_within_corr_e0_imr": -0.9,
            "models": {"M3_first_diff": {"beta_dimr": -0.05}},
        },
        "sweden_1800": {
            "e0": 32.19,
            "exp_death_65": 73.73,
            "imr": 227.06,
        },
    }
    gates = evaluate_gates(fake)
    assert gates["A"]["pass"] is True
    assert gates["A2"]["pass"] is True
    assert gates["B"]["pass"] is True
    assert gates["C"]["pass"] is True
    assert gates["S"]["pass"] is True
    assert gates["all_pass"] is True


def test_evaluate_gates_fails_when_myth_a_breaks():
    from src.analysis.bulletproof_suite import evaluate_gates

    fake = {
        "claim_A": {
            "n_country_years": 100,
            "median_s_to_65": 0.25,
            "median_imr": 180.0,
            "share_exp65_ge_70": 0.5,  # fail
            "exp_death_65": {"median": 50.0},  # fail
        },
        "claim_A2": {
            "ages": {
                "15": {"median": 56.0, "share_ge_45": 0.97, "n_rows": 40},
                "30": {"median": 60.0},
            }
        },
        "claim_B": {"delta_e65": {"mean": 8.0, "ci_low": 7.0, "n": 12}},
        "claim_C": {
            "median_within_corr_e0_imr": -0.9,
            "models": {"M3_first_diff": {"beta_dimr": -0.05}},
        },
        "sweden_1800": {"e0": 32.19, "exp_death_65": 73.73, "imr": 227.06},
    }
    gates = evaluate_gates(fake)
    assert gates["A"]["pass"] is False
    assert gates["all_pass"] is False


@pytest.mark.skipif(
    not (ROOT / "data/processed/analysis/hmd_summary_wide_both.parquet").exists(),
    reason="HMD analysis panel not built",
)
def test_live_panel_claim_a_still_holds():
    from src.analysis.final_agrade import claim_a_year_and_equal_country, strict_myth_band
    from src.analysis.ladder import load_hmd_wide

    panel = load_hmd_wide("both")
    assert len(panel) > 100
    # de-dupe should already be in load_hmd_wide path
    a = claim_a_year_and_equal_country(panel, 40.0)
    yw = a["year_weighted"]
    ec = a["equal_country"]
    assert yw["median_exp65"] >= 70
    assert yw["share_exp65_ge_70"] >= 0.95
    assert yw["median_s_to_65"] < 0.40
    assert yw["median_imr"] > 100
    assert ec["median_of_country_medians_exp65"] >= 70
    # equal-country close to year-weighted (robustness, not Sweden-only)
    assert abs(yw["median_exp65"] - ec["median_of_country_medians_exp65"]) < 2.0

    band = strict_myth_band(panel)
    assert band["n_country_years"] >= 20
    assert band["year_weighted"]["median_exp65"] >= 70
    assert band["year_weighted"]["share_exp65_ge_70"] >= 0.95


@pytest.mark.skipif(
    not (ROOT / "data/processed/analysis/hmd_summary_wide_both.parquet").exists(),
    reason="HMD analysis panel not built",
)
def test_sweden_1800_anchor():
    from src.analysis.ladder import load_hmd_wide

    panel = load_hmd_wide("both")
    s = panel[(panel["region_id"] == "SWE") & (panel["year"] == 1800)]
    assert len(s) == 1
    r = s.iloc[0]
    assert abs(float(r["e0"]) - 32.19) < 2.0
    assert abs(float(r["exp_death_65"]) - 73.73) < 2.0
    assert float(r["imr"]) > 150


@pytest.mark.skipif(
    not (ROOT / "outputs/bulletproof/claim_gate_results.json").exists(),
    reason="gates not generated yet",
)
def test_saved_gates_all_pass_if_present():
    gates = json.loads((ROOT / "outputs/bulletproof/claim_gate_results.json").read_text(encoding="utf-8"))
    assert gates.get("all_pass") is True
