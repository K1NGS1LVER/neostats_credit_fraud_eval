"""
Unit tests for NeoStats Credit Risk Intelligence XAI Engine
(EBM Glassbox, LightGBM TreeSHAP Waterfall, Global Importance, Adverse Action Notices)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.xai.explainer import (
    CreditRiskExplainer,
    engineer_applicant_features,
    get_ebm_local_explanation,
    get_shap_local_waterfall,
    get_global_importance,
)
from src.xai.adverse_action import (
    REASON_CODE_REGISTRY,
    extract_adverse_drivers,
    generate_adverse_action_notice,
    _map_feature_to_reason,
)


@pytest.fixture(scope="module")
def explainer():
    return CreditRiskExplainer()


@pytest.fixture
def sample_applicant():
    return {
        "SK_ID_CURR": 100002,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000.0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -9461,
        "DAYS_EMPLOYED": -637,
        "DAYS_REGISTRATION": -3648.0,
        "DAYS_ID_PUBLISH": -2120,
        "OCCUPATION_TYPE": "Laborers",
        "REGION_RATING_CLIENT": 2,
        "EXT_SOURCE_1": 0.0830,
        "EXT_SOURCE_2": 0.2629,
        "EXT_SOURCE_3": 0.1394,
        "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 2.0,
        "DAYS_LAST_PHONE_CHANGE": -1134.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    }


def test_feature_engineering_leverage_ratios(sample_applicant):
    df_feat = engineer_applicant_features(sample_applicant)
    assert "PAYMENT_RATE" in df_feat.columns
    assert "DTI" in df_feat.columns
    assert "CREDIT_INCOME_RATIO" in df_feat.columns
    assert "LTV" in df_feat.columns
    assert "AGE_YEARS" in df_feat.columns
    assert "EMPLOYED_YEARS" in df_feat.columns
    assert "EXT_SOURCES_WEIGHTED" in df_feat.columns

    # Verify mathematical definitions
    expected_payment_rate = sample_applicant["AMT_ANNUITY"] / sample_applicant["AMT_CREDIT"]
    assert np.isclose(df_feat["PAYMENT_RATE"].iloc[0], expected_payment_rate)

    expected_dti = sample_applicant["AMT_ANNUITY"] / sample_applicant["AMT_INCOME_TOTAL"]
    assert np.isclose(df_feat["DTI"].iloc[0], expected_dti)


def test_missing_artifacts_error():
    with pytest.raises(FileNotFoundError):
        CreditRiskExplainer(artifacts_dir="/non_existent_path_xyz")


def test_ebm_local_explanation_exactness(sample_applicant, explainer):
    res = explainer.get_ebm_local_explanation(sample_applicant)

    assert res["explanation_type"] == "EBM_Glassbox_Exact"
    assert res["approximation_error"] == 0.0
    assert "base_score" in res
    assert "total_log_odds" in res
    assert "predicted_probability" in res
    assert len(res["contributions"]) > 0

    # Additive scorecard exactness verification:
    # total_log_odds must equal base_score + sum of term scores
    sum_scores = sum(c["score"] for c in res["contributions"])
    expected_total = res["base_score"] + sum_scores
    assert np.isclose(res["total_log_odds"], expected_total, atol=1e-3)

    # Sigmoid link verification
    expected_prob = 1.0 / (1.0 + np.exp(-res["total_log_odds"]))
    assert np.isclose(res["predicted_probability"], expected_prob, atol=1e-3)

    # Check top drivers exist
    assert len(res["top_risk_drivers"]) > 0


def test_shap_local_waterfall_consistency(sample_applicant, explainer):
    res = explainer.get_shap_local_waterfall(sample_applicant)

    assert res["explanation_type"] == "TreeSHAP_LightGBM"
    assert "base_value" in res
    assert "prediction_margin" in res
    assert "predicted_probability" in res
    assert len(res["waterfall"]) == len(res["contributions"])

    # Waterfall cumulative continuity
    waterfall = res["waterfall"]
    running = res["base_value"]
    for step in waterfall:
        running += step["attribution"]
        assert np.isclose(step["cumulative"], running, atol=1e-4)

    # Top adverse and favorable lists
    assert len(res["top_adverse_drivers"]) <= 5
    for item in res["top_adverse_drivers"]:
        assert item["attribution"] > 0


def test_global_importance(explainer):
    res = explainer.get_global_importance(top_n=10)

    assert "top_ebm_importance" in res
    assert "top_shap_importance" in res
    assert "consensus_ranking" in res

    assert len(res["top_ebm_importance"]) == 10
    assert len(res["top_shap_importance"]) == 10
    assert len(res["consensus_ranking"]) == 10

    # Verify descending ordering
    ebm_scores = [item["ebm_score"] for item in res["top_ebm_importance"]]
    assert ebm_scores == sorted(ebm_scores, reverse=True)

    shap_scores = [item["mean_abs_shap"] for item in res["top_shap_importance"]]
    assert shap_scores == sorted(shap_scores, reverse=True)


def test_reason_code_registry_mappings():
    # Test key required mappings
    pr_reason = _map_feature_to_reason("PAYMENT_RATE")
    assert pr_reason["formal_reason"] == "Excessive debt obligation relative to credit line"

    ext_reason = _map_feature_to_reason("EXT_SOURCES_WEIGHTED")
    assert ext_reason["formal_reason"] == "Adverse credit history reported by external credit bureaus"

    emp_reason = _map_feature_to_reason("EMPLOYED_TO_AGE_RATIO")
    assert emp_reason["formal_reason"] == "Insufficient length of stable employment tenure relative to career timeline"


def test_adverse_action_driver_extraction(sample_applicant, explainer):
    drivers = extract_adverse_drivers(sample_applicant, explainer=explainer, method="ebm", top_k=4)

    assert len(drivers) <= 4
    for d in drivers:
        assert "reason_code" in d
        assert "formal_reason" in d
        assert "category" in d
        assert "applicant_value" in d
        assert "contribution" in d
        assert d["reason_code"].startswith("RC_")


def test_generate_adverse_action_notice(sample_applicant, explainer):
    notice = generate_adverse_action_notice(
        applicant_data=sample_applicant,
        applicant_name="Alex Morgan",
        application_id="APP-TEST-2026",
        decision="DECLINED",
        risk_score=0.35,
        explainer=explainer,
    )

    # Regulatory disclosures verification
    assert "STATEMENT OF ADVERSE ACTION" in notice
    assert "Alex Morgan" in notice
    assert "APP-TEST-2026" in notice
    assert "DECLINED" in notice
    assert "Equifax" in notice
    assert "Experian" in notice
    assert "TransUnion" in notice
    assert "Equal Credit Opportunity Act" in notice
    assert "Consumer Financial Protection Bureau" in notice
    assert "Principal Reasons" in notice
    assert "Credit Model Score" in notice
