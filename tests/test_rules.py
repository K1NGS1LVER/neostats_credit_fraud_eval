"""
Unit tests for NeoStats Credit Risk Intelligence Rules Derivation Engine
(Surrogate Rules Parsing, Tier Filtering, Condition Evaluation, Metrics)
"""

import pytest
import pandas as pd
from pathlib import Path

from src.rules.derivation import (
    Rule,
    load_surrogate_rules,
    get_rules_by_risk_tier,
    evaluate_condition,
    evaluate_applicant,
    evaluate_dataframe,
)


@pytest.fixture(scope="module")
def rules():
    return load_surrogate_rules()


def test_load_surrogate_rules(rules):
    assert len(rules) >= 3
    for r in rules:
        assert isinstance(r, Rule)
        assert r.id.startswith("RULE_")
        assert r.condition != ""
        assert r.action in ["DECLINE", "MANUAL_REVIEW", "AUTO_APPROVE"]
        assert r.risk_tier in ["HIGH", "MEDIUM", "LOW"]
        # Verify required metrics are present
        assert r.support != ""
        assert r.confidence != ""
        assert r.default_rate != ""
        assert r.risk_lift != ""


def test_missing_rules_file_error():
    with pytest.raises(FileNotFoundError):
        load_surrogate_rules("/invalid/path/to/surrogate_rules.json")


def test_filter_by_risk_tier(rules):
    high_rules = get_rules_by_risk_tier("HIGH", rules=rules)
    assert len(high_rules) >= 1
    for r in high_rules:
        assert r.risk_tier == "HIGH"

    med_rules = get_rules_by_risk_tier("medium", rules=rules)
    assert len(med_rules) >= 1
    for r in med_rules:
        assert r.risk_tier == "MEDIUM"

    low_rules = get_rules_by_risk_tier("Low", rules=rules)
    assert len(low_rules) >= 1
    for r in low_rules:
        assert r.risk_tier == "LOW"


def test_evaluate_condition_logic():
    features = {
        "EXT_SOURCES_WEIGHTED": 0.35,
        "PAYMENT_RATE": 0.075,
        "DTI": 0.30,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    }

    # Matches both
    cond_true = "EXT_SOURCES_WEIGHTED <= 0.38 AND PAYMENT_RATE > 0.065"
    assert evaluate_condition(cond_true, features) is True

    # First matches, second fails
    cond_false = "EXT_SOURCES_WEIGHTED <= 0.38 AND PAYMENT_RATE > 0.10"
    assert evaluate_condition(cond_false, features) is False

    # OR clause
    cond_or = "EXT_SOURCES_WEIGHTED > 0.50 OR DTI > 0.25"
    assert evaluate_condition(cond_or, features) is True


def test_evaluate_applicant_high_risk(rules):
    # Weak credit bureau and high payment burden
    applicant = {
        "EXT_SOURCES_WEIGHTED": 0.30,
        "PAYMENT_RATE": 0.070,
        "DTI": 0.18,
    }
    result = evaluate_applicant(applicant, rules=rules)

    assert result["has_triggered"] is True
    assert result["recommended_action"] == "DECLINE"
    assert result["risk_tier"] == "HIGH"
    assert result["primary_rule"]["id"] == "RULE_HIGH_01"
    assert "metrics" in result
    assert result["metrics"]["rule_id"] == "RULE_HIGH_01"
    assert result["metrics"]["default_rate"] == "26.4%"
    assert result["metrics"]["risk_lift"] == "3.27x"


def test_evaluate_applicant_medium_risk(rules):
    # Moderate credit with high DTI
    applicant = {
        "EXT_SOURCES_WEIGHTED": 0.42,
        "PAYMENT_RATE": 0.050,
        "DTI": 0.28,
    }
    result = evaluate_applicant(applicant, rules=rules)

    assert result["has_triggered"] is True
    assert result["recommended_action"] == "MANUAL_REVIEW"
    assert result["risk_tier"] == "MEDIUM"
    assert result["primary_rule"]["id"] == "RULE_MED_02"
    assert result["metrics"]["default_rate"] == "14.2%"


def test_evaluate_applicant_low_risk(rules):
    # Prime bureau and low payment rate
    applicant = {
        "EXT_SOURCES_WEIGHTED": 0.62,
        "PAYMENT_RATE": 0.045,
        "DTI": 0.10,
    }
    result = evaluate_applicant(applicant, rules=rules)

    assert result["has_triggered"] is True
    assert result["recommended_action"] == "AUTO_APPROVE"
    assert result["risk_tier"] == "LOW"
    assert result["primary_rule"]["id"] == "RULE_LOW_03"
    assert result["metrics"]["default_rate"] == "1.8%"


def test_evaluate_applicant_standard(rules):
    # Values between thresholds
    applicant = {
        "EXT_SOURCES_WEIGHTED": 0.48,
        "PAYMENT_RATE": 0.058,
        "DTI": 0.15,
    }
    result = evaluate_applicant(applicant, rules=rules)

    assert result["has_triggered"] is False
    assert result["recommended_action"] == "STANDARD_UNDERWRITING"
    assert result["risk_tier"] == "NEUTRAL"
    assert result["primary_rule"] is None


def test_evaluate_dataframe_empirical_metrics(rules):
    # Create synthetic test dataframe
    data = {
        "EXT_SOURCES_WEIGHTED": [0.30, 0.40, 0.65, 0.50, 0.25],
        "PAYMENT_RATE": [0.070, 0.050, 0.040, 0.055, 0.080],
        "DTI": [0.20, 0.30, 0.10, 0.15, 0.25],
        "TARGET": [1, 0, 0, 0, 1],
    }
    df = pd.DataFrame(data)

    eval_summary = evaluate_dataframe(df, rules=rules)

    assert eval_summary["total_population"] == 5
    assert "baseline_default_rate" in eval_summary
    assert "overall_rule_coverage" in eval_summary
    assert len(eval_summary["rule_evaluations"]) == len(rules)

    for rule_eval in eval_summary["rule_evaluations"]:
        assert "rule_id" in rule_eval
        assert "matched_applicants" in rule_eval
        assert "support" in rule_eval
        assert "confidence" in rule_eval
        assert "default_rate" in rule_eval
        assert "risk_lift" in rule_eval
