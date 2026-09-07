"""
NeoStats Credit Risk Intelligence Platform - Rules Engine Package
"""

from src.rules.derivation import (
    Rule,
    evaluate_applicant,
    evaluate_condition,
    evaluate_dataframe,
    get_rules_by_risk_tier,
    load_surrogate_rules,
)

__all__ = [
    "Rule",
    "load_surrogate_rules",
    "get_rules_by_risk_tier",
    "evaluate_condition",
    "evaluate_applicant",
    "evaluate_dataframe",
]
