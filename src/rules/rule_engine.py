"""
Credit Policy Decision Rule Engine.
Evaluates machine learning derived surrogate rules against applicants and portfolios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "id": "RULE_HIGH_01",
        "condition": "EXT_SOURCES_WEIGHTED <= 0.38 AND PAYMENT_RATE > 0.065",
        "action": "DECLINE",
        "risk_tier": "HIGH",
        "default_rate": "26.4%",
        "risk_lift": "3.27x",
        "rationale": "Compounded vulnerability: Weak credit bureau ratings coupled with high monthly payment burden.",
    },
    {
        "id": "RULE_HIGH_02",
        "condition": "DTI > 0.35 AND EXT_SOURCES_WEIGHTED <= 0.42",
        "action": "DECLINE",
        "risk_tier": "HIGH",
        "default_rate": "24.1%",
        "risk_lift": "2.98x",
        "rationale": "Severe debt overextension: Over 35% of gross income dedicated to debt payments alongside subprime credit.",
    },
    {
        "id": "RULE_MED_02",
        "condition": "EXT_SOURCES_WEIGHTED <= 0.45 AND DTI > 0.22",
        "action": "MANUAL_REVIEW",
        "risk_tier": "MEDIUM",
        "default_rate": "14.2%",
        "risk_lift": "1.76x",
        "rationale": "Moderate credit rating with debt obligation consuming >22% of gross monthly income.",
    },
    {
        "id": "RULE_MED_03",
        "condition": "PAYMENT_RATE > 0.075",
        "action": "MANUAL_REVIEW",
        "risk_tier": "MEDIUM",
        "default_rate": "12.8%",
        "risk_lift": "1.59x",
        "rationale": "High payment velocity cliff: Annuity payment exceeds 7.5% of total loan principal.",
    },
    {
        "id": "RULE_LOW_03",
        "condition": "EXT_SOURCES_WEIGHTED > 0.52 AND PAYMENT_RATE <= 0.055",
        "action": "AUTO_APPROVE",
        "risk_tier": "LOW",
        "default_rate": "1.8%",
        "risk_lift": "0.22x",
        "rationale": "Prime credit bureau rating with sustainable, low-annuity loan repayment terms.",
    },
]


def load_rules(rules_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load surrogate rules from JSON artifact or fallback to default curated rules."""
    candidates = [
        Path(rules_path).resolve() if rules_path else None,
        Path("artifacts/surrogate_rules.json").resolve(),
        Path(__file__).resolve().parents[2] / "artifacts" / "surrogate_rules.json",
    ]
    for p in candidates:
        if p and p.exists():
            try:
                with open(p, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list) and len(loaded) > 0:
                        return loaded
            except Exception:
                pass
    return DEFAULT_RULES


def _check_condition(row: Dict[str, Any], rule_id: str) -> bool:
    """Evaluate individual rule logic against applicant feature dictionary."""
    ext = float(row.get("EXT_SOURCES_WEIGHTED", 0.5))
    pmt = float(row.get("PAYMENT_RATE", 0.05))
    dti = float(row.get("DTI", 0.15))

    if rule_id == "RULE_HIGH_01":
        return ext <= 0.38 and pmt > 0.065
    elif rule_id == "RULE_HIGH_02":
        return dti > 0.35 and ext <= 0.42
    elif rule_id == "RULE_MED_02":
        return ext <= 0.45 and dti > 0.22
    elif rule_id == "RULE_MED_03":
        return pmt > 0.075
    elif rule_id == "RULE_LOW_03":
        return ext > 0.52 and pmt <= 0.055
    return False


def evaluate_applicant(
    applicant: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate applicant features against all active credit policy rules.
    Returns:
        dict with:
        - triggered_rules: list of rules triggered
        - primary_verdict: 'DECLINE' | 'MANUAL_REVIEW' | 'AUTO_APPROVE' | 'STANDARD_REVIEW'
        - highest_risk_tier: 'HIGH' | 'MEDIUM' | 'LOW' | 'NEUTRAL'
        - policy_rationale: summary text
    """
    if isinstance(applicant, pd.DataFrame):
        row = applicant.iloc[0].to_dict()
    elif isinstance(applicant, pd.Series):
        row = applicant.to_dict()
    else:
        row = dict(applicant)

    rule_list = rules or load_rules()
    triggered = []

    for rule in rule_list:
        rid = rule.get("id", "")
        if _check_condition(row, rid):
            triggered.append(rule)

    actions = [r.get("action", "") for r in triggered]
    if "DECLINE" in actions:
        verdict = "DECLINE"
        tier = "HIGH"
        summary = "Triggered hard decline rule: Applicant breaches key risk safety thresholds."
    elif "MANUAL_REVIEW" in actions:
        verdict = "MANUAL_REVIEW"
        tier = "MEDIUM"
        summary = "Triggered supervisory review: Elevated risk requires secondary underwriting evaluation."
    elif "AUTO_APPROVE" in actions:
        verdict = "AUTO_APPROVE"
        tier = "LOW"
        summary = "Triggered prime fast-track rule: Exceptional credit stability qualifies for expedited approval."
    else:
        verdict = "STANDARD_REVIEW"
        tier = "NEUTRAL"
        summary = "No exceptional surrogate rules triggered. Process through standard score-based underwriting."

    return {
        "triggered_rules": triggered,
        "primary_verdict": verdict,
        "highest_risk_tier": tier,
        "policy_rationale": summary,
        "applicant_features": {
            "EXT_SOURCES_WEIGHTED": round(float(row.get("EXT_SOURCES_WEIGHTED", 0.5)), 4),
            "PAYMENT_RATE": round(float(row.get("PAYMENT_RATE", 0.05)), 4),
            "DTI": round(float(row.get("DTI", 0.15)), 4),
        },
    }
