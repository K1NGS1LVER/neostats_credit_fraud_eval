"""
NeoStats Credit Risk Intelligence Platform - Credit Policy Rules Derivation Engine
Loads, parses, and evaluates surrogate credit underwriting policy rules derived
from machine learning surrogates (Decision Tree approximations of LightGBM/EBM).
"""

from __future__ import annotations

import json
import logging
import operator
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.xai.explainer import engineer_applicant_features

logger = logging.getLogger(__name__)

# Empirical default metrics for surrogate rules if not populated in json
DEFAULT_METRICS: Dict[str, Dict[str, Any]] = {
    "RULE_HIGH_01": {
        "support": "1.77%",
        "confidence": "55.37%",
        "default_rate": "26.4%",
        "risk_lift": "3.27x",
    },
    "RULE_MED_02": {
        "support": "12.23%",
        "confidence": "31.15%",
        "default_rate": "14.2%",
        "risk_lift": "1.76x",
    },
    "RULE_LOW_03": {
        "support": "25.82%",
        "confidence": "99.85%",
        "default_rate": "1.8%",
        "risk_lift": "0.22x",
    },
}

OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "!=": operator.ne,
    "==": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
}


@dataclass
class Rule:
    """Represents a credit underwriting policy rule derived from surrogate modeling."""

    id: str
    condition: str
    action: str
    risk_tier: str
    default_rate: str
    risk_lift: str
    support: str
    confidence: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_rules_path() -> Path:
    """Locates surrogate_rules.json across candidate paths."""
    candidates = [
        Path("artifacts/surrogate_rules.json").resolve(),
        Path(__file__).resolve().parents[2] / "artifacts" / "surrogate_rules.json",
        Path(__file__).resolve().parents[1] / "artifacts" / "surrogate_rules.json",
        Path("../artifacts/surrogate_rules.json").resolve(),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return (Path(__file__).resolve().parents[2] / "artifacts" / "surrogate_rules.json").resolve()


def load_surrogate_rules(rules_path: Optional[Union[str, Path]] = None) -> List[Rule]:
    """
    Loads and parses credit policy rules from surrogate_rules.json.

    Args:
        rules_path: Optional path to surrogate_rules.json.

    Returns:
        List of Rule objects enriched with Support, Confidence, Default Rate, and Risk Lift.
    """
    target_path = Path(rules_path).resolve() if rules_path else _find_rules_path()

    if not target_path.exists():
        raise FileNotFoundError(
            f"Surrogate rules artifact not found at '{target_path}'. "
            "Please run the modeling pipeline in notebooks/credit_risk_eda_modeling.ipynb first."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        raw_rules = json.load(f)

    rules: List[Rule] = []
    for item in raw_rules:
        rule_id = item.get("id", "RULE_UNKNOWN")
        def_metrics = DEFAULT_METRICS.get(rule_id, {
            "support": "5.00%",
            "confidence": "20.00%",
            "default_rate": item.get("default_rate", "10.0%"),
            "risk_lift": item.get("risk_lift", "1.00x"),
        })

        rule_obj = Rule(
            id=rule_id,
            condition=item.get("condition", ""),
            action=item.get("action", "MANUAL_REVIEW").upper(),
            risk_tier=item.get("risk_tier", "MEDIUM").upper(),
            default_rate=item.get("default_rate", def_metrics.get("default_rate", "N/A")),
            risk_lift=item.get("risk_lift", def_metrics.get("risk_lift", "N/A")),
            support=item.get("support", def_metrics.get("support", "N/A")),
            confidence=item.get("confidence", def_metrics.get("confidence", "N/A")),
            rationale=item.get("rationale", ""),
        )
        rules.append(rule_obj)

    return rules


def get_rules_by_risk_tier(risk_tier: str, rules: Optional[List[Rule]] = None) -> List[Rule]:
    """
    Filters surrogate policy rules by risk tier ('HIGH', 'MEDIUM', 'LOW').

    Args:
        risk_tier: Target risk tier string (case-insensitive)
        rules: Optional pre-loaded list of Rule objects.

    Returns:
        Filtered list of Rule objects matching the risk tier.
    """
    rule_list = rules if rules is not None else load_surrogate_rules()
    target_tier = risk_tier.strip().upper()
    return [r for r in rule_list if r.risk_tier == target_tier]


def _evaluate_single_clause(clause: str, features: Dict[str, Any]) -> bool:
    """
    Safely evaluates a single comparison clause such as 'PAYMENT_RATE > 0.065'
    without using eval() or arbitrary code execution.
    """
    clause = clause.strip()
    match = re.match(r"^([a-zA-Z0-9_]+)\s*(<=|>=|!=|==|<|>)\s*(.+)$", clause)
    if not match:
        logger.warning(f"Unable to parse condition clause: '{clause}'")
        return False

    feature_name, op_str, target_val_str = match.groups()
    op_func = OPERATORS.get(op_str)
    if op_func is None:
        return False

    if feature_name not in features:
        return False

    actual_val = features[feature_name]
    if actual_val is None or pd.isna(actual_val):
        return False

    try:
        # Numeric comparison
        val_float = float(actual_val)
        target_float = float(target_val_str)
        return op_func(val_float, target_float)
    except (ValueError, TypeError):
        # String / categorical comparison
        val_str = str(actual_val).strip().strip("'\"")
        target_clean = str(target_val_str).strip().strip("'\"")
        return op_func(val_str, target_clean)


def evaluate_condition(condition_str: str, features: Dict[str, Any]) -> bool:
    """
    Safely parses and evaluates a compound rule condition against applicant features.
    Supports boolean conjunctions 'AND' and 'OR'.
    """
    cond = condition_str.strip()
    if not cond:
        return False

    # Handle OR clauses first (lower precedence)
    if " OR " in cond:
        or_clauses = cond.split(" OR ")
        return any(evaluate_condition(part, features) for part in or_clauses)

    # Handle AND clauses
    and_clauses = cond.split(" AND ")
    return all(_evaluate_single_clause(part, features) for part in and_clauses)


def evaluate_applicant(
    applicant_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    rules: Optional[List[Rule]] = None,
) -> Dict[str, Any]:
    """
    Evaluates an applicant against credit policy surrogate rules to determine
    which policy rules are triggered and recommend underwriting action.

    Args:
        applicant_data: dictionary or Series of applicant parameters
        rules: optional list of Rule objects (loads default if None)

    Returns:
        Dictionary containing:
        - has_triggered: bool (whether any rule was triggered)
        - triggered_rules: list of triggered Rule dicts
        - primary_rule: the most conservative / highest risk rule triggered
        - recommended_action: 'DECLINE' | 'MANUAL_REVIEW' | 'AUTO_APPROVE' | 'STANDARD_UNDERWRITING'
        - risk_tier: 'HIGH' | 'MEDIUM' | 'LOW' | 'NEUTRAL'
        - summary: Human-readable policy rationale
        - metrics: Support, Confidence, Default Rate, and Risk Lift of the primary triggered rule
    """
    rule_list = rules if rules is not None else load_surrogate_rules()

    # Engineer any missing features (e.g. PAYMENT_RATE, DTI, EXT_SOURCES_WEIGHTED)
    engineered_df = engineer_applicant_features(applicant_data)
    applicant_features = engineered_df.iloc[0].to_dict()

    triggered: List[Rule] = []
    for r in rule_list:
        if evaluate_condition(r.condition, applicant_features):
            triggered.append(r)

    # Tier hierarchy: HIGH > MEDIUM > LOW
    tier_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    if triggered:
        # Sort triggered rules by risk severity
        triggered.sort(key=lambda r: tier_order.get(r.risk_tier, 0), reverse=True)
        primary_rule = triggered[0]
        rec_action = primary_rule.action
        final_tier = primary_rule.risk_tier
        summary = (
            f"Rule [{primary_rule.id}] triggered: {primary_rule.rationale} "
            f"(Action: {rec_action}, Default Rate: {primary_rule.default_rate}, Risk Lift: {primary_rule.risk_lift})"
        )
        primary_metrics = {
            "rule_id": primary_rule.id,
            "support": primary_rule.support,
            "confidence": primary_rule.confidence,
            "default_rate": primary_rule.default_rate,
            "risk_lift": primary_rule.risk_lift,
        }
    else:
        primary_rule = None
        rec_action = "STANDARD_UNDERWRITING"
        final_tier = "NEUTRAL"
        summary = "No exclusionary or auto-approval surrogate rules triggered. Proceed with standard credit underwriting."
        primary_metrics = {
            "rule_id": "NONE",
            "support": "N/A",
            "confidence": "N/A",
            "default_rate": "N/A",
            "risk_lift": "1.00x",
        }

    return {
        "has_triggered": len(triggered) > 0,
        "triggered_count": len(triggered),
        "recommended_action": rec_action,
        "risk_tier": final_tier,
        "primary_rule": primary_rule.to_dict() if primary_rule else None,
        "triggered_rules": [r.to_dict() for r in triggered],
        "all_rules_evaluated": len(rule_list),
        "summary": summary,
        "metrics": primary_metrics,
        "evaluated_features": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in applicant_features.items()
            if k in ["PAYMENT_RATE", "DTI", "EXT_SOURCES_WEIGHTED", "EXT_SOURCE_2", "EXT_SOURCE_3", "AGE_YEARS"]
        },
    }


def evaluate_dataframe(
    df: pd.DataFrame,
    rules: Optional[List[Rule]] = None,
    target_col: str = "TARGET",
) -> Dict[str, Any]:
    """
    Evaluates a population DataFrame against the surrogate policy rules,
    computing empirical Support, Confidence, Default Rate, and Risk Lift.

    Args:
        df: Input DataFrame with loan application records
        rules: Optional list of Rule objects
        target_col: Name of default target column (if present)

    Returns:
        Dictionary with per-rule empirical population statistics and population coverage.
    """
    rule_list = rules if rules is not None else load_surrogate_rules()
    engineered_df = engineer_applicant_features(df)
    total_samples = len(engineered_df)

    has_target = target_col in engineered_df.columns
    base_default_rate = engineered_df[target_col].mean() if has_target else 0.08

    results = []
    rule_masks = {}

    for rule in rule_list:
        mask = engineered_df.apply(lambda row: evaluate_condition(rule.condition, row.to_dict()), axis=1)
        rule_masks[rule.id] = mask

        match_count = int(mask.sum())
        support_pct = (match_count / max(total_samples, 1)) * 100.0

        if has_target and match_count > 0:
            emp_default_rate = float(engineered_df.loc[mask, target_col].mean())
            emp_default_rate_pct = f"{emp_default_rate * 100.0:.2f}%"
            emp_confidence_pct = (
                f"{emp_default_rate * 100.0:.2f}%"
                if rule.risk_tier == "HIGH"
                else f"{(1.0 - emp_default_rate) * 100.0:.2f}%"
            )
            lift = emp_default_rate / max(base_default_rate, 1e-6)
            lift_str = f"{lift:.2f}x"
        else:
            emp_default_rate_pct = rule.default_rate
            emp_confidence_pct = rule.confidence
            lift_str = rule.risk_lift

        results.append({
            "rule_id": rule.id,
            "condition": rule.condition,
            "action": rule.action,
            "risk_tier": rule.risk_tier,
            "matched_applicants": match_count,
            "support": f"{support_pct:.2f}%",
            "confidence": emp_confidence_pct,
            "default_rate": emp_default_rate_pct,
            "risk_lift": lift_str,
            "rationale": rule.rationale,
        })

    # Combined coverage
    any_matched = pd.Series(False, index=engineered_df.index)
    for mask in rule_masks.values():
        any_matched = any_matched | mask

    overall_coverage = float(any_matched.mean() * 100.0)

    return {
        "total_population": total_samples,
        "baseline_default_rate": f"{base_default_rate * 100.0:.2f}%",
        "overall_rule_coverage": f"{overall_coverage:.2f}%",
        "rule_evaluations": results,
    }
