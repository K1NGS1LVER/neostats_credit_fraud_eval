"""
NeoStats Credit Risk Intelligence Platform - Explainable AI (XAI) Package
"""

from src.xai.explainer import (
    CreditRiskExplainer,
    engineer_applicant_features,
    get_ebm_local_explanation,
    get_global_importance,
    get_shap_local_waterfall,
)
from src.xai.adverse_action import (
    REASON_CODE_REGISTRY,
    extract_adverse_drivers,
    format_feature_value,
    generate_adverse_action_notice,
)

__all__ = [
    "CreditRiskExplainer",
    "engineer_applicant_features",
    "get_ebm_local_explanation",
    "get_shap_local_waterfall",
    "get_global_importance",
    "REASON_CODE_REGISTRY",
    "extract_adverse_drivers",
    "format_feature_value",
    "generate_adverse_action_notice",
]
