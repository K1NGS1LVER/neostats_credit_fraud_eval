"""
NeoStats Credit Risk Intelligence Platform - Regulatory Adverse Action Engine
Compliant with FCRA (Fair Credit Reporting Act, 15 U.S.C. § 1681m(a)) and
ECOA (Equal Credit Opportunity Act, 12 C.F.R. § 1002.9 - Regulation B).

Translates machine learning feature contributions (EBM & TreeSHAP) into
formal, standardized consumer credit denial reason codes and generates
ready-to-issue Adverse Action Notices in Markdown format.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.xai.explainer import CreditRiskExplainer, _get_default_explainer

# Standardized Consumer Credit Reason Code Mapping (FCRA / ECOA Reg B compliant)
REASON_CODE_REGISTRY: Dict[str, Dict[str, str]] = {
    "PAYMENT_RATE": {
        "code": "RC_PR_01",
        "formal_reason": "Excessive debt obligation relative to credit line",
        "category": "Debt Burden & Leverage",
        "guidance": "Reducing the requested loan principal or extending loan tenure to lower the monthly payment rate will improve debt serviceability.",
    },
    "DTI": {
        "code": "RC_DTI_02",
        "formal_reason": "Debt-to-income ratio exceeds acceptable underwriting threshold",
        "category": "Debt Burden & Leverage",
        "guidance": "Lowering outstanding monthly debt obligations relative to verified gross monthly income will improve eligibility.",
    },
    "CREDIT_INCOME_RATIO": {
        "code": "RC_CIR_03",
        "formal_reason": "Requested credit limit is disproportionately high relative to verified annual income",
        "category": "Debt Burden & Leverage",
        "guidance": "Applying for a smaller credit facility aligned with documented income levels is recommended.",
    },
    "LTV": {
        "code": "RC_LTV_04",
        "formal_reason": "Loan-to-value ratio exceeds risk tolerance; insufficient collateral backing",
        "category": "Collateral & Facility Structure",
        "guidance": "Providing a higher down payment or securing higher valued goods reduces financing leverage.",
    },
    "EXT_SOURCES_WEIGHTED": {
        "code": "RC_EXT_05",
        "formal_reason": "Adverse credit history reported by external credit bureaus",
        "category": "External Bureau Credit History",
        "guidance": "Maintaining on-time payment records across all active credit trade lines will raise composite bureau ratings over time.",
    },
    "EXT_SOURCES_MEAN": {
        "code": "RC_EXT_06",
        "formal_reason": "Low composite credit bureau rating across independent credit agencies",
        "category": "External Bureau Credit History",
        "guidance": "Obtaining copies of your credit bureau files to correct reporting inaccuracies will improve score performance.",
    },
    "EXT_SOURCE_1": {
        "code": "RC_EXT_07",
        "formal_reason": "Delinquency or limited history on primary external credit bureau file",
        "category": "External Bureau Credit History",
        "guidance": "Consistently meeting debt payment deadlines enhances your primary bureau assessment.",
    },
    "EXT_SOURCE_2": {
        "code": "RC_EXT_08",
        "formal_reason": "Elevated credit risk score reported by secondary credit reporting agency",
        "category": "External Bureau Credit History",
        "guidance": "Lowering credit utilization across revolving credit cards will positively impact this bureau rating.",
    },
    "EXT_SOURCE_3": {
        "code": "RC_EXT_09",
        "formal_reason": "Adverse credit profile or recent collection events on bureau file 3",
        "category": "External Bureau Credit History",
        "guidance": "Resolving outstanding collection items or delinquent credit inquiries is advised.",
    },
    "EMPLOYED_TO_AGE_RATIO": {
        "code": "RC_EMP_10",
        "formal_reason": "Insufficient length of stable employment tenure relative to career timeline",
        "category": "Employment & Income Stability",
        "guidance": "Continued continuous employment tenure with your current employer strengthens stability scores.",
    },
    "EMPLOYED_YEARS": {
        "code": "RC_EMP_11",
        "formal_reason": "Insufficient duration of continuous employment history",
        "category": "Employment & Income Stability",
        "guidance": "Underwriting criteria require a minimum established track record of verifiable employment.",
    },
    "DAYS_EMPLOYED": {
        "code": "RC_EMP_12",
        "formal_reason": "Short tenure in current employment position",
        "category": "Employment & Income Stability",
        "guidance": "Accumulating at least 12 to 24 months of steady employment in the current role improves credit risk tiering.",
    },
    "AGE_YEARS": {
        "code": "RC_AGE_13",
        "formal_reason": "Limited depth of established credit and financial operating history",
        "category": "Credit Depth & Profile",
        "guidance": "Establishing seasoned revolving trade lines with perfect repayment history develops credit maturity.",
    },
    "DAYS_BIRTH": {
        "code": "RC_AGE_14",
        "formal_reason": "Limited credit operating maturity",
        "category": "Credit Depth & Profile",
        "guidance": "Gradually seasoning open credit accounts helps build an unblemished repayment track record.",
    },
    "AMT_INCOME_TOTAL": {
        "code": "RC_INC_15",
        "formal_reason": "Gross verifiable income insufficient for requested credit facility",
        "category": "Income & Affordability",
        "guidance": "Providing additional verifiable secondary income streams or co-signers may satisfy policy thresholds.",
    },
    "AMT_ANNUITY": {
        "code": "RC_ANN_16",
        "formal_reason": "High monthly annuity obligation relative to disposable cash flow",
        "category": "Debt Burden & Leverage",
        "guidance": "Structuring loan repayments over an extended term lowers the monthly annuity commitment.",
    },
    "AMT_CREDIT": {
        "code": "RC_CRD_17",
        "formal_reason": "Requested principal amount exceeds maximum unsecured credit limit",
        "category": "Credit Facility Terms",
        "guidance": "Requesting a smaller credit limit or providing qualifying collateral enables approval.",
    },
    "DEF_30_CNT_SOCIAL_CIRCLE": {
        "code": "RC_SOC_18",
        "formal_reason": "Delinquency indicators identified within applicant reference network",
        "category": "Peer & Network Risk",
        "guidance": "Maintaining verified primary residence and professional reference stability mitigates network risk factors.",
    },
    "DEF_60_CNT_SOCIAL_CIRCLE": {
        "code": "RC_SOC_19",
        "formal_reason": "Severe payment default indicators identified in reference network",
        "category": "Peer & Network Risk",
        "guidance": "Providing independent verifiable credit guarantees helps overcome external risk markers.",
    },
    "DAYS_LAST_PHONE_CHANGE": {
        "code": "RC_VER_20",
        "formal_reason": "Recent modification of primary contact credentials creates verification risk",
        "category": "Identity & Contact Stability",
        "guidance": "Maintaining long-term consistent contact credentials and phone tenure satisfies anti-fraud safeguards.",
    },
    "AMT_REQ_CREDIT_BUREAU_YEAR": {
        "code": "RC_INQ_21",
        "formal_reason": "Excessive number of credit inquiries submitted to credit bureaus within past 12 months",
        "category": "Inquiry Velocity",
        "guidance": "Ceasing new credit applications for 6 months allows hard bureau inquiries to age off.",
    },
    "OCCUPATION_TYPE": {
        "code": "RC_OCC_22",
        "formal_reason": "Occupational classification does not satisfy standard credit risk criteria",
        "category": "Employment & Income Stability",
        "guidance": "Providing multi-year wage statements (W-2 / tax returns) may provide compensatory underwriting proof.",
    },
    "ORGANIZATION_TYPE": {
        "code": "RC_ORG_23",
        "formal_reason": "Employer business sector exhibits elevated macroeconomic volatility",
        "category": "Employment & Income Stability",
        "guidance": "Documenting tenure stability and career continuity can offset industry-specific volatility.",
    },
    "NAME_EDUCATION_TYPE": {
        "code": "RC_EDU_24",
        "formal_reason": "Educational profile does not meet credit policy benchmark for requested limit",
        "category": "Demographic Profile",
        "guidance": "Supplementing the application with certified vocational or professional licenses can be submitted for review.",
    },
    "NAME_HOUSING_TYPE": {
        "code": "RC_HOU_25",
        "formal_reason": "Residential tenure or living arrangement does not meet housing stability criteria",
        "category": "Residential Stability",
        "guidance": "Establishing longer continuous tenancy at your current residential address improves housing stability scoring.",
    },
    "NAME_INCOME_TYPE": {
        "code": "RC_INC_26",
        "formal_reason": "Income source category does not meet standard underwriting guidelines",
        "category": "Income & Affordability",
        "guidance": "Supplying audited financial statements or verified recurring payroll receipts will assist re-evaluation.",
    },
    "REGION_RATING_CLIENT": {
        "code": "RC_GEO_27",
        "formal_reason": "Geographic regional default index exceeds policy threshold",
        "category": "Regional Risk",
        "guidance": "Providing localized employment verification helps contextualize regional ratings.",
    },
    "CNT_CHILDREN": {
        "code": "RC_DEP_28",
        "formal_reason": "High number of financial dependents relative to declared disposable income",
        "category": "Affordability & Dependents",
        "guidance": "Documenting aggregate household income or joint spousal earnings improves disposable surplus calculations.",
    },
}


def _map_feature_to_reason(feature_name: str) -> Dict[str, str]:
    """
    Maps an individual feature or interaction term to a formal regulatory adverse reason.
    Handles EBM interaction terms such as 'EXT_SOURCES_WEIGHTED x PAYMENT_RATE'.
    """
    clean_feat = feature_name.strip()

    # Exact match in registry
    if clean_feat in REASON_CODE_REGISTRY:
        return REASON_CODE_REGISTRY[clean_feat]

    # Check for EBM interaction terms (e.g. "FEAT_A x FEAT_B")
    if " x " in clean_feat:
        parts = [p.strip() for p in clean_feat.split(" x ")]
        reasons = [REASON_CODE_REGISTRY.get(p) for p in parts if p in REASON_CODE_REGISTRY]
        if reasons:
            main_r = reasons[0]
            sec_r = reasons[1] if len(reasons) > 1 else main_r
            return {
                "code": f"RC_INT_{main_r['code'][-2:]}_{sec_r['code'][-2:]}",
                "formal_reason": f"Compounded risk: {main_r['formal_reason']} combined with {sec_r['formal_reason'].lower()}",
                "category": "Compound Interaction Risk",
                "guidance": f"{main_r['guidance']} Additionally, {sec_r['guidance'].lower()}",
            }

    # Partial substring matches for common prefixes
    for key, val in REASON_CODE_REGISTRY.items():
        if key in clean_feat:
            return val

    # Default fallback
    readable_name = clean_feat.replace("_", " ").title()
    return {
        "code": f"RC_GEN_{abs(hash(clean_feat)) % 100:02d}",
        "formal_reason": f"Underwriting threshold not satisfied for {readable_name}",
        "category": "General Credit Policy Criteria",
        "guidance": "Contact your credit representative for detailed underwriting review options.",
    }


def format_feature_value(feature_name: str, value: Any) -> str:
    """Formats numeric and categorical values for business presentation."""
    if value is None or value == "Missing" or pd.isna(value):
        return "Not Disclosed / Unavailable"

    try:
        fval = float(value)
        if "RATE" in feature_name or "DTI" in feature_name or "RATIO" in feature_name:
            return f"{fval * 100:.2f}%"
        elif "EXT_SOURCE" in feature_name:
            return f"{fval:.4f}"
        elif "AMT_" in feature_name:
            return f"${fval:,.2f}"
        elif "YEARS" in feature_name:
            return f"{fval:.1f} yrs"
        elif "DAYS_" in feature_name:
            return f"{abs(fval):,.0f} days"
        else:
            return f"{fval:.2f}"
    except (ValueError, TypeError):
        return str(value).replace("_", " ").title()


def extract_adverse_drivers(
    applicant_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    explainer: Optional[CreditRiskExplainer] = None,
    method: str = "ebm",
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Extracts the top 3-4 adverse factors (risk drivers) from EBM or TreeSHAP explanations.

    Args:
        applicant_data: dictionary or series of applicant inputs
        explainer: instance of CreditRiskExplainer (or default if None)
        method: 'ebm' (exact log-odds scores), 'shap' (TreeSHAP attributions), or 'ensemble'
        top_k: number of adverse factors to retrieve (standard FCRA/ECOA is 4)

    Returns:
        List of dictionaries with formal reason code, descriptive text, applicant value, and contribution.
    """
    exp = explainer or _get_default_explainer()

    if method.lower() == "shap":
        shap_res = exp.get_shap_local_waterfall(applicant_data)
        raw_drivers = [
            {"feature": c["feature"], "value": c["value"], "contribution": c["attribution"]}
            for c in shap_res["top_adverse_drivers"]
        ]
    elif method.lower() == "ensemble":
        ebm_res = exp.get_ebm_local_explanation(applicant_data)
        shap_res = exp.get_shap_local_waterfall(applicant_data)

        # Merge unique top drivers
        seen_feats = set()
        merged = []
        for c in ebm_res["top_risk_drivers"] + shap_res["top_adverse_drivers"]:
            feat = c["feature"]
            if feat not in seen_feats:
                seen_feats.add(feat)
                merged.append({
                    "feature": feat,
                    "value": c["value"],
                    "contribution": c.get("score", c.get("attribution", 0.0)),
                })
        raw_drivers = merged
    else:
        # Default: EBM Exact GA2M Glassbox
        ebm_res = exp.get_ebm_local_explanation(applicant_data)
        raw_drivers = [
            {"feature": c["feature"], "value": c["value"], "contribution": c["score"]}
            for c in ebm_res["top_risk_drivers"]
        ]

    # Map each feature to regulatory reasons and eliminate duplicate reason codes
    adverse_reasons = []
    seen_codes = set()

    for item in raw_drivers:
        feat = item["feature"]
        reason_meta = _map_feature_to_reason(feat)
        code = reason_meta["code"]

        if code in seen_codes:
            continue
        seen_codes.add(code)

        adverse_reasons.append({
            "feature": feat,
            "reason_code": code,
            "formal_reason": reason_meta["formal_reason"],
            "category": reason_meta["category"],
            "guidance": reason_meta["guidance"],
            "applicant_value": format_feature_value(feat, item["value"]),
            "contribution": round(float(item["contribution"]), 4),
        })

        if len(adverse_reasons) >= top_k:
            break

    # If fewer than top_k were extracted (e.g. highly prime borrower), add default credit depth reasons
    if len(adverse_reasons) < min(3, top_k):
        fallbacks = ["EXT_SOURCES_WEIGHTED", "PAYMENT_RATE", "DTI", "EMPLOYED_TO_AGE_RATIO"]
        for fb in fallbacks:
            meta = REASON_CODE_REGISTRY[fb]
            if meta["code"] not in seen_codes:
                seen_codes.add(meta["code"])
                adverse_reasons.append({
                    "feature": fb,
                    "reason_code": meta["code"],
                    "formal_reason": meta["formal_reason"],
                    "category": meta["category"],
                    "guidance": meta["guidance"],
                    "applicant_value": "Benchmark threshold not reached",
                    "contribution": 0.0,
                })
            if len(adverse_reasons) >= top_k:
                break

    return adverse_reasons[:top_k]


def generate_adverse_action_notice(
    applicant_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    applicant_name: str = "Valued Applicant",
    application_id: Optional[Union[str, int]] = None,
    decision_date: Optional[str] = None,
    decision: str = "DECLINED",
    risk_score: Optional[float] = None,
    explainer: Optional[CreditRiskExplainer] = None,
    method: str = "ebm",
    creditor_name: str = "NeoStats Credit Risk Intelligence Lending Facility",
) -> str:
    """
    Generates a formal, regulatory-ready Adverse Action Notice formatted in Markdown,
    compliant with FCRA § 615(a) and ECOA Regulation B (12 C.F.R. § 1002.9).

    Args:
        applicant_data: Dict or Series of applicant application parameters
        applicant_name: Name of credit applicant
        application_id: Loan application ID / SK_ID_CURR
        decision_date: Date string (defaults to current date)
        decision: Credit decision ('DECLINED', 'MANUAL_REVIEW_REQUIRED', 'UNFAVORABLE_TERMS')
        risk_score: Probability of default or calibrated risk score
        explainer: Optional CreditRiskExplainer instance
        method: 'ebm' or 'shap'
        creditor_name: Institution name

    Returns:
        Formatted Markdown text of the Adverse Action Notice.
    """
    if isinstance(applicant_data, (pd.Series, pd.DataFrame)):
        data_dict = applicant_data.iloc[0].to_dict() if isinstance(applicant_data, pd.DataFrame) else applicant_data.to_dict()
    else:
        data_dict = dict(applicant_data)

    app_id = application_id or data_dict.get("SK_ID_CURR", "APP-2026-0907-882")
    notice_date = decision_date or datetime.date.today().strftime("%B %d, %Y")

    # Extract Adverse Factors
    adverse_factors = extract_adverse_drivers(data_dict, explainer=explainer, method=method, top_k=4)

    # Compute Risk Score if not provided
    if risk_score is None:
        exp = explainer or _get_default_explainer()
        ebm_res = exp.get_ebm_local_explanation(data_dict)
        default_prob = ebm_res["predicted_probability"]
    else:
        default_prob = risk_score

    # Convert probability to credit score scale (300 to 850 inverted)
    # Higher default prob -> lower credit score
    calculated_credit_score = int(np.clip(850 - (default_prob * 550), 300, 850))

    # Format Principal Reasons in Markdown
    reasons_markdown = ""
    for idx, item in enumerate(adverse_factors, 1):
        reasons_markdown += (
            f"{idx}. **{item['formal_reason']}** (Reason Code: `{item['reason_code']}`)\n"
            f"   - *Underwriting Metric*: `{item['feature']}` = **{item['applicant_value']}**\n"
            f"   - *Impact on Risk*: Shifts default log-odds by `{'+' if item['contribution'] > 0 else ''}{item['contribution']:.4f}`\n"
            f"   - *Consumer Guidance*: {item['guidance']}\n\n"
        )

    notice_template = f"""# STATEMENT OF ADVERSE ACTION & CREDIT EVALUATION NOTICE
**Notice of Credit Denial, Termination, or Change**  
*Pursuant to the Equal Credit Opportunity Act (Regulation B, 12 C.F.R. § 1002.9) and the Fair Credit Reporting Act (15 U.S.C. § 1681m)*

---

### Transaction & Applicant Details
- **Date of Decision**: {notice_date}
- **Application Reference Number**: `{app_id}`
- **Applicant Name**: {applicant_name}
- **Creditor / Lender**: {creditor_name}
- **Action Taken**: **{decision.upper()}**

---

### Part I: Principal Reasons for Credit Decision
Our credit underwriting policies utilize a dual clearbox/blackbox automated scoring system that assesses multiple financial, employment, and credit bureau indicators. Your application was not approved in accordance with established credit risk standards for the following **principal reasons**:

{reasons_markdown}
---

### Part II: Credit Score Disclosure
In evaluating your application, the creditor utilized a calibrated internal risk assessment model:

- **Credit Model Score**: **{calculated_credit_score}** (Score Range: 300 to 850, where higher scores represent lower credit risk)
- **Model Assessed Probability of Default**: **{default_prob:.2%}**
- **Date Score Calculated**: {notice_date}
- **Key Factors Adversely Affecting Score**: The primary factors contributing negatively to your score are the principal reasons enumerated in Part I above.

---

### Part III: Consumer Reporting Agency (CRA) Information
Although our credit decision was based in part on information obtained from external consumer reporting agencies, **the consumer reporting agencies played no part in our decision and are unable to supply specific reasons why credit was denied**.

Under the Fair Credit Reporting Act, you have the right to:
1. **Free Consumer Report**: Obtain a free copy of your credit report from the consumer reporting agency if requested within **60 days** of receiving this notice.
2. **Right to Dispute**: Dispute the accuracy or completeness of any information in the report directly with the consumer reporting agency.

**Major Consumer Reporting Agencies:**
- **Equifax Information Services LLC**  
  P.O. Box 740241, Atlanta, GA 30374-0241 | Toll-free: (800) 685-1111 | [www.equifax.com](https://www.equifax.com)
- **Experian National Consumer Assistance**  
  P.O. Box 4500, Allen, TX 75013 | Toll-free: (888) 397-3742 | [www.experian.com](https://www.experian.com)
- **TransUnion LLC Consumer Dispute Center**  
  P.O. Box 2000, Chester, PA 19016 | Toll-free: (800) 916-8800 | [www.transunion.com](https://www.transunion.com)

---

### Part IV: Federal Equal Credit Opportunity Act (ECOA) Notice
The **Federal Equal Credit Opportunity Act** prohibits creditors from discriminating against credit applicants on the basis of race, color, religion, national origin, sex, marital status, age (provided the applicant has the capacity to enter into a binding contract); because all or part of the applicant's income derives from any public assistance program; or because the applicant has in good faith exercised any right under the Consumer Credit Protection Act.

The federal agency that administers compliance with this law concerning this creditor is:
> **Consumer Financial Protection Bureau (CFPB)**  
> 1700 G Street NW, Washington, DC 20552  
> Telephone: (855) 411-2372 | [www.consumerfinance.gov](https://www.consumerfinance.gov)

---
*This document constitutes a formal regulatory Adverse Action Notice generated by the NeoStats Credit Risk Intelligence Platform.*
"""
    return notice_template
