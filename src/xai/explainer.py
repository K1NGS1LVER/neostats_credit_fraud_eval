"""
NeoStats Credit Risk Intelligence Platform - Explainable AI (XAI) Engine
Provides dual explainability for credit risk decisions:
1. Glassbox Explainable Boosting Machine (EBM / GA2M): Exact additive feature contributions
   to default log-odds with zero approximation error.
2. Blackbox LightGBM + TreeSHAP: Local feature attributions, waterfall data, and global SHAP importance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def _find_artifacts_dir() -> Path:
    """Locate the artifacts directory across various execution roots."""
    candidates = [
        Path("artifacts").resolve(),
        Path(__file__).resolve().parents[2] / "artifacts",
        Path(__file__).resolve().parents[1] / "artifacts",
        Path("..") / "artifacts",
    ]
    for p in candidates:
        if p.exists() and (p / "ebm_model.joblib").exists():
            return p.resolve()
    # Fallback to standard project artifacts dir
    return (Path(__file__).resolve().parents[2] / "artifacts").resolve()


def engineer_applicant_features(data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Ensure all banking domain features (leverage ratios, composites) are present.
    Computes missing engineered ratios from raw loan application fields.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = pd.DataFrame([data.to_dict()])
    else:
        df = data.copy()

    # 1. PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
    if "PAYMENT_RATE" not in df.columns or df["PAYMENT_RATE"].isnull().all():
        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / np.maximum(df["AMT_CREDIT"], 1.0)
        else:
            df["PAYMENT_RATE"] = 0.05

    # 2. DTI = AMT_ANNUITY / AMT_INCOME_TOTAL
    if "DTI" not in df.columns or df["DTI"].isnull().all():
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["DTI"] = df["AMT_ANNUITY"] / np.maximum(df["AMT_INCOME_TOTAL"], 1.0)
        else:
            df["DTI"] = 0.15

    # 3. CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
    if "CREDIT_INCOME_RATIO" not in df.columns or df["CREDIT_INCOME_RATIO"].isnull().all():
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / np.maximum(df["AMT_INCOME_TOTAL"], 1.0)
        else:
            df["CREDIT_INCOME_RATIO"] = 3.0

    # 4. LTV = AMT_CREDIT / AMT_GOODS_PRICE
    if "LTV" not in df.columns or df["LTV"].isnull().all():
        if "AMT_CREDIT" in df.columns:
            goods = df["AMT_GOODS_PRICE"] if "AMT_GOODS_PRICE" in df.columns else df["AMT_CREDIT"]
            goods_safe = np.maximum(goods.fillna(1000.0) if hasattr(goods, "fillna") else 1000.0, 1000.0)
            df["LTV"] = df["AMT_CREDIT"] / goods_safe
        else:
            df["LTV"] = 0.8

    # 5. AGE_YEARS
    if "AGE_YEARS" not in df.columns or df["AGE_YEARS"].isnull().all():
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(2)
        else:
            df["AGE_YEARS"] = 40.0

    # 6. EMPLOYED_YEARS
    if "EMPLOYED_YEARS" not in df.columns or df["EMPLOYED_YEARS"].isnull().all():
        if "DAYS_EMPLOYED" in df.columns:
            days_emp = df["DAYS_EMPLOYED"]
            df["EMPLOYED_YEARS"] = np.where(days_emp == 365243, 0.0, -days_emp / 365.25).round(2)
        else:
            df["EMPLOYED_YEARS"] = 5.0

    # 7. EMPLOYED_TO_AGE_RATIO
    if "EMPLOYED_TO_AGE_RATIO" not in df.columns or df["EMPLOYED_TO_AGE_RATIO"].isnull().all():
        df["EMPLOYED_TO_AGE_RATIO"] = df["EMPLOYED_YEARS"] / np.maximum(df["AGE_YEARS"], 18.0)

    # 8. External Credit Bureau Composites
    e1 = df["EXT_SOURCE_1"] if "EXT_SOURCE_1" in df.columns else pd.Series(0.5, index=df.index)
    e2 = df["EXT_SOURCE_2"] if "EXT_SOURCE_2" in df.columns else pd.Series(0.5, index=df.index)
    e3 = df["EXT_SOURCE_3"] if "EXT_SOURCE_3" in df.columns else pd.Series(0.5, index=df.index)

    e1_val = e1.fillna(0.5) if hasattr(e1, "fillna") else 0.5
    e2_val = e2.fillna(0.5) if hasattr(e2, "fillna") else 0.5
    e3_val = e3.fillna(0.5) if hasattr(e3, "fillna") else 0.5

    if "EXT_SOURCES_MEAN" not in df.columns or df["EXT_SOURCES_MEAN"].isnull().all():
        df["EXT_SOURCES_MEAN"] = (e1_val + e2_val + e3_val) / 3.0

    if "EXT_SOURCES_WEIGHTED" not in df.columns or df["EXT_SOURCES_WEIGHTED"].isnull().all():
        df["EXT_SOURCES_WEIGHTED"] = (e1_val + 2.0 * e2_val + 3.0 * e3_val) / 6.0

    return df


class CreditRiskExplainer:
    """
    Dual Explainability Engine for Credit Risk Intelligence.
    Integrates Microsoft InterpretML EBM (Glassbox) and LightGBM TreeSHAP (Blackbox).
    """

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None):
        self.artifacts_dir = Path(artifacts_dir).resolve() if artifacts_dir else _find_artifacts_dir()
        self._load_artifacts()
        self._tree_explainer = None

    def _load_artifacts(self) -> None:
        """Load trained models, transformers, and baseline background dataset."""
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory '{self.artifacts_dir}' does not exist. "
                "Please execute notebooks/credit_risk_eda_modeling.ipynb to build model artifacts."
            )

        required_files = {
            "ebm_model": "ebm_model.joblib",
            "lgbm_model": "lgbm_model.joblib",
            "base_lgbm_model": "base_lgbm_model.joblib",
            "cat_encoder": "cat_encoder.joblib",
            "shap_background": "shap_background.parquet",
        }

        missing = [fn for k, fn in required_files.items() if not (self.artifacts_dir / fn).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required artifacts in '{self.artifacts_dir}': {missing}. "
                "Please run the model training pipeline first."
            )

        logger.info(f"Loading credit risk models from {self.artifacts_dir}")
        self.ebm_model = joblib.load(self.artifacts_dir / "ebm_model.joblib")
        self.lgbm_model = joblib.load(self.artifacts_dir / "lgbm_model.joblib")
        self.base_lgbm_model = joblib.load(self.artifacts_dir / "base_lgbm_model.joblib")
        self.cat_encoder = joblib.load(self.artifacts_dir / "cat_encoder.joblib")
        self.shap_background = pd.read_parquet(self.artifacts_dir / "shap_background.parquet")

        # Optional metadata
        feature_cols_path = self.artifacts_dir / "feature_cols.joblib"
        cat_cols_path = self.artifacts_dir / "cat_cols.joblib"
        num_cols_path = self.artifacts_dir / "num_cols.joblib"

        if feature_cols_path.exists():
            self.feature_cols = joblib.load(feature_cols_path)
        else:
            self.feature_cols = self.shap_background.columns.tolist()

        if cat_cols_path.exists():
            self.cat_cols = joblib.load(cat_cols_path)
        else:
            self.cat_cols = [
                c for c in self.feature_cols
                if self.feature_cols and ("TYPE" in c or "FLAG" in c or "GENDER" in c or "STATUS" in c)
            ]

        if num_cols_path.exists():
            self.num_cols = joblib.load(num_cols_path)
        else:
            self.num_cols = [c for c in self.feature_cols if c not in self.cat_cols]

    @property
    def tree_explainer(self) -> shap.TreeExplainer:
        """Cached SHAP TreeExplainer initialized on the base LightGBM model."""
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self.base_lgbm_model)
        return self._tree_explainer

    def _prepare_inputs(
        self, applicant_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares standardized inputs for both EBM and LightGBM models.
        Returns:
            (ebm_df, lgb_df)
        """
        df_feat = engineer_applicant_features(applicant_data)

        # Align to expected feature set
        aligned_df = pd.DataFrame(index=df_feat.index)
        for col in self.feature_cols:
            if col in df_feat.columns:
                aligned_df[col] = df_feat[col]
            else:
                # Median/default fallback for missing columns
                if col in self.shap_background.columns:
                    aligned_df[col] = self.shap_background[col].median()
                else:
                    aligned_df[col] = 0.0

        # EBM input: categorical columns as string with 'Missing' imputation
        ebm_df = aligned_df.copy()
        for c in self.cat_cols:
            if c in ebm_df.columns:
                ebm_df[c] = ebm_df[c].fillna("Missing").astype(str)

        for c in self.num_cols:
            if c in ebm_df.columns:
                ebm_df[c] = pd.to_numeric(ebm_df[c], errors="coerce").fillna(0.0)

        # LightGBM input: categorical columns encoded via OrdinalEncoder
        lgb_df = ebm_df.copy()
        try:
            lgb_df[self.cat_cols] = self.cat_encoder.transform(ebm_df[self.cat_cols])
        except Exception as err:
            logger.warning(f"Error transforming categoricals with cat_encoder: {err}. Imputing 0.0")
            for c in self.cat_cols:
                lgb_df[c] = 0.0

        return ebm_df, lgb_df

    def get_ebm_local_explanation(
        self, applicant_dict: Union[Dict[str, Any], pd.Series, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Extracts exact additive feature contributions (scores) from the Glassbox EBM model (GA^2M).
        Every feature contribution directly shifts the log-odds of default with ZERO approximation error.

        Returns:
            dict containing:
            - base_score (intercept in log-odds)
            - total_log_odds (sum of intercept and all feature/interaction scores)
            - predicted_probability (logistic sigmoid of total log-odds)
            - contributions: list of sorted feature attributions with direction and value
            - top_risk_drivers: features with largest positive contribution (increasing default probability)
            - top_protective_factors: features with largest negative contribution (reducing default probability)
        """
        ebm_df, _ = self._prepare_inputs(applicant_dict)
        single_row = ebm_df.iloc[[0]]

        explanation = self.ebm_model.explain_local(single_row)
        data = explanation.data(0)

        # Extract intercept and scores
        extra = data.get("extra", {})
        intercept = float(extra["scores"][0]) if "scores" in extra and len(extra["scores"]) > 0 else 0.0

        names = data.get("names", [])
        scores = [float(s) for s in data.get("scores", [])]
        values = data.get("values", [])

        contributions = []
        total_scores = sum(scores)
        total_log_odds = intercept + total_scores
        # Logistic link
        predicted_prob = float(1.0 / (1.0 + np.exp(-np.clip(total_log_odds, -25.0, 25.0))))

        for name, score, val in zip(names, scores, values):
            contributions.append({
                "feature": str(name),
                "value": val,
                "score": round(score, 6),
                "abs_score": round(abs(score), 6),
                "direction": "increases_risk" if score > 0 else ("decreases_risk" if score < 0 else "neutral"),
                "log_odds_impact": f"{'+' if score > 0 else ''}{score:.4f}",
            })

        # Sort contributions by absolute score magnitude
        contributions.sort(key=lambda x: x["abs_score"], reverse=True)

        top_risk_drivers = [c for c in contributions if c["score"] > 0][:5]
        top_protective_factors = [c for c in contributions if c["score"] < 0][:5]

        return {
            "explanation_type": "EBM_Glassbox_Exact",
            "model_architecture": "ExplainableBoostingClassifier (GA^2M)",
            "approximation_error": 0.0,
            "base_score": round(intercept, 6),
            "total_log_odds": round(total_log_odds, 6),
            "predicted_probability": round(predicted_prob, 4),
            "contributions": contributions,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "num_terms_evaluated": len(contributions),
        }

    def get_shap_local_waterfall(
        self, applicant_dict: Union[Dict[str, Any], pd.Series, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Computes TreeSHAP local feature attributions on LightGBM to construct waterfall diagnostic data.

        Returns:
            dict containing:
            - base_value: expected model margin (log-odds base value)
            - prediction_margin: base_value + sum(shap_values)
            - predicted_probability: calibrated default probability
            - waterfall: sequential cumulative step data for waterfall plotting
            - contributions: list of feature attributions ranked by absolute SHAP value
            - top_adverse_drivers: top positive SHAP values (increasing default risk)
            - top_favorable_drivers: top negative SHAP values (decreasing default risk)
        """
        ebm_df, lgb_df = self._prepare_inputs(applicant_dict)
        single_row = lgb_df.iloc[[0]]

        shap_vals_obj = self.tree_explainer(single_row)
        shap_values = shap_vals_obj.values[0]
        base_value = float(shap_vals_obj.base_values[0])

        feature_names = self.feature_cols
        # Raw or original values for human readability
        orig_values = [ebm_df.iloc[0][col] for col in feature_names]

        # Model predicted probability
        try:
            pred_prob = float(self.lgbm_model.predict_proba(single_row)[:, 1][0])
        except Exception:
            pred_prob = float(1.0 / (1.0 + np.exp(-np.clip(base_value + float(np.sum(shap_values)), -25.0, 25.0))))

        # Build feature attribution list
        contributions = []
        for feat, val, attr in zip(feature_names, orig_values, shap_values):
            contributions.append({
                "feature": str(feat),
                "value": val,
                "attribution": round(float(attr), 6),
                "abs_attribution": round(float(abs(attr)), 6),
                "direction": "increases_risk" if attr > 0 else ("decreases_risk" if attr < 0 else "neutral"),
            })

        # Rank by absolute attribution magnitude
        contributions.sort(key=lambda x: x["abs_attribution"], reverse=True)

        # Build waterfall sequence
        waterfall_steps = []
        running_total = base_value
        for item in contributions:
            running_total += item["attribution"]
            waterfall_steps.append({
                "feature": item["feature"],
                "value": item["value"],
                "attribution": item["attribution"],
                "cumulative": round(running_total, 6),
                "direction": item["direction"],
            })

        top_adverse = [c for c in contributions if c["attribution"] > 0][:5]
        top_favorable = [c for c in contributions if c["attribution"] < 0][:5]

        return {
            "explanation_type": "TreeSHAP_LightGBM",
            "model_architecture": "LightGBM + TreeSHAP",
            "base_value": round(base_value, 6),
            "prediction_margin": round(base_value + float(np.sum(shap_values)), 6),
            "predicted_probability": round(pred_prob, 4),
            "waterfall": waterfall_steps,
            "contributions": contributions,
            "top_adverse_drivers": top_adverse,
            "top_favorable_drivers": top_favorable,
        }

    def get_global_importance(self, top_n: int = 15) -> Dict[str, Any]:
        """
        Returns top features ranked by:
        1. EBM global term importance (exact mean absolute score contribution)
        2. LightGBM global TreeSHAP importance (mean absolute SHAP across background population)
        3. Combined consensus ranking
        """
        # 1. EBM Global Importance
        ebm_global = self.ebm_model.explain_global(name="EBM Global Explanations")
        ebm_data = ebm_global.data()
        ebm_names = ebm_data.get("names", [])
        ebm_scores = [float(s) for s in ebm_data.get("scores", [])]

        ebm_ranked = sorted(
            [{"feature": n, "ebm_score": round(s, 6)} for n, s in zip(ebm_names, ebm_scores)],
            key=lambda x: x["ebm_score"],
            reverse=True,
        )

        # 2. TreeSHAP Global Importance over background dataset
        shap_vals_bg = self.tree_explainer(self.shap_background)
        mean_abs_shap = np.abs(shap_vals_bg.values).mean(axis=0)

        shap_ranked = sorted(
            [
                {"feature": col, "mean_abs_shap": round(float(mas), 6)}
                for col, mas in zip(self.shap_background.columns, mean_abs_shap)
            ],
            key=lambda x: x["mean_abs_shap"],
            reverse=True,
        )

        # 3. Consensus ranking mapping
        ebm_rank_map = {item["feature"]: idx + 1 for idx, item in enumerate(ebm_ranked)}
        shap_rank_map = {item["feature"]: idx + 1 for idx, item in enumerate(shap_ranked)}

        all_features = set(ebm_rank_map.keys()).union(set(shap_rank_map.keys()))
        combined = []
        for feat in all_features:
            e_rank = ebm_rank_map.get(feat, 999)
            s_rank = shap_rank_map.get(feat, 999)
            e_score = next((x["ebm_score"] for x in ebm_ranked if x["feature"] == feat), 0.0)
            s_score = next((x["mean_abs_shap"] for x in shap_ranked if x["feature"] == feat), 0.0)
            combined.append({
                "feature": feat,
                "ebm_score": e_score,
                "shap_mean_abs": s_score,
                "ebm_rank": e_rank,
                "shap_rank": s_rank,
                "average_rank": round((e_rank + s_rank) / 2.0, 1),
            })

        combined.sort(key=lambda x: x["average_rank"])

        return {
            "top_ebm_importance": ebm_ranked[:top_n],
            "top_shap_importance": shap_ranked[:top_n],
            "consensus_ranking": combined[:top_n],
        }


# Singleton instance for module-level convenience functions
_DEFAULT_EXPLAINER: Optional[CreditRiskExplainer] = None


def _get_default_explainer() -> CreditRiskExplainer:
    global _DEFAULT_EXPLAINER
    if _DEFAULT_EXPLAINER is None:
        _DEFAULT_EXPLAINER = CreditRiskExplainer()
    return _DEFAULT_EXPLAINER


def get_ebm_local_explanation(applicant_dict: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """Module-level function to extract exact EBM feature contributions."""
    return _get_default_explainer().get_ebm_local_explanation(applicant_dict)


def get_shap_local_waterfall(applicant_dict: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """Module-level function to extract TreeSHAP local waterfall attributions."""
    return _get_default_explainer().get_shap_local_waterfall(applicant_dict)


def get_global_importance(top_n: int = 15) -> Dict[str, Any]:
    """Module-level function to retrieve top global feature importance from EBM and SHAP."""
    return _get_default_explainer().get_global_importance(top_n=top_n)
