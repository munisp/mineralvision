"""
Explainable AI (XAI) module for MineralVision.

Provides model interpretability and explainability capabilities.
"""

from .explainability import (
    ExplanationType,
    AggregationType,
    PlotType,
    FeatureContribution,
    LocalExplanation,
    GlobalExplanation,
    PartialDependence,
    Counterfactual,
    Explainer,
    SHAPExplainer,
    LIMEExplainer,
    FeatureImportanceExplainer,
    PartialDependenceCalculator,
    CounterfactualGenerator,
    ExplanationReport,
    XAIService,
    create_xai_service,
    create_shap_explainer,
    create_lime_explainer,
)

__all__ = [
    'ExplanationType',
    'AggregationType',
    'PlotType',
    'FeatureContribution',
    'LocalExplanation',
    'GlobalExplanation',
    'PartialDependence',
    'Counterfactual',
    'Explainer',
    'SHAPExplainer',
    'LIMEExplainer',
    'FeatureImportanceExplainer',
    'PartialDependenceCalculator',
    'CounterfactualGenerator',
    'ExplanationReport',
    'XAIService',
    'create_xai_service',
    'create_shap_explainer',
    'create_lime_explainer',
]
