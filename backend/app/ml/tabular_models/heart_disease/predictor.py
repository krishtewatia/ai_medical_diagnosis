import time
from typing import Any, Dict
import pandas as pd

from app.ml.base_predictor import BaseTabularPredictor
from app.ml.loaders import load_model_for_disease
from app.schemas.disease_config import DiseaseConfig
from app.schemas.prediction import PredictionResult


class HeartDiseasePredictor(BaseTabularPredictor):
    """
    Production Predictor for Heart Disease Risk Assessment.
    Wraps the exported scikit-learn Pipeline (SimpleImputer + XGBClassifier)
    and executes inference across the 13 clinical features with the stored 0.40 decision threshold.
    """

    FEATURE_ALIASES = {
        "cp": "chest_pain_type",
        "trestbps": "resting_bp",
        "chol": "cholestoral",
        "fbs": "fasting_blood_sugar",
        "thalach": "max_hr",
        "ca": "num_major_vessels",
    }

    def __init__(self, config: DiseaseConfig):
        super().__init__(config)
        self.feature_names = [f.name for f in config.tabular_features] if config.tabular_features else [
            "age",
            "sex",
            "chest_pain_type",
            "resting_bp",
            "cholestoral",
            "fasting_blood_sugar",
            "restecg",
            "max_hr",
            "exang",
            "oldpeak",
            "slope",
            "num_major_vessels",
            "thal"
        ]

    def load_model(self) -> Any:
        """Loads the heart_disease_model.pkl artifact using the centralized loader."""
        return load_model_for_disease(self.config)

    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Normalizes aliases and structures the validated 13 inputs into an ordered single-row DataFrame.
        """
        normalized_inputs = {}
        for k, v in features.items():
            canonical_name = self.FEATURE_ALIASES.get(k, k)
            normalized_inputs[canonical_name] = v

        self.validate_feature_names(normalized_inputs)
        ordered_data = {feat: [normalized_inputs[feat]] for feat in self.feature_names}
        return pd.DataFrame(ordered_data)

    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Executes end-to-end heart disease prediction:
        13 Input Features -> Ordered DataFrame -> XGBoost Pipeline -> 0.40 Threshold -> PredictionResult.
        """
        start_time = time.perf_counter()

        df = self.preprocess_features(input_data)
        model = self.get_model()

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            prob = float(probs[0, 1])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(df)[0])
            prob = float(1.0 / (1.0 + (2.718281828459045 ** (-score))))
        else:
            pred = int(model.predict(df)[0])
            prob = 1.0 if pred == 1 else 0.0

        is_positive, label = self.apply_threshold(prob)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return PredictionResult(
            disease_id=self.disease_id,
            disease_display_name=self.config.display_name,
            model_version=self.config.version,
            model_type=self.config.model_type,
            prediction_label=label,
            is_positive=is_positive,
            probability=round(prob, 4),
            decision_threshold=self.config.decision_threshold,
            metadata={
                "latency_ms": latency_ms,
                "features_evaluated": len(self.feature_names),
                "framework": self.config.framework,
            }
        )
