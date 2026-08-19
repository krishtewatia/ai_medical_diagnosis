import time
from typing import Any, Dict
import pandas as pd

from app.ml.base_predictor import BaseTabularPredictor
from app.ml.loaders import load_model_for_disease
from app.schemas.disease_config import DiseaseConfig
from app.schemas.prediction import PredictionResult


class DiabetesPredictor(BaseTabularPredictor):
    """
    Production Predictor for Diabetes Risk Assessment.
    Wraps the exported scikit-learn Pipeline (SimpleImputer + StandardScaler + LogisticRegression)
    and executes inference with the stored 0.40 decision threshold.
    """

    def __init__(self, config: DiseaseConfig):
        super().__init__(config)
        self.feature_names = [f.name for f in config.tabular_features] if config.tabular_features else [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

    def load_model(self) -> Any:
        """Loads the diabetes_model_v1.joblib artifact using the centralized loader."""
        return load_model_for_disease(self.config)

    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Structures the validated input feature dictionary into an ordered single-row DataFrame.
        Feature scaling and imputation are handled internally by the serialized pipeline.
        """
        self.validate_feature_names(features)
        ordered_data = {feat: [features[feat]] for feat in self.feature_names}
        return pd.DataFrame(ordered_data)

    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Executes end-to-end diabetes prediction:
        Input Dict -> Ordered DataFrame -> Pipeline Inference -> 0.40 Threshold -> PredictionResult.
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
