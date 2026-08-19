import io
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from app.ml.base_predictor import (
    BaseImagePredictor,
    BasePredictor,
    BaseTabularPredictor,
)
from app.ml.loaders import load_model_for_disease
from app.schemas.disease_config import DiseaseCategory, DiseaseConfig
from app.schemas.prediction import PredictionResult


class InferenceError(RuntimeError):
    """Raised when an error occurs during model execution/inference."""
    pass


class InputValidationError(ValueError):
    """Raised when input features or image payloads fail validation."""
    pass


class GenericTabularPredictor(BaseTabularPredictor):
    """
    Generic predictor for structured tabular ML models (e.g. Diabetes, Heart Disease).
    Orchestrates: Feature Validation -> DataFrame Structuring -> Model Inference -> Thresholding -> PredictionResult.
    """

    def load_model(self) -> Any:
        """Loads model artifact from disk via loaders layer."""
        return load_model_for_disease(self.config)

    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates feature keys and transforms feature dictionary into an ordered pandas DataFrame.
        """
        self.validate_feature_names(features)

        if not self.config.tabular_features:
            return pd.DataFrame([features])

        # Maintain exact feature ordering as specified in config
        ordered_data: Dict[str, List[Any]] = {}
        for feature_spec in self.config.tabular_features:
            default_val = getattr(feature_spec, "default_value", None)
            val = features.get(feature_spec.name, default_val)
            if val is None and feature_spec.required:
                raise InputValidationError(
                    f"Required feature '{feature_spec.name}' is missing for disease '{self.disease_id}'."
                )
            
            # Cast numeric types if defined
            if feature_spec.data_type in ["float", "numeric"] and val is not None:
                try:
                    val = float(val)
                except (ValueError, TypeError) as e:
                    raise InputValidationError(
                        f"Feature '{feature_spec.name}' must be float, got '{val}'."
                    ) from e
            elif feature_spec.data_type == "int" and val is not None:
                try:
                    val = int(val)
                except (ValueError, TypeError) as e:
                    raise InputValidationError(
                        f"Feature '{feature_spec.name}' must be int, got '{val}'."
                    ) from e

            ordered_data[feature_spec.name] = [val]

        return pd.DataFrame(ordered_data)

    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Executes inference for tabular input data.
        """
        start_time = time.perf_counter()

        try:
            df = self.preprocess_features(input_data)
            model = self.get_model()

            probability: Optional[float] = None

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)
                # Binary classification: positive class is index 1
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    probability = float(proba[0][1])
                elif proba.ndim == 1:
                    probability = float(proba[0])
                else:
                    probability = float(proba[0][0])
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(df)[0])
                # Sigmoid scaling for decision function score
                probability = float(1.0 / (1.0 + np.exp(-score)))
            elif hasattr(model, "predict"):
                raw_pred = model.predict(df)[0]
                probability = float(raw_pred)
            else:
                raise InferenceError(f"Model for '{self.disease_id}' does not implement predict or predict_proba.")

            # Apply decision threshold
            if probability is not None:
                is_positive, label = self.apply_threshold(probability)
            else:
                is_positive = False
                label = self.config.negative_label

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return PredictionResult(
                disease_id=self.disease_id,
                disease_display_name=self.config.display_name,
                model_version=self.config.version,
                model_type=self.config.model_type,
                prediction_label=label,
                is_positive=is_positive,
                probability=round(probability, 4) if probability is not None else None,
                decision_threshold=self.config.decision_threshold,
                metadata={
                    "latency_ms": round(elapsed_ms, 2),
                    "features_count": len(input_data),
                },
            )

        except (InputValidationError, ValueError):
            raise
        except Exception as e:
            raise InferenceError(
                f"Inference failed for disease '{self.disease_id}': {str(e)}"
            ) from e


class GenericImagePredictor(BaseImagePredictor):
    """
    Generic predictor for computer vision / deep learning models (e.g. Pneumonia).
    Orchestrates: Byte Decoding -> Resizing & Normalization -> Tensor Inference -> Thresholding -> PredictionResult.
    """

    def load_model(self) -> Any:
        """Loads deep learning model artifact via loaders layer."""
        return load_model_for_disease(self.config)

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Decodes raw image bytes, resizes according to image_spec, normalizes pixels,
        and returns a 4D batch tensor of shape (1, H, W, C).
        """
        if not image_bytes or len(image_bytes) == 0:
            raise InputValidationError("Image payload is empty.")

        spec = self.config.image_spec
        if not spec:
            raise InputValidationError(
                f"Disease '{self.disease_id}' is configured as IMAGE but lacks image_spec."
            )

        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Channel formatting
            if spec.channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

            # Extract dimensions (target_dimensions is [height, width] or [width, height])
            dims = spec.target_dimensions if hasattr(spec, "target_dimensions") and spec.target_dimensions else [224, 224]
            target_h, target_w = dims[0], dims[1]

            # Resize to target dimensions (PIL expects (width, height))
            img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

            # Convert to float numpy array
            arr = np.array(img, dtype=np.float32)

            # Apply grayscale channel dimension if needed
            if spec.channels == 1 and arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)

            # Rescaling (e.g. 1/255.0)
            rescaling = getattr(spec, "rescaling_factor", None) or (1.0 / 255.0)
            arr = arr * rescaling

            # Standardization (mean / std) if provided
            norm_mean = getattr(spec, "normalization_mean", None)
            norm_std = getattr(spec, "normalization_std", None)
            if norm_mean and norm_std:
                mean = np.array(norm_mean, dtype=np.float32)
                std = np.array(norm_std, dtype=np.float32)
                arr = (arr - mean) / std

            # Add batch dimension: (1, H, W, C)
            batch = np.expand_dims(arr, axis=0)
            return batch

        except Exception as e:
            if isinstance(e, InputValidationError):
                raise
            raise InputValidationError(f"Failed to decode and preprocess image: {str(e)}") from e

    def predict(self, input_data: bytes) -> PredictionResult:
        """
        Executes computer vision inference on raw image bytes.
        """
        start_time = time.perf_counter()

        try:
            batch = self.preprocess_image(input_data)
            model = self.get_model()

            # Execute forward pass
            if callable(model):
                raw_output = model(batch)
            elif hasattr(model, "predict"):
                raw_output = model.predict(batch, verbose=0)
            else:
                raise InferenceError(f"Model for '{self.disease_id}' is not callable and lacks predict().")

            # Convert torch/tensorflow/numpy output tensor to scalar float
            if hasattr(raw_output, "detach"):
                raw_output = raw_output.detach().cpu().numpy()
            elif hasattr(raw_output, "numpy"):
                raw_output = raw_output.numpy()

            raw_val = float(np.ravel(raw_output)[0])

            # Apply sigmoid if raw logits were returned (< 0 or > 1)
            if raw_val < 0.0 or raw_val > 1.0:
                probability = float(1.0 / (1.0 + np.exp(-raw_val)))
            else:
                probability = float(np.clip(raw_val, 0.0, 1.0))

            is_positive, label = self.apply_threshold(probability)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return PredictionResult(
                disease_id=self.disease_id,
                disease_display_name=self.config.display_name,
                model_version=self.config.version,
                model_type=self.config.model_type,
                prediction_label=label,
                is_positive=is_positive,
                probability=round(probability, 4),
                decision_threshold=self.config.decision_threshold,
                metadata={
                    "latency_ms": round(elapsed_ms, 2),
                    "input_shape": list(batch.shape),
                },
            )

        except (InputValidationError, ValueError):
            raise
        except Exception as e:
            raise InferenceError(
                f"Image inference failed for disease '{self.disease_id}': {str(e)}"
            ) from e


def get_generic_predictor(config: DiseaseConfig) -> BasePredictor:
    """
    Factory resolving the appropriate generic predictor based on disease category.
    """
    if config.category == DiseaseCategory.TABULAR:
        return GenericTabularPredictor(config)
    elif config.category == DiseaseCategory.IMAGE:
        return GenericImagePredictor(config)
    else:
        raise ValueError(f"Unsupported disease category: '{config.category}'")
