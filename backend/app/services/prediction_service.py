import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from bson import ObjectId

from app.ml.base_predictor import BaseImagePredictor, BasePredictor, BaseTabularPredictor
from app.ml.disease_registry import (
    DiseaseNotFoundError,
    DiseaseRegistry,
    PredictorNotRegisteredError,
    disease_registry as default_registry,
)
from app.ml.predictor import (
    InferenceError,
    InputValidationError,
    get_generic_predictor,
)
from app.ml.validator import DiseaseInputValidator, PredictionValidationError
from app.schemas.disease_config import DiseaseCategory, DiseaseConfig
from app.schemas.prediction import (
    PredictionRequest,
    PredictionRequestPayload,
    PredictionResponse,
    PredictionResult,
)
from app.schemas.prediction_history import (
    PredictionHistoryCreate,
    PredictionModelInfo,
    PredictionResultRecord,
)
from app.services.prediction_history_service import PredictionHistoryService


class PredictionServiceError(Exception):
    """Base exception for all prediction service failures."""
    pass


class PredictionInferenceError(RuntimeError, PredictionServiceError):
    """Raised when model inference fails during execution."""
    pass


class PredictionService:
    """
    Business logic layer that orchestrates the end-to-end prediction lifecycle:
    Prediction Request -> Input Validation -> Disease Lookup -> Predictor Resolution
    -> Model Inference -> History Persistence -> Standard PredictionResponse.
    
    Guarantees:
    1. Validation First: Invalid or malformed inputs are halted and rejected BEFORE reaching model.
    2. Atomic History Saving: History is saved ONLY after successful model inference.
    3. User Isolation: Predictions are tagged with the verified JWT user_id.
    """

    def __init__(
        self,
        registry: Optional[DiseaseRegistry] = None,
        history_service: Optional[PredictionHistoryService] = None
    ):
        self.registry = registry or default_registry
        self.history_service = history_service

    def _resolve_predictor(self, config: DiseaseConfig) -> BasePredictor:
        """
        Resolves the predictor instance for a disease, falling back to
        the generic predictor factory if no custom predictor was explicitly bound.
        """
        try:
            return self.registry.get_predictor(config.id)
        except PredictorNotRegisteredError:
            predictor = get_generic_predictor(config)
            self.registry.register_predictor(config.id, predictor.__class__)
            return predictor

    def _save_history_safely(
        self,
        user_id: Union[str, ObjectId],
        history_record: PredictionHistoryCreate
    ) -> None:
        """Helper to persist prediction history without allowing database telemetry failures to crash response."""
        if not self.history_service or not user_id:
            return
        try:
            self.history_service.create_prediction(user_id=user_id, payload=history_record)
        except Exception as e:
            # Telemetry/history persistence warning logged without interrupting user flow
            print(f"Warning: Failed to persist prediction history for user {user_id}: {e}")

    def predict_tabular(
        self,
        request: PredictionRequest,
        user_id: Optional[Union[str, ObjectId]] = None
    ) -> PredictionResponse:
        """
        Validates inputs against disease configuration and executes screening inference for tabular diseases.
        Saves immutable prediction record to MongoDB upon successful inference.
        """
        disease_id = request.disease_id.strip().lower()
        config = self.registry.get_or_raise(disease_id)

        # 1. Category validation
        if config.category != DiseaseCategory.TABULAR:
            raise PredictionValidationError(
                f"Disease '{disease_id}' is configured as '{config.category}', not 'tabular'. "
                f"Use the image prediction pipeline instead."
            )

        # 2. Strict Input Validation (Required fields, types, allowed categorical values, min/max ranges)
        validated_inputs = DiseaseInputValidator.validate_tabular_inputs(config, request.inputs)

        # 3. Model Resolution & Inference Execution (only reached if validation completely passes)
        predictor = self._resolve_predictor(config)

        start_time = time.perf_counter()
        try:
            result: PredictionResult = predictor.predict(validated_inputs)
        except (InputValidationError, ValueError) as e:
            raise PredictionValidationError(
                f"Validation failed for disease '{disease_id}': {str(e)}"
            ) from e
        except (InferenceError, Exception) as e:
            raise PredictionInferenceError(
                f"Inference failed for disease '{disease_id}': {str(e)}"
            ) from e

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result.metadata["latency_ms"] = latency_ms
        result.metadata["features_evaluated"] = len(validated_inputs)

        response = PredictionResponse.from_result(
            result=result,
            clinical_purpose=config.clinical_purpose,
            disclaimer=config.disclaimer,
        )

        # 4. Save to MongoDB Prediction History (only on verified success)
        if user_id:
            history_record = PredictionHistoryCreate(
                user_id=str(user_id),
                disease=config.id,
                disease_display_name=config.display_name,
                input_type="tabular",
                model=PredictionModelInfo(
                    version=config.version,
                    model_type=config.model_type,
                    threshold=config.decision_threshold,
                ),
                input_data=validated_inputs,
                result=PredictionResultRecord(
                    prediction=result.prediction_label,
                    is_positive=result.is_positive,
                    probability=result.probability,
                    confidence=None,
                ),
                explanation=response.explanation,
                metadata={
                    "source": "api",
                    "latency_ms": latency_ms,
                    "features_evaluated": len(validated_inputs),
                },
            )
            self._save_history_safely(user_id, history_record)

        return response

    def predict_image(
        self,
        disease_id: str,
        image_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[Union[str, ObjectId]] = None
    ) -> PredictionResponse:
        """
        Validates image payload and executes screening inference for medical image diseases.
        Saves immutable prediction record to MongoDB upon successful inference.
        """
        d_id = disease_id.strip().lower()
        config = self.registry.get_or_raise(d_id)

        # 1. Category validation
        if config.category != DiseaseCategory.IMAGE:
            raise PredictionValidationError(
                f"Disease '{d_id}' is configured as '{config.category}', not 'image'. "
                f"Use the tabular prediction pipeline instead."
            )

        # 2. Strict Image Input Validation (Size limits, non-emptiness, allowed formats, corruption checks)
        validated_bytes = DiseaseInputValidator.validate_image_input(
            config=config,
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type
        )

        # 3. Model Resolution & Inference Execution
        predictor = self._resolve_predictor(config)

        start_time = time.perf_counter()
        try:
            result: PredictionResult = predictor.predict(validated_bytes)
        except (InputValidationError, ValueError) as e:
            raise PredictionValidationError(
                f"Image validation failed for disease '{d_id}': {str(e)}"
            ) from e
        except (InferenceError, Exception) as e:
            raise PredictionInferenceError(
                f"Image inference failed for disease '{d_id}': {str(e)}"
            ) from e

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result.metadata["latency_ms"] = latency_ms

        response = PredictionResponse.from_result(
            result=result,
            clinical_purpose=config.clinical_purpose,
            disclaimer=config.disclaimer,
        )

        # 4. Save to MongoDB Prediction History (only on verified success)
        if user_id:
            history_record = PredictionHistoryCreate(
                user_id=str(user_id),
                disease=config.id,
                disease_display_name=config.display_name,
                input_type="image",
                model=PredictionModelInfo(
                    version=config.version,
                    model_type=config.model_type,
                    threshold=config.decision_threshold,
                ),
                input_data={
                    "filename": Path(filename).name if filename else "medical_image",
                    "content_type": content_type or "image/png",
                    "size_bytes": len(image_bytes),
                },
                result=PredictionResultRecord(
                    prediction=result.prediction_label,
                    is_positive=result.is_positive,
                    probability=result.probability,
                    confidence=None,
                ),
                explanation=response.explanation,
                metadata={
                    "source": "api",
                    "latency_ms": latency_ms,
                },
            )
            self._save_history_safely(user_id, history_record)

        return response

    def predict(
        self,
        payload: PredictionRequestPayload,
        user_id: Optional[Union[str, ObjectId]] = None
    ) -> PredictionResponse:
        """
        Unified dispatch accepting a generic PredictionRequestPayload.
        """
        config = self.registry.get_or_raise(payload.disease_id)

        if config.category == DiseaseCategory.TABULAR:
            inputs = payload.tabular_inputs or {}
            req = PredictionRequest(disease_id=payload.disease_id, inputs=inputs)
            return self.predict_tabular(req, user_id=user_id)
        elif config.category == DiseaseCategory.IMAGE:
            if not payload.image_bytes:
                raise PredictionValidationError(f"Image bytes required for disease '{payload.disease_id}'.")
            return self.predict_image(
                disease_id=payload.disease_id,
                image_bytes=payload.image_bytes,
                content_type=payload.image_content_type,
                user_id=user_id
            )
        else:
            raise PredictionValidationError(f"Unsupported disease category '{config.category}'.")
