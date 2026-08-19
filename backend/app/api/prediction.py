from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_current_user
from app.ml.disease_registry import DiseaseNotFoundError, disease_registry
from app.schemas.disease_config import DiseasePublicInfo
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_service import (
    PredictionInferenceError,
    PredictionService,
    PredictionValidationError,
)


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


def get_prediction_service() -> PredictionService:
    """Dependency provider for PredictionService with PredictionHistoryService attached."""
    return PredictionService(
        registry=disease_registry,
        history_service=PredictionHistoryService()
    )


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Generic Tabular Disease Screening",
    description=(
        "Executes disease risk prediction for tabular disease models (e.g. Diabetes, Heart Disease). "
        "Requires authentication. Input features are validated dynamically against the disease configuration. "
        "Successful predictions are automatically recorded to user history."
    )
)
def predict_tabular_disease(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Generic tabular prediction endpoint.
    Flow:
    1. Authenticate user via JWT
    2. Lookup disease in registry
    3. Validate disease-specific tabular features
    4. Run ML model inference & apply decision threshold
    5. Save successful prediction to user's MongoDB history
    6. Return standardized PredictionResponse
    """
    user_id = current_user.get("_id")

    try:
        return service.predict_tabular(request, user_id=user_id)

    except DiseaseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{request.disease_id}' is not registered or not available."
        )

    except PredictionValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    except PredictionInferenceError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed for the specified model."
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction processing."
        )


@router.post(
    "/image",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Generic Medical Image Screening",
    description=(
        "Accepts medical image file uploads (e.g. Pneumonia chest X-rays, Brain MRI scans) "
        "and executes deep learning screening through the generic predictor pipeline. "
        "Validates file size, format, and image integrity before inference. "
        "Successful predictions are automatically recorded to user history."
    )
)
async def predict_image_disease(
    disease_id: str = Form(..., description="Unique disease identifier for image screening (e.g. 'pneumonia')."),
    file: UploadFile = File(..., description="Medical image file (.png, .jpg, .jpeg)."),
    current_user: dict = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Generic file-upload prediction endpoint.
    Flow:
    1. Authenticate user via JWT
    2. Read uploaded file bytes & metadata
    3. Validate file presence, MIME type, size limits, and image decodability
    4. Pass validated bytes to image predictor for model-specific preprocessing & inference
    5. Save successful prediction to user's MongoDB history
    6. Return standardized PredictionResponse identical to tabular screening
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Medical image file must be provided."
        )

    user_id = current_user.get("_id")

    try:
        image_bytes = await file.read()
        return service.predict_image(
            disease_id=disease_id,
            image_bytes=image_bytes,
            filename=file.filename,
            content_type=file.content_type,
            user_id=user_id
        )

    except DiseaseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{disease_id}' is not registered or not available."
        )

    except PredictionValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    except PredictionInferenceError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image inference execution failed for the specified model."
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during medical image processing."
        )


@router.get(
    "/diseases",
    response_model=List[DiseasePublicInfo],
    status_code=status.HTTP_200_OK,
    summary="List Registered Disease Models",
    description="Returns public metadata, expected features, metrics, and disclaimers for all active disease models."
)
def list_available_diseases(
    current_user: dict = Depends(get_current_user),
) -> List[DiseasePublicInfo]:
    """Returns metadata for all available disease models."""
    return disease_registry.list_public_info()
