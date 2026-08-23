from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_current_user
from app.ml.disease_registry import DiseaseRegistry, disease_registry
from app.schemas.prediction_history import (
    PredictionHistoryListResponse,
    PredictionHistoryResponse,
)
from app.services.prediction_history_service import (
    InvalidPredictionIdError,
    InvalidUserIdError,
    PredictionHistoryError,
    PredictionHistoryService,
)


router = APIRouter(
    prefix="/history",
    tags=["Prediction History"]
)


def get_history_service() -> PredictionHistoryService:
    """Dependency provider for PredictionHistoryService."""
    return PredictionHistoryService()


def get_disease_registry() -> DiseaseRegistry:
    """Dependency provider for DiseaseRegistry."""
    return disease_registry


@router.get(
    "",
    response_model=PredictionHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get User Prediction History",
    description=(
        "Retrieves a paginated list of historical medical screening predictions for the authenticated user. "
        "Results are ordered reverse-chronologically (newest first). "
        "Access is strictly scoped to the authenticated user. "
        "Supports optional disease filtering (e.g. ?disease=diabetes)."
    )
)
def get_prediction_history(
    disease: Optional[str] = Query(default=None, description="Optional disease identifier to filter results (e.g. 'diabetes', 'heart_disease')."),
    skip: int = Query(default=0, ge=0, description="Number of records to skip for pagination."),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of records to return (1-100)."),
    current_user: dict = Depends(get_current_user),
    history_service: PredictionHistoryService = Depends(get_history_service),
    registry: DiseaseRegistry = Depends(get_disease_registry)
) -> PredictionHistoryListResponse:
    """
    Paginated prediction history retrieval endpoint with optional disease filtering.
    Flow:
    1. Authenticate user via JWT
    2. Extract user_id from verified token
    3. If disease param provided, validate against DiseaseRegistry (return 404 for unknown disease)
    4. Query MongoDB predictions collection for only this user's records (filtered by disease if provided)
    5. Return chronologically sorted history list with pagination metadata
    """
    user_id = current_user.get("_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or invalid."
        )

    filter_disease: Optional[str] = None
    if disease is not None:
        cleaned_disease = disease.strip().lower()
        if cleaned_disease:
            if not registry.has_disease(cleaned_disease):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Disease '{disease}' is not registered or not available."
                )
            filter_disease = cleaned_disease

    try:
        items = history_service.get_user_predictions(
            user_id=user_id,
            disease=filter_disease,
            skip=skip,
            limit=limit
        )
        total = history_service.count_user_predictions(
            user_id=user_id,
            disease=filter_disease
        )

        return PredictionHistoryListResponse(
            items=[PredictionHistoryResponse(**item) for item in items],
            total=total,
            limit=limit,
            skip=skip
        )

    except HTTPException:
        raise
    except PredictionHistoryError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction history from database."
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while querying prediction history."
        )


@router.get(
    "/{prediction_id}",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Specific Prediction Record",
    description=(
        "Retrieves a single historical prediction record by ID. "
        "Strictly enforces user ownership: queries both prediction ID and user ID. "
        "Cross-user or nonexistent lookups return 404 Not Found."
    )
)
def get_prediction_by_id(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
    history_service: PredictionHistoryService = Depends(get_history_service)
) -> PredictionHistoryResponse:
    """
    Single prediction history record retrieval endpoint.
    Flow:
    1. Authenticate user via JWT
    2. Extract user_id from verified token
    3. Validate prediction_id
    4. Query MongoDB predictions collection with both user_id AND prediction_id
    5. Return complete prediction record (PredictionHistoryResponse)
    6. Return 400 for malformed prediction ID, 404 for not found / inaccessible
    """
    user_id = current_user.get("_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or invalid."
        )

    try:
        record = history_service.get_prediction_by_id(
            user_id=user_id,
            prediction_id=prediction_id
        )
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction record not found."
            )
        return PredictionHistoryResponse(**record)

    except HTTPException:
        raise
    except (InvalidPredictionIdError, InvalidUserIdError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid prediction ID format."
        )
    except PredictionHistoryError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction record from database."
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while querying prediction record."
        )

