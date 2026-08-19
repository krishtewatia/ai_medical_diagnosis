from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_current_user
from app.schemas.prediction_history import (
    PredictionHistoryListResponse,
    PredictionHistoryResponse,
)
from app.services.prediction_history_service import (
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


@router.get(
    "",
    response_model=PredictionHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get User Prediction History",
    description=(
        "Retrieves a paginated list of historical medical screening predictions for the authenticated user. "
        "Results are ordered reverse-chronologically (newest first). "
        "Access is strictly scoped to the authenticated user."
    )
)
def get_prediction_history(
    skip: int = Query(default=0, ge=0, description="Number of records to skip for pagination."),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of records to return (1-100)."),
    current_user: dict = Depends(get_current_user),
    history_service: PredictionHistoryService = Depends(get_history_service)
) -> PredictionHistoryListResponse:
    """
    Paginated prediction history retrieval endpoint.
    Flow:
    1. Authenticate user via JWT
    2. Extract user_id from verified token
    3. Query MongoDB predictions collection for only this user's records
    4. Return chronologically sorted history list with pagination metadata
    """
    user_id = current_user.get("_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or invalid."
        )

    try:
        items = history_service.get_user_predictions(
            user_id=user_id,
            skip=skip,
            limit=limit
        )
        total = history_service.count_user_predictions(user_id=user_id)

        return PredictionHistoryListResponse(
            items=[PredictionHistoryResponse(**item) for item in items],
            total=total,
            limit=limit,
            skip=skip
        )

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
