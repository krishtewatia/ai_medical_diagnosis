from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.ml.disease_registry import DiseaseNotFoundError, disease_registry
from app.schemas.disease import DiseaseResponse
from app.services.disease_service import DiseaseService


router = APIRouter(
    prefix="/diseases",
    tags=["Disease Discovery"]
)


def get_disease_service() -> DiseaseService:
    """Dependency provider for DiseaseService."""
    return DiseaseService(registry=disease_registry)


@router.get(
    "",
    response_model=List[DiseaseResponse],
    status_code=status.HTTP_200_OK,
    summary="List Available Disease Modules",
    description=(
        "Returns public configuration and capability metadata for all actively registered disease screening modules. "
        "Allows the frontend to dynamically discover available modules and render disease selection cards."
    )
)
def list_diseases(
    service: DiseaseService = Depends(get_disease_service)
) -> List[DiseaseResponse]:
    """
    Retrieves all currently active disease modules.
    Public/read-only capability discovery endpoint.
    """
    return service.get_all_diseases(active_only=True)


@router.get(
    "/{disease}",
    response_model=DiseaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Specific Disease Module Configuration",
    description=(
        "Retrieves detailed declarative metadata for a specific disease module, "
        "including required tabular input fields, units, validation ranges, image upload specifications, "
        "model telemetry, and clinical disclaimers for dynamic form generation."
    )
)
def get_disease(
    disease: str,
    service: DiseaseService = Depends(get_disease_service)
) -> DiseaseResponse:
    """
    Retrieves configuration metadata for a single disease module by slug.
    Returns 404 Not Found if the disease is unknown or inactive.
    """
    try:
        return service.get_disease_or_raise(disease)
    except DiseaseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{disease}' is not registered or not available."
        )
