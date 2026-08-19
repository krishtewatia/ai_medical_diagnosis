from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.database.connection import get_database
from app.schemas.medical_profile import (
    MedicalProfileCreate,
    MedicalProfileResponse,
    MedicalProfileUpdate,
)
from app.services.medical_profile_service import (
    DuplicateProfileError,
    MedicalProfileService,
)


router = APIRouter(
    prefix="/profile",
    tags=["Medical Profile"]
)


def _to_profile_response(doc: dict) -> MedicalProfileResponse:
    return MedicalProfileResponse(
        id=str(doc["_id"]),
        user_id=str(doc["user_id"]),
        date_of_birth=doc["date_of_birth"],
        gender=doc["gender"],
        blood_type=doc.get("blood_type"),
        height_cm=doc.get("height_cm"),
        weight_kg=doc.get("weight_kg"),
        allergies=doc.get("allergies", []),
        chronic_conditions=doc.get("chronic_conditions", []),
        current_medications=doc.get("current_medications", []),
        smoking_status=doc.get("smoking_status"),
        alcohol_consumption=doc.get("alcohol_consumption"),
        emergency_contact=doc.get("emergency_contact"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.post(
    "",
    response_model=MedicalProfileResponse,
    status_code=status.HTTP_201_CREATED
)
def create_profile(
    profile_in: MedicalProfileCreate,
    current_user: dict = Depends(get_current_user),
    database = Depends(get_database),
):
    service = MedicalProfileService(database)
    user_id = str(current_user["_id"])

    try:
        doc = service.create_profile(user_id, profile_in)
        return _to_profile_response(doc)
    except DuplicateProfileError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A medical profile already exists for this user."
        )


@router.get(
    "",
    response_model=MedicalProfileResponse
)
def get_profile(
    current_user: dict = Depends(get_current_user),
    database = Depends(get_database),
):
    service = MedicalProfileService(database)
    user_id = str(current_user["_id"])

    doc = service.find_by_user_id(user_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medical profile not found."
        )

    return _to_profile_response(doc)


@router.patch(
    "",
    response_model=MedicalProfileResponse
)
def update_profile(
    profile_update: MedicalProfileUpdate,
    current_user: dict = Depends(get_current_user),
    database = Depends(get_database),
):
    service = MedicalProfileService(database)
    user_id = str(current_user["_id"])

    if not service.has_profile(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medical profile not found."
        )

    doc = service.update_profile(user_id, profile_update)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medical profile not found."
        )

    return _to_profile_response(doc)
