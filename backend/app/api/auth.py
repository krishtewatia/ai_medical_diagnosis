from fastapi import APIRouter, HTTPException, status

from app.core.security import hash_password
from app.database.connection import get_database
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import UserService


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def register_user(user: UserCreate):
    database = get_database()

    user_service = UserService(database)

    existing_user = user_service.find_by_email(user.email)

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )

    password_hash = hash_password(user.password)

    created_user = user_service.create_user(
        name=user.name,
        email=user.email,
        password_hash=password_hash
    )

    return UserResponse(
        id=str(created_user["_id"]),
        name=created_user["name"],
        email=created_user["email"]
    )

