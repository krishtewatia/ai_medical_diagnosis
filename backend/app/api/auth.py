from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.database.connection import get_database
from app.schemas.user import TokenResponse, UserCreate, UserLogin, UserResponse
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
def register_user(
    user: UserCreate,
    database = Depends(get_database),
):
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


@router.post(
    "/login",
    response_model=TokenResponse
)
def login_user(
    user: UserLogin,
    database = Depends(get_database),
):
    user_service = UserService(database)

    existing_user = user_service.find_by_email(user.email)

    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not verify_password(
        user.password,
        existing_user["password_hash"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    access_token = create_access_token(
        str(existing_user["_id"])
    )

    return TokenResponse(
        access_token=access_token
    )


@router.get(
    "/me",
    response_model=UserResponse
)
def get_me(
    current_user: dict = Depends(get_current_user)
):
    return UserResponse(
        id=str(current_user["_id"]),
        name=current_user["name"],
        email=current_user["email"]
    )
