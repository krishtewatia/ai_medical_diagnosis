from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.database.connection import get_database
from app.services.user_service import UserService


security = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    return decode_access_token(credentials.credentials)


def get_current_user(
    user_id: str = Depends(get_current_user_id),
    database = Depends(get_database),
) -> dict:
    user_service = UserService(database)

    user = user_service.find_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found."
        )

    return user
