from fastapi import APIRouter, Response, status

from app.database.connection import check_database_connection


router = APIRouter()


@router.get("/database/health")
def database_health(response: Response):
    connected = check_database_connection()

    if connected:
        return {
            "status": "healthy",
            "database": "connected"
        }

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "unhealthy",
        "database": "disconnected"
    }
