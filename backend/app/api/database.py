from fastapi import APIRouter

from app.database.connection import check_database_connection


router = APIRouter()


@router.get("/database/health")
def database_health():
    connected = check_database_connection()

    if connected:
        return {
            "status": "healthy",
            "database": "connected"
        }

    return {
        "status": "unhealthy",
        "database": "disconnected"
    }