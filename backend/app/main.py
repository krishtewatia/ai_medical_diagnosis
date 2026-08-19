from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.database import router as database_router
from app.api.health import router as health_router
from app.api.history import router as history_router
from app.api.prediction import router as prediction_router
from app.api.profile import router as profile_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)


app.include_router(health_router)
app.include_router(database_router)
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(prediction_router)
app.include_router(history_router)