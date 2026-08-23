from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.database import router as database_router
from app.api.disease import router as disease_router
from app.api.health import router as health_router
from app.api.history import router as history_router
from app.api.prediction import router as prediction_router
from app.api.profile import router as profile_router
from app.api.report import router as report_router
from app.api.storage import router as storage_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# Enable Cross-Origin Resource Sharing (CORS) for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(database_router)
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(prediction_router)
app.include_router(history_router)
app.include_router(disease_router)
app.include_router(storage_router)
app.include_router(report_router)