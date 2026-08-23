from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "AI Medical Diagnosis Assistant"
    app_version: str = "0.1.0"
    debug: bool = True

    # Database
    mongodb_uri: str
    database_name: str = "ai_medical_diagnosis"

    # JWT Authentication
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # Environment & Telemetry
    environment: str = "development"  # "development", "production", "staging"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"

    # Production Medical Object Storage (S3 / Cloudflare R2 / Local Fallback)
    storage_driver: str = "local"  # "local", "s3", "r2"
    storage_local_dir: str = str(BACKEND_ROOT / "storage" / "uploads")
    storage_bucket_name: Optional[str] = "medical-imaging-records"
    storage_endpoint_url: Optional[str] = None
    storage_access_key_id: Optional[str] = None
    storage_secret_access_key: Optional[str] = None
    storage_region_name: str = "auto"
    storage_signed_url_expire_seconds: int = 900  # 15 minutes default

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        extra="ignore"
    )

    def get_cors_origins(self) -> list[str]:
        """Returns parsed list of allowed CORS origins from comma-separated string."""
        if not self.cors_origins:
            return ["*"]
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins if origins else ["*"]


settings = Settings()