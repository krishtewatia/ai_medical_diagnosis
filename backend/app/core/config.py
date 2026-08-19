from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AI Medical Diagnosis Assistant"
    app_version: str = "0.1.0"
    debug: bool = True

    mongodb_uri: str
    database_name: str = "ai_medical_diagnosis"

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore"
    )


settings = Settings()