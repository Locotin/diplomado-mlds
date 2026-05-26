from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "emotion-stack"
    ingesta_port: int = 8001
    model_service_url: str = "http://modelo:8000"
    mongodb_uri: str = ""
    mongodb_database: str = "emotion_analytics"
    mongodb_collection: str = "predictions"
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    cloudinary_folder: str = "emotion-stack"
    request_timeout_seconds: int = Field(default=60, ge=5)

    @model_validator(mode="after")
    def validate_required_integrations(self) -> "Settings":
        required = {
            "mongodb_uri": self.mongodb_uri,
            "cloudinary_cloud_name": self.cloudinary_cloud_name,
            "cloudinary_api_key": self.cloudinary_api_key,
            "cloudinary_api_secret": self.cloudinary_api_secret,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required settings: {joined}")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
