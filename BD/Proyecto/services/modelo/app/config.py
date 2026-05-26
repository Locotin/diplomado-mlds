from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "emotion-stack"
    modelo_port: int = 8000
    model_provider: str = "huggingface"
    model_id: str = "abhilash88/face-emotion-detection"
    model_revision: str | None = None
    model_device: str = "cpu"
    model_top_k: int = Field(default=5, ge=1, le=10)
    request_timeout_seconds: int = Field(default=60, ge=5)


@lru_cache
def get_settings() -> Settings:
    return Settings()
