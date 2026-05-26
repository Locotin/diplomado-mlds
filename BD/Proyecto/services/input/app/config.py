from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "emotion-stack"
    ingesta_service_url: str = "http://ingesta:8001"
    input_port: int = 8002
    input_max_file_size_mb: int = Field(default=10, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
