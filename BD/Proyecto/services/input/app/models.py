from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str


class EmotionScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class CloudinaryAsset(BaseModel):
    asset_id: str
    public_id: str
    secure_url: HttpUrl
    bytes: int | None = None
    format: str | None = None
    version: int | None = None


class PredictionRecord(BaseModel):
    prediction_id: str
    source_id: str
    status: str
    model_provider: str
    model_id: str
    top_emotion: str
    predictions: list[EmotionScore]
    cloudinary: CloudinaryAsset
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
