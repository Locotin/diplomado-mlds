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


class ModelInferenceRequest(BaseModel):
    source_id: str
    image_url: HttpUrl


class ModelInferenceResponse(BaseModel):
    source_id: str
    model_provider: str
    model_id: str
    predictions: list[EmotionScore]
    top_emotion: str
    inference_ms: int = Field(ge=0)


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


class PredictionListResponse(BaseModel):
    items: list[PredictionRecord]
    total: int = Field(ge=0)
