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


class BatchManifestRow(BaseModel):
    source_id: str
    filename: str
    content_type: str | None = None
    byte_size: int = Field(ge=0)


class BatchEmotionSummary(BaseModel):
    top_emotion: str
    count: int = Field(ge=0)


class BatchPredictionFailure(BaseModel):
    filename: str
    detail: str
    source_id: str | None = None


class BatchPredictionResponse(BaseModel):
    batch_id: str
    total_received: int = Field(ge=0)
    total_processed: int = Field(ge=0)
    total_failed: int = Field(ge=0)
    manifest_preview: list[BatchManifestRow] = Field(default_factory=list)
    emotion_summary: list[BatchEmotionSummary] = Field(default_factory=list)
    items: list[PredictionRecord] = Field(default_factory=list)
    failures: list[BatchPredictionFailure] = Field(default_factory=list)
