from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str


class EmotionScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


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
