from io import BytesIO
from time import perf_counter

import anyio
import httpx
from PIL import Image
import torch
from transformers import pipeline

from app.config import Settings
from app.models import ModelInferenceResponse
from app.services.normalization import normalize_predictions


def _resolve_device(device_name: str) -> str | int:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return -1
    if normalized == "cpu":
        return -1
    if normalized == "cuda":
        return 0
    if normalized == "mps":
        return "mps"
    return -1


class EmotionInferenceService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline = None
        self._pipeline_lock = anyio.Lock()

    async def infer_from_url(self, *, source_id: str, image_url: str) -> ModelInferenceResponse:
        await self._ensure_pipeline()

        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.get(image_url)
            response.raise_for_status()

        started_at = perf_counter()
        predictions = await anyio.to_thread.run_sync(self._run_pipeline, response.content)
        inference_ms = int((perf_counter() - started_at) * 1000)

        if not predictions:
            raise ValueError("Model pipeline returned no predictions.")

        top_emotion = max(predictions, key=lambda item: item.score).label
        return ModelInferenceResponse(
            source_id=source_id,
            model_provider=self._settings.model_provider,
            model_id=self._settings.model_id,
            predictions=predictions,
            top_emotion=top_emotion,
            inference_ms=inference_ms,
        )

    async def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        async with self._pipeline_lock:
            if self._pipeline is not None:
                return
            self._pipeline = await anyio.to_thread.run_sync(self._build_pipeline)

    def _build_pipeline(self):
        return pipeline(
            task="image-classification",
            model=self._settings.model_id,
            revision=self._settings.model_revision or None,
            device=_resolve_device(self._settings.model_device),
        )

    def _run_pipeline(self, image_bytes: bytes):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        raw_predictions = self._pipeline(image, top_k=self._settings.model_top_k)
        return normalize_predictions(raw_predictions, self._settings.model_top_k)
