import httpx

from app.models import ModelInferenceRequest, ModelInferenceResponse


class ModelServiceClient:
    def __init__(self, *, base_url: str, timeout_seconds: int) -> None:
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def infer(self, *, source_id: str, image_url: str) -> ModelInferenceResponse:
        payload = ModelInferenceRequest(source_id=source_id, image_url=image_url)
        response = await self._client.post("/infer", json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return ModelInferenceResponse.model_validate(response.json())

    def infer_sync(self, *, source_id: str, image_url: str) -> ModelInferenceResponse:
        payload = ModelInferenceRequest(source_id=source_id, image_url=image_url)
        with httpx.Client(base_url=self._base_url, timeout=self._timeout_seconds) as client:
            response = client.post("/infer", json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return ModelInferenceResponse.model_validate(response.json())
