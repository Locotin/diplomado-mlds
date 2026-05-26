import json
from typing import Any

import httpx


class IngestaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 60) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def create_prediction(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        source_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        files = {
            "file": (filename, file_bytes, content_type),
            "source_id": (None, source_id),
            "metadata_json": (None, json.dumps(metadata)),
        }
        response = await self._client.post("/predictions", files=files)
        response.raise_for_status()
        return response.json()
