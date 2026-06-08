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

    async def create_prediction_batch(
        self,
        *,
        files_payload: list[dict[str, Any]],
        metadata: dict[str, Any],
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        files: list[tuple[str, tuple[Any, ...]]] = []
        for item in files_payload:
            files.append(
                (
                    "files",
                    (
                        item["filename"],
                        item["file_bytes"],
                        item["content_type"],
                    ),
                )
            )

        files.append(("metadata_json", (None, json.dumps(metadata))))
        if batch_id:
            files.append(("batch_id", (None, batch_id)))

        response = await self._client.post("/predictions/batch", files=files)
        response.raise_for_status()
        return response.json()
