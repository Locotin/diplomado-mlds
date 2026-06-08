import json
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.config import Settings, get_settings
from app.models import BatchPredictionResponse, PredictionRecord
from app.utils.ids import generate_source_id

router = APIRouter(tags=["uploads"])


def _ensure_image(file: UploadFile) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image uploads are supported.")


def _parse_metadata(raw_metadata: str | None) -> dict[str, Any]:
    if not raw_metadata:
        return {}

    try:
        parsed = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="metadata_json must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="metadata_json must be a JSON object.")

    return parsed


@router.post("/uploads", response_model=PredictionRecord, status_code=status.HTTP_201_CREATED)
async def upload_image(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    source_id: Annotated[str | None, Form()] = None,
    metadata_json: Annotated[str | None, Form()] = None,
    settings: Settings = Depends(get_settings),
) -> PredictionRecord:
    _ensure_image(file)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty files are not allowed.")

    max_bytes = settings.input_max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.input_max_file_size_mb} MB limit.",
        )

    metadata = _parse_metadata(metadata_json)
    metadata.setdefault("content_type", file.content_type)
    metadata.setdefault("filename", file.filename or "upload")
    metadata.setdefault("client_host", request.client.host if request.client else "unknown")

    resolved_source_id = source_id or generate_source_id()
    ingesta_client = request.app.state.ingesta_client

    try:
        payload = await ingesta_client.create_prediction(
            file_bytes=file_bytes,
            filename=file.filename or f"{resolved_source_id}.bin",
            content_type=file.content_type,
            source_id=resolved_source_id,
            metadata=metadata,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Ingesta service rejected the request."
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach the ingesta service.",
        ) from exc

    return PredictionRecord.model_validate(payload)


@router.post("/uploads/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_201_CREATED)
async def upload_images_batch(
    request: Request,
    files: Annotated[list[UploadFile], File(...)],
    metadata_json: Annotated[str | None, Form()] = None,
    batch_id: Annotated[str | None, Form()] = None,
    settings: Settings = Depends(get_settings),
) -> BatchPredictionResponse:
    metadata = _parse_metadata(metadata_json)
    metadata.setdefault("source", "batch-upload")
    metadata.setdefault("client_host", request.client.host if request.client else "unknown")

    max_bytes = settings.input_max_file_size_mb * 1024 * 1024
    files_payload: list[dict[str, Any]] = []
    for index, file in enumerate(files, start=1):
        _ensure_image(file)

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File {index} is empty.")

        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename or index} exceeds the {settings.input_max_file_size_mb} MB limit.",
            )

        files_payload.append(
            {
                "filename": file.filename or generate_source_id(),
                "file_bytes": file_bytes,
                "content_type": file.content_type or "application/octet-stream",
            }
        )

    ingesta_client = request.app.state.ingesta_client
    try:
        payload = await ingesta_client.create_prediction_batch(
            files_payload=files_payload,
            metadata=metadata,
            batch_id=batch_id,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Ingesta service rejected the batch request."
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach the ingesta service.",
        ) from exc

    return BatchPredictionResponse.model_validate(payload)
