import json
import re
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi import Form as FormField

from app.models import BatchPredictionFailure, BatchPredictionResponse, PredictionListResponse, PredictionRecord
from app.services.batch_service import BatchFilePayload
from app.services.document_builder import build_prediction_document

router = APIRouter(tags=["predictions"])
_SOURCE_ID_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


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


def _build_batch_source_id(*, filename: str | None, batch_id: str, index: int) -> str:
    token = Path(filename or f"upload-{index}").stem or f"upload-{index}"
    sanitized = _SOURCE_ID_SANITIZER.sub("_", token).strip("._-") or f"upload-{index}"
    return f"{sanitized}-{batch_id[:8]}-{index:04d}"


@router.post("/predictions", response_model=PredictionRecord, status_code=status.HTTP_201_CREATED)
async def create_prediction(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    source_id: Annotated[str, FormField(...)],
    metadata_json: Annotated[str | None, FormField()] = None,
) -> PredictionRecord:
    _ensure_image(file)
    metadata = _parse_metadata(metadata_json)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty files are not allowed.")

    cloudinary_service = request.app.state.cloudinary_service
    model_service = request.app.state.model_service
    repository = request.app.state.prediction_repository

    cloudinary_asset = await cloudinary_service.upload_image(
        file_bytes=file_bytes,
        filename=file.filename or f"{source_id}.bin",
        source_id=source_id,
    )

    try:
        inference = await model_service.infer(source_id=source_id, image_url=str(cloudinary_asset.secure_url))
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or "Model service rejected the request."
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to reach the model service.",
        ) from exc

    enriched_metadata = {
        **metadata,
        "filename": file.filename or "upload",
        "content_type": file.content_type,
    }
    document = build_prediction_document(
        source_id=source_id,
        cloudinary_asset=cloudinary_asset,
        inference=inference,
        metadata=enriched_metadata,
    )
    return await repository.create(document)


@router.post("/predictions/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction_batch(
    request: Request,
    files: Annotated[list[UploadFile], File(...)],
    metadata_json: Annotated[str | None, FormField()] = None,
    batch_id: Annotated[str | None, FormField()] = None,
) -> BatchPredictionResponse:
    metadata = _parse_metadata(metadata_json)
    resolved_batch_id = batch_id or uuid4().hex

    batch_service = request.app.state.batch_prediction_service
    repository = request.app.state.prediction_repository

    valid_items: list[BatchFilePayload] = []
    validation_failures: list[BatchPredictionFailure] = []

    for index, file in enumerate(files, start=1):
        try:
            _ensure_image(file)
        except HTTPException as exc:
            validation_failures.append(
                BatchPredictionFailure(
                    filename=file.filename or f"upload-{index}",
                    detail=str(exc.detail),
                )
            )
            continue

        file_bytes = await file.read()
        if not file_bytes:
            validation_failures.append(
                BatchPredictionFailure(
                    filename=file.filename or f"upload-{index}",
                    detail="Empty files are not allowed.",
                )
            )
            continue

        source_id = _build_batch_source_id(filename=file.filename, batch_id=resolved_batch_id, index=index)
        item_metadata = {
            **metadata,
            "batch_id": resolved_batch_id,
            "batch_index": index,
            "filename": file.filename or f"upload-{index}",
            "content_type": file.content_type,
        }
        valid_items.append(
            BatchFilePayload(
                source_id=source_id,
                filename=file.filename or f"{source_id}.bin",
                content_type=file.content_type,
                file_bytes=file_bytes,
                metadata=item_metadata,
            )
        )

    return await batch_service.process_batch(
        batch_id=resolved_batch_id,
        items=valid_items,
        repository=repository,
        validation_failures=validation_failures,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionRecord)
async def get_prediction(request: Request, prediction_id: str) -> PredictionRecord:
    repository = request.app.state.prediction_repository
    prediction = await repository.get(prediction_id)
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    return prediction


@router.get("/predictions", response_model=PredictionListResponse)
async def list_predictions(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> PredictionListResponse:
    repository = request.app.state.prediction_repository
    return await repository.list_recent(limit)
