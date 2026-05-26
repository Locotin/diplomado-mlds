import json
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi import Form as FormField

from app.models import PredictionListResponse, PredictionRecord
from app.services.document_builder import build_prediction_document

router = APIRouter(tags=["predictions"])


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
