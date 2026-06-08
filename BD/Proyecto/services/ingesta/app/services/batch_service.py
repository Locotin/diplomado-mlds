from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anyio
import dask.bag as db
import dask.dataframe as dd
import httpx
import pandas as pd

from app.models import (
    BatchEmotionSummary,
    BatchManifestRow,
    BatchPredictionFailure,
    BatchPredictionResponse,
)
from app.repositories.predictions import PredictionRepository
from app.services.cloudinary_service import CloudinaryService
from app.services.document_builder import build_prediction_document
from app.services.model_service import ModelServiceClient


@dataclass(frozen=True)
class BatchFilePayload:
    source_id: str
    filename: str
    content_type: str | None
    file_bytes: bytes
    metadata: dict[str, Any]


@dataclass
class BatchProcessingArtifacts:
    manifest_preview: list[dict[str, Any]]
    emotion_summary: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    failures: list[dict[str, Any]]


class BatchPredictionService:
    def __init__(
        self,
        *,
        cloudinary_service: CloudinaryService,
        model_service: ModelServiceClient,
        dask_workers: int,
        dask_partitions: int,
        manifest_preview_rows: int,
    ) -> None:
        self._cloudinary_service = cloudinary_service
        self._model_service = model_service
        self._dask_workers = dask_workers
        self._dask_partitions = dask_partitions
        self._manifest_preview_rows = manifest_preview_rows

    async def process_batch(
        self,
        *,
        batch_id: str,
        items: list[BatchFilePayload],
        repository: PredictionRepository,
        validation_failures: list[BatchPredictionFailure] | None = None,
    ) -> BatchPredictionResponse:
        artifacts = await anyio.to_thread.run_sync(self._process_batch_sync, items)
        stored_items = await repository.create_many(artifacts.documents)
        failures = [BatchPredictionFailure.model_validate(item) for item in artifacts.failures]
        failures.extend(validation_failures or [])

        return BatchPredictionResponse(
            batch_id=batch_id,
            total_received=len(items) + len(validation_failures or []),
            total_processed=len(stored_items),
            total_failed=len(failures),
            manifest_preview=[BatchManifestRow.model_validate(row) for row in artifacts.manifest_preview],
            emotion_summary=[BatchEmotionSummary.model_validate(row) for row in artifacts.emotion_summary],
            items=stored_items,
            failures=failures,
        )

    def _process_batch_sync(self, items: list[BatchFilePayload]) -> BatchProcessingArtifacts:
        manifest_preview = self._build_manifest_preview(items)
        if not items:
            return BatchProcessingArtifacts(
                manifest_preview=manifest_preview,
                emotion_summary=[],
                documents=[],
                failures=[],
            )

        partitions = min(max(1, len(items)), self._dask_partitions)
        results = (
            db.from_sequence(items, npartitions=partitions)
            .map(self._process_single_item)
            .compute(scheduler="threads", num_workers=self._dask_workers)
        )

        documents: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for result in results:
            if result["status"] == "ok":
                documents.append(result["document"])
            else:
                failures.append(result["failure"])

        emotion_summary = self._build_emotion_summary(documents)
        return BatchProcessingArtifacts(
            manifest_preview=manifest_preview,
            emotion_summary=emotion_summary,
            documents=documents,
            failures=failures,
        )

    def _build_manifest_preview(self, items: list[BatchFilePayload]) -> list[dict[str, Any]]:
        if not items:
            return []

        manifest_rows = [
            {
                "source_id": item.source_id,
                "filename": item.filename,
                "content_type": item.content_type,
                "byte_size": len(item.file_bytes),
            }
            for item in items
        ]
        dataframe = pd.DataFrame(manifest_rows)
        dask_frame = dd.from_pandas(
            dataframe,
            npartitions=min(max(1, len(dataframe)), self._dask_partitions),
        )
        preview = dask_frame.head(self._manifest_preview_rows)
        return preview.to_dict(orient="records")

    def _build_emotion_summary(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not documents:
            return []

        dataframe = pd.DataFrame(
            [{"top_emotion": document["top_emotion"]} for document in documents]
        )
        dask_frame = dd.from_pandas(
            dataframe,
            npartitions=min(max(1, len(dataframe)), self._dask_partitions),
        )
        summary = (
            dask_frame.groupby("top_emotion")
            .size()
            .compute()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        return summary.to_dict(orient="records")

    def _process_single_item(self, item: BatchFilePayload) -> dict[str, Any]:
        try:
            cloudinary_asset = self._cloudinary_service.upload_image_sync(
                item.file_bytes,
                item.filename,
                item.source_id,
            )
            inference = self._model_service.infer_sync(
                source_id=item.source_id,
                image_url=str(cloudinary_asset.secure_url),
            )
            document = build_prediction_document(
                source_id=item.source_id,
                cloudinary_asset=cloudinary_asset,
                inference=inference,
                metadata=item.metadata,
            )
            return {"status": "ok", "document": document}
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text or "Model service rejected the request."
        except httpx.HTTPError:
            detail = "Unable to reach the model service."
        except Exception as exc:  # noqa: BLE001 - batch should report per-file failures
            detail = str(exc)

        return {
            "status": "error",
            "failure": {
                "filename": item.filename,
                "source_id": item.source_id,
                "detail": detail,
            },
        }
