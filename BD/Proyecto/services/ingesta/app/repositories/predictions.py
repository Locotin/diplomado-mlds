from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from app.models import PredictionListResponse, PredictionRecord


def _serialize_prediction(document: dict[str, Any]) -> PredictionRecord:
    serialized = {**document}
    serialized["prediction_id"] = str(serialized.pop("_id"))
    return PredictionRecord.model_validate(serialized)


class PredictionRepository:
    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        self._collection = collection

    async def create(self, document: dict[str, Any]) -> PredictionRecord:
        timestamp = datetime.now(UTC)
        payload = {**document, "created_at": timestamp, "updated_at": timestamp}
        result = await self._collection.insert_one(payload)
        stored = await self._collection.find_one({"_id": result.inserted_id})
        return _serialize_prediction(stored)

    async def create_many(self, documents: list[dict[str, Any]]) -> list[PredictionRecord]:
        if not documents:
            return []

        timestamp = datetime.now(UTC)
        payloads = [{**document, "created_at": timestamp, "updated_at": timestamp} for document in documents]
        result = await self._collection.insert_many(payloads)

        items: list[PredictionRecord] = []
        for payload, inserted_id in zip(payloads, result.inserted_ids, strict=False):
            stored = {**payload, "_id": inserted_id}
            items.append(_serialize_prediction(stored))
        return items

    async def get(self, prediction_id: str) -> PredictionRecord | None:
        if not ObjectId.is_valid(prediction_id):
            return None

        document = await self._collection.find_one({"_id": ObjectId(prediction_id)})
        if not document:
            return None

        return _serialize_prediction(document)

    async def list_recent(self, limit: int) -> PredictionListResponse:
        cursor = self._collection.find().sort("created_at", -1).limit(limit)
        items = [_serialize_prediction(document) async for document in cursor]
        return PredictionListResponse(items=items, total=len(items))
