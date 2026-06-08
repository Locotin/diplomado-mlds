from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import get_settings
from app.models import HealthResponse
from app.repositories.predictions import PredictionRepository
from app.routers.predictions import router as predictions_router
from app.services.batch_service import BatchPredictionService
from app.services.cloudinary_service import CloudinaryService
from app.services.model_service import ModelServiceClient

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
    database = mongo_client[settings.mongodb_database]
    collection = database[settings.mongodb_collection]

    app.state.mongo_client = mongo_client
    app.state.prediction_repository = PredictionRepository(collection)
    app.state.cloudinary_service = CloudinaryService(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        folder=settings.cloudinary_folder,
    )
    app.state.model_service = ModelServiceClient(
        base_url=settings.model_service_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    app.state.batch_prediction_service = BatchPredictionService(
        cloudinary_service=app.state.cloudinary_service,
        model_service=app.state.model_service,
        dask_workers=settings.dask_batch_workers,
        dask_partitions=settings.dask_batch_partitions,
        manifest_preview_rows=settings.dask_manifest_preview_rows,
    )
    yield
    await app.state.model_service.close()
    mongo_client.close()


app = FastAPI(title=f"{settings.project_name}-ingesta", lifespan=lifespan)
app.include_router(predictions_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="ingesta")


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.ingesta_port, reload=False)


if __name__ == "__main__":
    main()
