from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.config import get_settings
from app.models import HealthResponse
from app.routers.inference import router as inference_router
from app.services.inference_pipeline import EmotionInferenceService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = EmotionInferenceService(settings)
    yield


app = FastAPI(title=f"{settings.project_name}-modelo", lifespan=lifespan)
app.include_router(inference_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="modelo")


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.modelo_port, reload=False)


if __name__ == "__main__":
    main()
