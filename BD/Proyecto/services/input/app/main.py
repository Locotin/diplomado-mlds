from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.clients.ingesta_client import IngestaClient
from app.config import get_settings
from app.models import HealthResponse
from app.routers.uploads import router as uploads_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ingesta_client = IngestaClient(base_url=settings.ingesta_service_url)
    yield
    await app.state.ingesta_client.close()


app = FastAPI(title=f"{settings.project_name}-input", lifespan=lifespan)
app.include_router(uploads_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="input")


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.input_port, reload=False)


if __name__ == "__main__":
    main()
