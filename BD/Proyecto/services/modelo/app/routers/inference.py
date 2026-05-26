import httpx
from fastapi import APIRouter, HTTPException, Request, status

from app.models import ModelInferenceRequest, ModelInferenceResponse

router = APIRouter(tags=["inference"])


@router.post("/infer", response_model=ModelInferenceResponse)
async def infer_emotions(request: Request, payload: ModelInferenceRequest) -> ModelInferenceResponse:
    inference_service = request.app.state.inference_service
    try:
        return await inference_service.infer_from_url(
            source_id=payload.source_id,
            image_url=str(payload.image_url),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to download the source image for inference.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
