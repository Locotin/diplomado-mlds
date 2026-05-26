from typing import Any

from app.models import CloudinaryAsset, ModelInferenceResponse


def build_prediction_document(
    *,
    source_id: str,
    cloudinary_asset: CloudinaryAsset,
    inference: ModelInferenceResponse,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "status": "completed",
        "model_provider": inference.model_provider,
        "model_id": inference.model_id,
        "top_emotion": inference.top_emotion,
        "predictions": [prediction.model_dump() for prediction in inference.predictions],
        "cloudinary": cloudinary_asset.model_dump(mode="json"),
        "metadata": metadata,
    }
