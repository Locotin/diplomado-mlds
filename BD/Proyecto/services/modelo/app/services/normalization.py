from app.models import EmotionScore


def normalize_predictions(raw_predictions: list[dict], limit: int) -> list[EmotionScore]:
    normalized: list[EmotionScore] = []
    for item in raw_predictions[:limit]:
        label = str(item.get("label", "unknown"))
        score = float(item.get("score", 0.0))
        normalized.append(EmotionScore(label=label, score=score))
    return normalized
