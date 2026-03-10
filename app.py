import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import (
    DEFAULT_FEATURE_NAMES_PATH,
    DEFAULT_MODEL_PATH,
    configure_logging,
    load_model_artifacts,
    predict_delay_from_event,
)


LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context for loading model artifacts."""
    configure_logging()
    model_path: Path = DEFAULT_MODEL_PATH
    feature_names_path: Path = DEFAULT_FEATURE_NAMES_PATH

    if not model_path.exists() or not feature_names_path.exists():
        LOGGER.warning(
            "Model artifacts not found at %s and %s. "
            "Ensure train_model.py has been run before serving predictions.",
            model_path,
            feature_names_path,
        )
        yield
        return

    model, feature_names = load_model_artifacts(model_path, feature_names_path)
    app.state.model = model
    app.state.feature_names = feature_names
    LOGGER.info("Model and feature names loaded into application state.")

    yield


app = FastAPI(
    title="RainCheckAI",
    description="Predict public transport delays using weather-aware ML models.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Request body for delay prediction.

    The `features` field should contain numeric values keyed by the
    model's feature names (for example, `sin_hour`, `cos_hour`,
    `prev_stop_delay_minutes`, and weather features used during training).
    """

    features: Dict[str, Any]


class PredictResponse(BaseModel):
    """Response body for delay prediction."""

    delay_minutes: float


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Return a predicted delay in minutes for a given feature payload."""
    if not hasattr(app.state, "model") or not hasattr(app.state, "feature_names"):
        raise RuntimeError(
            "Model artifacts are not loaded. "
            "Run train_model.py and restart the API server.",
        )

    prediction = predict_delay_from_event(
        model=app.state.model,
        feature_names=app.state.feature_names,
        event=request.features,
    )
    return PredictResponse(delay_minutes=prediction)


if __name__ == "__main__":
    import uvicorn  # pyright: ignore[reportMissingImports]

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )