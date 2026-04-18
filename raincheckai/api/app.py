"""FastAPI entrypoint for RainCheckAI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from raincheckai.api.schemas import HealthResponse, MetricsResponse, PredictRequest, PredictResponse
from raincheckai.inference import InferenceService
from raincheckai.logging_utils import configure_logging, reset_request_id, set_request_id
from raincheckai.telemetry import TelemetryCollector

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared application state."""
    configure_logging()
    telemetry = TelemetryCollector()
    service = InferenceService(telemetry=telemetry)
    service.load()
    app.state.telemetry = telemetry
    app.state.inference_service = service
    yield


def create_app() -> FastAPI:
    """Create a configured FastAPI application."""
    app = FastAPI(
        title="RainCheckAI",
        description="Production inference API for transit delay forecasting.",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Attach a request identifier and HTTP latency telemetry to each request."""
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = request_id
        token = set_request_id(request_id)
        started_at = perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = (perf_counter() - started_at) * 1000.0
            telemetry: TelemetryCollector = request.app.state.telemetry
            telemetry.record_latency(
                "http_request_latency_ms",
                duration_ms,
                tags={"path": request.url.path, "method": request.method},
            )
            reset_request_id(token)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Return a stable JSON payload for unexpected errors."""
        del exc
        LOGGER.exception(
            "Unhandled application exception.",
            extra={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(
            status_code=500,
            content={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "detail": "Internal server error.",
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return application health state."""
        service: InferenceService = app.state.inference_service
        return HealthResponse(
            status="ok",
            model_loaded=service.model_loaded,
            model_version=service.model_version,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        """Return an in-process telemetry snapshot."""
        telemetry: TelemetryCollector = app.state.telemetry
        return MetricsResponse(metrics=telemetry.snapshot())

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        """Score a realtime prediction request."""
        service: InferenceService = app.state.inference_service
        result = service.predict(
            context=payload.to_domain(),
            request_id=request.state.request_id,
        )
        return PredictResponse(
            request_id=result.request_id,
            predicted_delay_minutes=result.predicted_delay_minutes,
            degraded=result.degraded,
            degradation_reasons=list(result.degradation_reasons),
            model_version=result.model_version,
        )

    return app


app = create_app()
