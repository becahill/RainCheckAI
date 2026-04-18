"""Online inference service and graceful degradation logic."""

from __future__ import annotations

import logging
from time import perf_counter

from raincheckai.config import ArtifactPaths, FeatureConfig
from raincheckai.contracts import (
    EventContext,
    ModelBundle,
    PredictionContext,
    PredictionResult,
    WeatherContext,
)
from raincheckai.errors import ArtifactNotAvailableError
from raincheckai.feature_engineering import build_inference_frame
from raincheckai.telemetry import TelemetryCollector
from raincheckai.training import load_model_bundle

LOGGER = logging.getLogger(__name__)


class InferenceService:
    """Serve online predictions with resilient fallback behavior."""

    def __init__(
        self,
        telemetry: TelemetryCollector,
        artifact_paths: ArtifactPaths | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        """Initialize an inference service."""
        self.telemetry = telemetry
        self.artifact_paths = artifact_paths or ArtifactPaths()
        self.feature_config = feature_config or FeatureConfig()
        self.model_bundle: ModelBundle | None = None

    @property
    def model_loaded(self) -> bool:
        """Return whether a trained model bundle is available."""
        return self.model_bundle is not None

    @property
    def model_version(self) -> str | None:
        """Return the loaded model version, if available."""
        if self.model_bundle is None:
            return None
        return self.model_bundle.metadata.model_version

    def load(self) -> None:
        """Load model artifacts if they exist."""
        try:
            self.model_bundle = load_model_bundle(self.artifact_paths)
            self.telemetry.set_gauge("model_loaded", 1.0)
            LOGGER.info(
                "Loaded model bundle.",
                extra={"model_version": self.model_bundle.metadata.model_version},
            )
        except ArtifactNotAvailableError:
            self.model_bundle = None
            self.telemetry.set_gauge("model_loaded", 0.0)
            LOGGER.warning(
                (
                    "Model bundle unavailable. The service will use fallback heuristics "
                    "until artifacts exist."
                ),
                extra={"artifact_path": self.artifact_paths.model_bundle_path},
            )

    def _apply_context_fallbacks(
        self,
        context: PredictionContext,
    ) -> tuple[PredictionContext, list[str]]:
        """Inject fallback weather and event contexts when upstream inputs are absent."""
        degradation_reasons: list[str] = []
        weather = context.weather
        event = context.event

        if weather is None:
            weather = WeatherContext(
                precipitation_mm=0.0,
                wind_speed_kph=15.0,
                temperature_c=15.0,
                visibility_km=10.0,
                source="fallback-default",
            )
            degradation_reasons.append("weather_context_missing")
        if event is None:
            event = EventContext(
                event_type="none",
                event_severity="none",
                attendance=0.0,
                is_active=False,
                source="fallback-default",
            )
            degradation_reasons.append("event_context_missing")

        enriched_context = PredictionContext(
            transit=context.transit,
            weather=weather,
            event=event,
        )
        return enriched_context, degradation_reasons

    def _fallback_prediction(self, feature_frame: object) -> float:
        """Produce a conservative heuristic prediction when the model is unavailable."""
        if not hasattr(feature_frame, "iloc"):
            return 0.0
        baseline = (
            self.model_bundle.metadata.baseline_delay_minutes
            if self.model_bundle is not None
            else 4.0
        )
        prev_delay = float(feature_frame.iloc[0]["prev_delay_minutes"])
        weather_risk = float(feature_frame.iloc[0]["weather_risk_index"])
        event_pressure = float(feature_frame.iloc[0]["event_attendance_log"])
        prediction = (
            (baseline * 0.4) + (prev_delay * 0.55) + (weather_risk * 0.5) + (event_pressure * 0.3)
        )
        return max(prediction, 0.0)

    def predict(
        self,
        context: PredictionContext,
        request_id: str,
    ) -> PredictionResult:
        """Run inference and return a structured prediction result."""
        started_at = perf_counter()
        enriched_context, degradation_reasons = self._apply_context_fallbacks(context)
        feature_frame = build_inference_frame(
            context=enriched_context,
            feature_config=self.feature_config,
        )

        try:
            if self.model_bundle is None:
                raise ArtifactNotAvailableError("No trained model bundle is loaded.")
            prediction = float(self.model_bundle.pipeline.predict(feature_frame)[0])
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ArtifactNotAvailableError):
                degradation_reasons.append("model_bundle_unavailable")
            else:
                degradation_reasons.append("model_inference_failed")
                LOGGER.exception("Model inference failed.")
            prediction = self._fallback_prediction(feature_frame)

        duration_ms = (perf_counter() - started_at) * 1000.0
        self.telemetry.increment_counter("inference_requests_total")
        if degradation_reasons:
            self.telemetry.increment_counter("inference_degraded_total")
        self.telemetry.record_latency("inference_latency_ms", duration_ms)

        return PredictionResult(
            predicted_delay_minutes=max(prediction, 0.0),
            degraded=bool(degradation_reasons),
            degradation_reasons=tuple(sorted(set(degradation_reasons))),
            model_version=self.model_version,
            request_id=request_id,
        )
