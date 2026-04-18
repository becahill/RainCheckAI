"""Integration tests for RainCheckAI model training and serving."""

from __future__ import annotations

from datetime import UTC, datetime

from generate_synthetic_data import (
    generate_synthetic_events,
    generate_synthetic_transport,
    generate_synthetic_weather,
)
from raincheckai.config import ArtifactPaths, TrainingConfig
from raincheckai.contracts import (
    DataBundle,
    EventContext,
    PredictionContext,
    TransitContext,
    WeatherContext,
)
from raincheckai.feature_engineering import engineer_training_dataset
from raincheckai.inference import InferenceService
from raincheckai.ingestion import clean_event_data, clean_transport_data, clean_weather_data
from raincheckai.telemetry import TelemetryCollector
from raincheckai.training import save_model_bundle, train_model_bundle


def test_training_bundle_can_be_loaded_for_online_inference(tmp_path) -> None:
    """A trained bundle should round-trip through persistence and serve predictions."""
    transport = clean_transport_data(generate_synthetic_transport(num_rows=120))
    weather = clean_weather_data(generate_synthetic_weather(num_rows=48))
    events = clean_event_data(generate_synthetic_events(num_rows=6))
    engineered = engineer_training_dataset(
        bundle=DataBundle(transport=transport, weather=weather, events=events)
    )
    config = TrainingConfig(n_splits=3, search_iterations=2)

    bundle = train_model_bundle(engineered, config=config)
    artifact_paths = ArtifactPaths(root_dir=tmp_path)
    save_model_bundle(bundle, artifact_paths=artifact_paths)

    service = InferenceService(
        telemetry=TelemetryCollector(),
        artifact_paths=artifact_paths,
    )
    service.load()

    result = service.predict(
        context=PredictionContext(
            transit=TransitContext(
                route_id="BLUE_LINE",
                observed_at=datetime(2026, 4, 18, 8, 30, tzinfo=UTC),
                stop_id="STOP_101",
                city_zone="downtown",
                observed_delay_minutes=5.0,
                historical_delay_minutes=(1.0, 3.0, 4.0),
                scheduled_headway_minutes=10.0,
                service_alert_level="normal",
            ),
            weather=WeatherContext(
                precipitation_mm=3.0,
                wind_speed_kph=16.0,
                temperature_c=12.0,
                visibility_km=7.0,
            ),
            event=EventContext(
                event_type="sports",
                event_severity="medium",
                attendance=10000.0,
                is_active=True,
            ),
        ),
        request_id="integration-test",
    )

    assert service.model_loaded is True
    assert result.degraded is False
    assert result.model_version == bundle.metadata.model_version
    assert result.predicted_delay_minutes >= 0.0
