"""Typed domain contracts used across RainCheckAI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, TypeAlias

import pandas as pd
from sklearn.pipeline import Pipeline

JSONScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class TransitContext:
    """Realtime transit context used for online inference."""

    route_id: str
    observed_at: datetime
    stop_id: str | None = None
    city_zone: str | None = None
    observed_delay_minutes: float | None = None
    historical_delay_minutes: tuple[float, ...] = ()
    scheduled_headway_minutes: float | None = None
    service_alert_level: str | None = None


@dataclass(frozen=True, slots=True)
class WeatherContext:
    """Weather attributes available at inference time."""

    precipitation_mm: float | None = None
    wind_speed_kph: float | None = None
    temperature_c: float | None = None
    visibility_km: float | None = None
    source: str = "request"


@dataclass(frozen=True, slots=True)
class EventContext:
    """Event attributes available at inference time."""

    event_type: str = "none"
    event_severity: str = "none"
    attendance: float | None = None
    is_active: bool = False
    source: str = "request"


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Complete inference context for a single prediction request."""

    transit: TransitContext
    weather: WeatherContext | None = None
    event: EventContext | None = None


@dataclass(frozen=True)
class DataBundle:
    """Container for cleaned transport, weather, and event frames."""

    transport: pd.DataFrame
    weather: pd.DataFrame
    events: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Serializable metadata about a trained model bundle."""

    model_name: str
    model_version: str
    trained_at_utc: str
    random_seed: int
    target_column: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    cv_rmse: float
    cv_mae: float
    training_rmse: float
    baseline_delay_minutes: float
    training_rows: int
    best_params: dict[str, JSONScalar]

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ModelBundle:
    """Serialized model bundle used for offline and online inference."""

    pipeline: Pipeline
    metadata: ModelMetadata


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Response object returned by the inference service."""

    predicted_delay_minutes: float
    degraded: bool
    degradation_reasons: tuple[str, ...]
    model_version: str | None
    request_id: str
