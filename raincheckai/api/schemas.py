"""Pydantic request and response schemas for RainCheckAI."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from raincheckai.contracts import EventContext, PredictionContext, TransitContext, WeatherContext


class TransitRequest(BaseModel):
    """API payload describing the transit state to score."""

    model_config = ConfigDict(str_strip_whitespace=True)

    route_id: str
    observed_at: datetime
    stop_id: str | None = None
    city_zone: str | None = None
    observed_delay_minutes: float | None = Field(default=None, ge=-10.0, le=240.0)
    historical_delay_minutes: list[float] = Field(default_factory=list)
    scheduled_headway_minutes: float | None = Field(default=None, ge=0.0, le=180.0)
    service_alert_level: str | None = None

    def to_domain(self) -> TransitContext:
        """Convert the API schema into the internal domain contract."""
        return TransitContext(
            route_id=self.route_id,
            observed_at=self.observed_at,
            stop_id=self.stop_id,
            city_zone=self.city_zone,
            observed_delay_minutes=self.observed_delay_minutes,
            historical_delay_minutes=tuple(self.historical_delay_minutes),
            scheduled_headway_minutes=self.scheduled_headway_minutes,
            service_alert_level=self.service_alert_level,
        )


class WeatherRequest(BaseModel):
    """Optional realtime weather context."""

    model_config = ConfigDict(str_strip_whitespace=True)

    precipitation_mm: float | None = Field(default=None, ge=0.0, le=250.0)
    wind_speed_kph: float | None = Field(default=None, ge=0.0, le=180.0)
    temperature_c: float | None = Field(default=None, ge=-40.0, le=55.0)
    visibility_km: float | None = Field(default=None, ge=0.0, le=30.0)

    def to_domain(self) -> WeatherContext:
        """Convert the API schema into the internal weather contract."""
        return WeatherContext(
            precipitation_mm=self.precipitation_mm,
            wind_speed_kph=self.wind_speed_kph,
            temperature_c=self.temperature_c,
            visibility_km=self.visibility_km,
        )


class EventRequest(BaseModel):
    """Optional realtime event context."""

    model_config = ConfigDict(str_strip_whitespace=True)

    event_type: str = "none"
    event_severity: str = "none"
    attendance: float | None = Field(default=None, ge=0.0, le=500_000.0)
    is_active: bool = False

    def to_domain(self) -> EventContext:
        """Convert the API schema into the internal event contract."""
        return EventContext(
            event_type=self.event_type,
            event_severity=self.event_severity,
            attendance=self.attendance,
            is_active=self.is_active,
        )


class PredictRequest(BaseModel):
    """Prediction request body."""

    transit: TransitRequest
    weather: WeatherRequest | None = None
    event: EventRequest | None = None

    def to_domain(self) -> PredictionContext:
        """Convert the API schema into the internal prediction contract."""
        return PredictionContext(
            transit=self.transit.to_domain(),
            weather=self.weather.to_domain() if self.weather is not None else None,
            event=self.event.to_domain() if self.event is not None else None,
        )


class PredictResponse(BaseModel):
    """Prediction response body."""

    request_id: str
    predicted_delay_minutes: float
    degraded: bool
    degradation_reasons: list[str]
    model_version: str | None = None


class HealthResponse(BaseModel):
    """Health-check response body."""

    status: str
    model_loaded: bool
    model_version: str | None = None


class MetricsResponse(BaseModel):
    """Serialized telemetry snapshot."""

    metrics: dict[str, Any]
