"""Tests for RainCheckAI feature engineering."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from raincheckai.contracts import (
    DataBundle,
    EventContext,
    PredictionContext,
    TransitContext,
    WeatherContext,
)
from raincheckai.feature_engineering import (
    add_cyclical_time_encoding,
    build_inference_frame,
    engineer_training_dataset,
)


def test_add_cyclical_time_encoding_hour_wraparound() -> None:
    """Hour 23 and hour 0 should remain close on the unit circle."""
    timestamps = [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T23:00:00Z"),
    ]
    df = pd.DataFrame({"timestamp": timestamps})

    engineered = add_cyclical_time_encoding(df, timestamp_col="timestamp")

    assert "sin_hour" in engineered.columns
    assert "cos_hour" in engineered.columns

    v0 = engineered.loc[0, ["sin_hour", "cos_hour"]].to_numpy(dtype=float)
    v23 = engineered.loc[1, ["sin_hour", "cos_hour"]].to_numpy(dtype=float)
    assert float(np.linalg.norm(v0 - v23)) < 0.5


def test_engineer_training_dataset_creates_weather_and_event_features() -> None:
    """Offline feature engineering should emit the full training contract."""
    transport = pd.DataFrame(
        {
            "route_id": ["BLUE_LINE", "BLUE_LINE", "GREEN_LINE"],
            "stop_id": ["STOP_101", "STOP_102", "STOP_205"],
            "city_zone": ["downtown", "downtown", "midtown"],
            "timestamp": pd.to_datetime(
                [
                    "2026-04-18T08:00:00Z",
                    "2026-04-18T08:15:00Z",
                    "2026-04-18T09:00:00Z",
                ],
                utc=True,
            ),
            "service_alert_level": ["normal", "minor", "normal"],
            "scheduled_headway_minutes": [10.0, 10.0, 12.0],
            "delay_minutes": [4.0, 7.0, 3.0],
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-04-18T07:00:00Z", "2026-04-18T08:00:00Z"],
                utc=True,
            ),
            "precipitation_mm": [0.0, 6.0],
            "wind_speed_kph": [12.0, 20.0],
            "temperature_c": [15.0, 8.0],
            "visibility_km": [10.0, 4.0],
        }
    )
    events = pd.DataFrame(
        {
            "start_timestamp": pd.to_datetime(["2026-04-18T07:30:00Z"], utc=True),
            "end_timestamp": pd.to_datetime(["2026-04-18T09:30:00Z"], utc=True),
            "event_type": ["concert"],
            "event_severity": ["high"],
            "attendance": [15000.0],
        }
    )

    engineered = engineer_training_dataset(
        DataBundle(transport=transport, weather=weather, events=events)
    )

    assert {"prev_delay_minutes", "weather_risk_index", "event_attendance_log"} <= set(
        engineered.columns
    )
    assert engineered["is_event_active"].max() == 1
    assert engineered["event_type"].isin(["concert", "none"]).all()


def test_build_inference_frame_matches_feature_contract() -> None:
    """Realtime feature assembly should create the model-ready contract."""
    context = PredictionContext(
        transit=TransitContext(
            route_id="BLUE_LINE",
            observed_at=datetime(2026, 4, 18, 8, 30, tzinfo=UTC),
            stop_id="STOP_101",
            city_zone="downtown",
            observed_delay_minutes=6.0,
            historical_delay_minutes=(2.0, 4.0, 5.0),
            scheduled_headway_minutes=10.0,
            service_alert_level="minor",
        ),
        weather=WeatherContext(
            precipitation_mm=4.0,
            wind_speed_kph=18.0,
            temperature_c=11.0,
            visibility_km=6.0,
        ),
        event=EventContext(
            event_type="concert",
            event_severity="medium",
            attendance=12000.0,
            is_active=True,
        ),
    )

    frame = build_inference_frame(context)

    assert frame.shape[0] == 1
    assert frame.loc[0, "prev_delay_minutes"] == 6.0
    assert frame.loc[0, "event_type"] == "concert"
    assert frame.loc[0, "service_alert_level"] == "minor"
