"""Generate synthetic datasets for end-to-end RainCheckAI demos."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from raincheckai.contracts import DataBundle
from raincheckai.feature_engineering import engineer_training_dataset
from raincheckai.ingestion import load_data_bundle, write_clean_data_bundle
from raincheckai.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def generate_synthetic_weather(num_rows: int = 96) -> pd.DataFrame:
    """Create a synthetic hourly weather feed."""
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2026-04-15T00:00:00Z", periods=num_rows, freq="1h")
    precipitation = rng.gamma(shape=1.8, scale=1.5, size=num_rows)
    wind_speed = np.clip(rng.normal(loc=18.0, scale=7.0, size=num_rows), 0.0, None)
    temperature = rng.normal(loc=14.0, scale=6.0, size=num_rows)
    visibility = np.clip(12.0 - (precipitation * 0.7), 0.5, 15.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "precipitation_mm": precipitation,
            "wind_speed_kph": wind_speed,
            "temperature_c": temperature,
            "visibility_km": visibility,
        },
    )


def generate_synthetic_events(num_rows: int = 12) -> pd.DataFrame:
    """Create a synthetic event calendar."""
    rng = np.random.default_rng(11)
    event_starts = pd.date_range("2026-04-15T12:00:00Z", periods=num_rows, freq="8h")
    event_types = np.array(["concert", "sports", "festival", "conference"])
    severities = np.array(["low", "medium", "high"])
    durations = rng.integers(low=2, high=5, size=num_rows)
    attendance = rng.integers(low=3000, high=35000, size=num_rows)
    return pd.DataFrame(
        {
            "start_timestamp": event_starts,
            "end_timestamp": event_starts + pd.to_timedelta(durations, unit="h"),
            "event_type": rng.choice(event_types, size=num_rows),
            "event_severity": rng.choice(severities, size=num_rows),
            "attendance": attendance,
        },
    )


def generate_synthetic_transport(num_rows: int = 480) -> pd.DataFrame:
    """Create synthetic transport observations with realistic delay structure."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-04-15T05:00:00Z", periods=num_rows, freq="15min")
    route_ids = rng.choice(["BLUE_LINE", "GREEN_LINE", "RED_LINE"], size=num_rows)
    stop_ids = rng.choice(["STOP_101", "STOP_102", "STOP_205", "STOP_410"], size=num_rows)
    city_zones = rng.choice(["downtown", "midtown", "suburban"], size=num_rows, p=[0.5, 0.3, 0.2])
    headways = rng.choice([8.0, 10.0, 12.0], size=num_rows)
    service_alerts = rng.choice(["normal", "minor", "major"], size=num_rows, p=[0.82, 0.14, 0.04])

    peak_window = ((timestamps.hour >= 7) & (timestamps.hour <= 9)) | (
        (timestamps.hour >= 16) & (timestamps.hour <= 19)
    )
    peak_boost = np.where(peak_window, 2.5, 0.0)
    route_bias = np.where(route_ids == "RED_LINE", 1.2, 0.0)
    alert_boost = np.select(
        [service_alerts == "normal", service_alerts == "minor", service_alerts == "major"],
        [0.0, 2.0, 5.0],
        default=0.0,
    )
    delay_minutes = np.clip(
        rng.normal(loc=3.5, scale=1.8, size=num_rows) + peak_boost + route_bias + alert_boost,
        0.0,
        None,
    )
    return pd.DataFrame(
        {
            "route_id": route_ids,
            "stop_id": stop_ids,
            "city_zone": city_zones,
            "timestamp": timestamps,
            "scheduled_headway_minutes": headways,
            "service_alert_level": service_alerts,
            "delay_minutes": delay_minutes,
        },
    )


def main() -> None:
    """Generate synthetic raw, cleaned, and engineered datasets."""
    configure_logging()
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    transport_raw = generate_synthetic_transport()
    weather_raw = generate_synthetic_weather()
    events_raw = generate_synthetic_events()

    transport_raw_path = raw_dir / "transport_raw.csv"
    weather_raw_path = raw_dir / "weather_raw.csv"
    events_raw_path = raw_dir / "events_raw.csv"
    transport_raw.to_csv(transport_raw_path, index=False)
    weather_raw.to_csv(weather_raw_path, index=False)
    events_raw.to_csv(events_raw_path, index=False)
    LOGGER.info(
        "Wrote raw synthetic datasets.",
        extra={
            "transport_raw_path": transport_raw_path,
            "weather_raw_path": weather_raw_path,
            "events_raw_path": events_raw_path,
        },
    )

    bundle = load_data_bundle(
        transport_path=transport_raw_path,
        weather_path=weather_raw_path,
        events_path=events_raw_path,
    )
    write_clean_data_bundle(bundle=bundle, output_dir=processed_dir)

    engineered = engineer_training_dataset(bundle=DataBundle(**bundle.__dict__))
    engineered_path = processed_dir / "training_features.csv"
    engineered.to_csv(engineered_path, index=False)
    LOGGER.info("Wrote engineered synthetic dataset.", extra={"engineered_path": engineered_path})


if __name__ == "__main__":
    main()
