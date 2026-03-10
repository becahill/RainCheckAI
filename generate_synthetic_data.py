import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engineering import configure_logging, run_engineering_pipeline


LOGGER = logging.getLogger(__name__)


def generate_synthetic_transport(num_rows: int = 200) -> pd.DataFrame:
    """Create a small synthetic transport dataset for demo purposes."""
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        start="2024-01-01 06:00:00",
        periods=num_rows,
        freq="10min",
        tz="UTC",
    )
    route_ids = rng.integers(low=1, high=4, size=num_rows)

    base_delay = rng.normal(loc=3.0, scale=2.0, size=num_rows)
    rush_hour_boost = np.where((timestamps.hour >= 7) & (timestamps.hour <= 9), 2.0, 0.0)
    delay_minutes = np.clip(base_delay + rush_hour_boost, a_min=0.0, a_max=None)

    transport_df = pd.DataFrame(
        {
            "route_id": route_ids,
            "timestamp": timestamps,
            "delay_minutes": delay_minutes,
        },
    )
    return transport_df


def generate_synthetic_weather(num_rows: int = 100) -> pd.DataFrame:
    """Create a small synthetic hourly weather dataset for demo purposes."""
    rng = np.random.default_rng(123)

    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=num_rows,
        freq="1H",
        tz="UTC",
    )

    precipitation = rng.gamma(shape=1.5, scale=1.0, size=num_rows)
    wind_speed = rng.normal(loc=5.0, scale=2.0, size=num_rows)
    temperature = rng.normal(loc=12.0, scale=8.0, size=num_rows)

    weather_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "precipitation": np.clip(precipitation, a_min=0.0, a_max=None),
            "wind_speed": np.clip(wind_speed, a_min=0.0, a_max=None),
            "temperature": temperature,
        },
    )
    return weather_df


def main() -> None:
    """Generate synthetic data and run the engineering pipeline."""
    configure_logging()

    transport_df = generate_synthetic_transport()
    weather_df = generate_synthetic_weather()

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    transport_path = processed_dir / "transport_clean.csv"
    weather_path = processed_dir / "weather_clean.csv"
    engineered_path = processed_dir / "transport_with_features.csv"

    LOGGER.info("Writing synthetic transport data to %s", transport_path)
    transport_df.to_csv(transport_path, index=False)

    LOGGER.info("Writing synthetic weather data to %s", weather_path)
    weather_df.to_csv(weather_path, index=False)

    LOGGER.info("Running engineering pipeline on synthetic data.")
    run_engineering_pipeline(
        transport_csv=transport_path,
        weather_csv=weather_path,
        output_path=engineered_path,
        route_id_col="route_id",
        timestamp_col="timestamp",
        delay_col="delay_minutes",
    )

    LOGGER.info("Synthetic data generation and feature engineering complete.")


if __name__ == "__main__":
    main()

