import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure basic logging for the module."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def round_timestamp_to_hour(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Round timestamps to the nearest hour to create a join key.

    Parameters
    ----------
    df:
        Input DataFrame with a timestamp column.
    timestamp_col:
        Name of the timestamp column (assumed to be datetime-like).

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional ``time_key`` column.
    """
    df = df.copy()
    if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
        raise TypeError(f"Column '{timestamp_col}' must be datetime64[ns], got {df[timestamp_col].dtype}")

    # Round to the nearest hour using pandas' dt.round.
    LOGGER.info("Creating hourly time_key from '%s'", timestamp_col)
    df["time_key"] = df[timestamp_col].dt.round("1H")
    return df


def left_join_transport_weather(
    transport_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    transport_timestamp_col: str = "timestamp",
    weather_timestamp_col: str = "timestamp",
    weather_suffix: str = "_weather",
) -> pd.DataFrame:
    """Left-join transport and weather data on a rounded 1‑hour time key.

    Parameters
    ----------
    transport_df:
        Cleaned transport DataFrame.
    weather_df:
        Cleaned weather DataFrame.
    transport_timestamp_col:
        Timestamp column in the transport DataFrame.
    weather_timestamp_col:
        Timestamp column in the weather DataFrame.
    weather_suffix:
        Suffix to apply to overlapping weather columns after the join.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame with weather features aligned to transport events.
    """
    LOGGER.info("Preparing time keys for transport and weather data.")
    transport_with_key = round_timestamp_to_hour(transport_df, timestamp_col=transport_timestamp_col)
    weather_with_key = round_timestamp_to_hour(weather_df, timestamp_col=weather_timestamp_col)

    LOGGER.info("Performing left join on 'time_key'.")
    merged = pd.merge(
        transport_with_key,
        weather_with_key.drop(columns=[weather_timestamp_col]),
        how="left",
        on="time_key",
        suffixes=("", weather_suffix),
    )
    LOGGER.info("Merged dataset has %d rows and %d columns", len(merged), merged.shape[1])
    return merged


def add_cyclical_time_encoding(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add cyclical encoding for hour of day using sine and cosine transforms.

    This preserves the circular nature of time where 23:00 is close to 00:00.

    Parameters
    ----------
    df:
        Input DataFrame with a timestamp column.
    timestamp_col:
        Name of the timestamp column (datetime64[ns]).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``sin_hour`` and ``cos_hour`` features.
    """
    df = df.copy()
    if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
        raise TypeError(f"Column '{timestamp_col}' must be datetime64[ns], got {df[timestamp_col].dtype}")

    LOGGER.info("Adding cyclical hour-of-day encoding from '%s'.", timestamp_col)
    hours = df[timestamp_col].dt.hour.astype(float)
    radians = 2 * np.pi * hours / 24.0

    df["sin_hour"] = np.sin(radians)
    df["cos_hour"] = np.cos(radians)
    return df


def add_previous_stop_delay(
    df: pd.DataFrame,
    route_id_col: str = "route_id",
    timestamp_col: str = "timestamp",
    delay_col: str = "delay_minutes",
    new_col: str = "prev_stop_delay_minutes",
) -> pd.DataFrame:
    """Create a lag feature representing the previous stop's delay per route.

    The function assumes that delays propagate along a route over time,
    so it groups by route and sorts chronologically before computing the lag.

    Parameters
    ----------
    df:
        Input DataFrame containing route and delay information.
    route_id_col:
        Column identifying the route or line.
    timestamp_col:
        Timestamp column used for chronological ordering.
    delay_col:
        Column containing the current stop delay (target-like variable).
    new_col:
        Name of the lag feature column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional lag feature column.
    """
    df = df.copy()
    LOGGER.info(
        "Creating lag feature '%s' from '%s' grouped by '%s'.",
        new_col,
        delay_col,
        route_id_col,
    )

    df = df.sort_values(by=[route_id_col, timestamp_col])
    df[new_col] = (
        df.groupby(route_id_col, sort=False)[delay_col]
        .shift(1)
        .astype(float)
    )
    return df


def engineer_features(
    transport_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    route_id_col: str = "route_id",
    timestamp_col: str = "timestamp",
    delay_col: str = "delay_minutes",
) -> pd.DataFrame:
    """Run the complete temporal and feature engineering pipeline.

    Parameters
    ----------
    transport_df:
        Cleaned transport DataFrame.
    weather_df:
        Cleaned weather DataFrame.
    route_id_col:
        Column identifying each route.
    timestamp_col:
        Timestamp column used for temporal alignment.
    delay_col:
        Delay column in minutes (regression target).

    Returns
    -------
    pd.DataFrame
        Fully engineered DataFrame ready for model training.
    """
    merged = left_join_transport_weather(
        transport_df=transport_df,
        weather_df=weather_df,
        transport_timestamp_col=timestamp_col,
        weather_timestamp_col=timestamp_col,
    )
    merged = add_cyclical_time_encoding(merged, timestamp_col=timestamp_col)
    merged = add_previous_stop_delay(
        merged,
        route_id_col=route_id_col,
        timestamp_col=timestamp_col,
        delay_col=delay_col,
    )
    return merged


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    ignore_cols: List[str] | None = None,
) -> List[str]:
    """Select candidate feature columns for model training.

    Parameters
    ----------
    df:
        Engineered DataFrame.
    target_col:
        Name of the target column to exclude.
    ignore_cols:
        Additional columns to exclude from the feature set.

    Returns
    -------
    List[str]
        List of feature column names.
    """
    ignore_cols = (ignore_cols or []) + [target_col]
    feature_cols: List[str] = [
        c
        for c in df.columns
        if c not in ignore_cols and not df[c].dtype == "O"
    ]
    LOGGER.info("Selected %d numeric feature columns for modelling.", len(feature_cols))
    return feature_cols


def run_engineering_pipeline(
    transport_csv: Path,
    weather_csv: Path,
    output_path: Path,
    route_id_col: str = "route_id",
    timestamp_col: str = "timestamp",
    delay_col: str = "delay_minutes",
) -> None:
    """Convenience wrapper to run feature engineering from CSV inputs.

    Parameters
    ----------
    transport_csv:
        Path to cleaned transport CSV (output of ``ingest_data.py``).
    weather_csv:
        Path to cleaned weather CSV.
    output_path:
        Path where the engineered dataset will be written.
    route_id_col:
        Column identifying routes.
    timestamp_col:
        Timestamp column name.
    delay_col:
        Delay column name.
    """
    LOGGER.info("Loading cleaned transport data from %s", transport_csv)
    transport_df = pd.read_csv(transport_csv, parse_dates=[timestamp_col])

    LOGGER.info("Loading cleaned weather data from %s", weather_csv)
    weather_df = pd.read_csv(weather_csv, parse_dates=[timestamp_col])

    engineered_df = engineer_features(
        transport_df=transport_df,
        weather_df=weather_df,
        route_id_col=route_id_col,
        timestamp_col=timestamp_col,
        delay_col=delay_col,
    )

    LOGGER.info("Writing engineered dataset to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output_path, index=False)
    LOGGER.info("Feature engineering complete.")


if __name__ == "__main__":
    configure_logging()
    # Lightweight CLI without argparse to keep the focus on modular functions.
    # For portfolio usage, these paths can be edited directly or wired into
    # a separate orchestration script or notebook.
    default_transport = Path("data/processed/transport_clean.csv")
    default_weather = Path("data/processed/weather_clean.csv")
    default_output = Path("data/processed/transport_with_features.csv")

    run_engineering_pipeline(
        transport_csv=default_transport,
        weather_csv=default_weather,
        output_path=default_output,
        route_id_col="route_id",
        timestamp_col="timestamp",
        delay_col="delay_minutes",
    )

