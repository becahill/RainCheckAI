"""Feature engineering for RainCheckAI training and inference."""

from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from raincheckai.config import FeatureConfig, IngestionConfig, TrainingConfig
from raincheckai.contracts import DataBundle, PredictionContext

LOGGER = logging.getLogger(__name__)


def _ensure_datetime_column(df: pd.DataFrame, column: str) -> None:
    """Validate that a DataFrame column is datetime-like."""
    if not is_datetime64_any_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be datetime-like, got {df[column].dtype}.")


def merge_transport_weather(
    transport_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Merge transport observations with the latest known weather context."""
    left = transport_df.sort_values(timestamp_column).reset_index(drop=True)
    right = weather_df.sort_values(timestamp_column).reset_index(drop=True)
    _ensure_datetime_column(left, timestamp_column)
    _ensure_datetime_column(right, timestamp_column)

    merged = pd.merge_asof(
        left,
        right,
        on=timestamp_column,
        direction="backward",
        tolerance=pd.Timedelta(hours=3),
    )
    for column, default_value in (
        ("precipitation_mm", 0.0),
        ("wind_speed_kph", 15.0),
        ("temperature_c", 15.0),
        ("visibility_km", 10.0),
    ):
        if column not in merged.columns:
            merged[column] = default_value
        merged[column] = merged[column].fillna(default_value)
    return merged


def merge_transport_events(
    transport_df: pd.DataFrame,
    event_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    event_start_column: str = "start_timestamp",
    event_end_column: str = "end_timestamp",
) -> pd.DataFrame:
    """Attach the latest active event context to each transport observation."""
    if event_df.empty:
        merged = transport_df.copy()
        merged["event_type"] = "none"
        merged["event_severity"] = "none"
        merged["attendance"] = 0.0
        merged["is_event_active"] = 0
        return merged

    left = transport_df.sort_values(timestamp_column).reset_index(drop=True)
    right = event_df.sort_values(event_start_column).reset_index(drop=True)
    _ensure_datetime_column(left, timestamp_column)
    _ensure_datetime_column(right, event_start_column)
    _ensure_datetime_column(right, event_end_column)

    merged = pd.merge_asof(
        left,
        right,
        left_on=timestamp_column,
        right_on=event_start_column,
        direction="backward",
        tolerance=pd.Timedelta(hours=12),
    )
    is_active = merged[event_end_column].notna() & (
        merged[timestamp_column] <= merged[event_end_column]
    )
    merged["is_event_active"] = is_active.astype(int)
    merged.loc[~is_active, "event_type"] = "none"
    merged.loc[~is_active, "event_severity"] = "none"
    merged.loc[~is_active, "attendance"] = 0.0
    merged["attendance"] = merged["attendance"].fillna(0.0)
    return merged


def add_temporal_features(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create cyclical and operational time features."""
    featured = df.copy()
    _ensure_datetime_column(featured, timestamp_column)

    timestamps = featured[timestamp_column].dt.tz_convert(timezone.utc)
    hour_fraction = timestamps.dt.hour + (timestamps.dt.minute / 60.0)
    day_of_week = timestamps.dt.dayofweek.astype(float)

    featured["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction / 24.0)
    featured["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction / 24.0)
    featured["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    featured["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    featured["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    featured["is_peak_service_window"] = (
        ((timestamps.dt.hour >= 7) & (timestamps.dt.hour <= 9))
        | ((timestamps.dt.hour >= 16) & (timestamps.dt.hour <= 19))
    ).astype(int)
    return featured


def add_delay_history_features(
    df: pd.DataFrame,
    route_column: str = "route_id",
    delay_column: str = "delay_minutes",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create route-level lag and rolling delay features without target leakage."""
    featured = df.sort_values([route_column, timestamp_column]).copy()
    grouped_delay = featured.groupby(route_column, sort=False)[delay_column]
    previous_delay = grouped_delay.shift(1)
    rolling_delay = previous_delay.groupby(featured[route_column], sort=False)

    featured["prev_delay_minutes"] = previous_delay.fillna(0.0)
    featured["rolling_delay_mean_3"] = rolling_delay.transform(
        lambda values: values.rolling(window=3, min_periods=1).mean()
    ).fillna(0.0)
    featured["rolling_delay_std_3"] = rolling_delay.transform(
        lambda values: values.rolling(window=3, min_periods=2).std()
    ).fillna(0.0)
    return featured.sort_values(timestamp_column).reset_index(drop=True)


def add_weather_event_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered weather and event intensity features."""
    featured = df.copy()
    featured["weather_risk_index"] = (
        (featured["precipitation_mm"] * 0.7)
        + (featured["wind_speed_kph"] * 0.04)
        + np.where(featured["temperature_c"] < 0.0, 2.0, 0.0)
        + np.where(featured["visibility_km"] < 2.0, 2.0, 0.0)
    )
    featured["is_heavy_rain"] = (featured["precipitation_mm"] >= 8.0).astype(int)
    featured["is_low_visibility"] = (featured["visibility_km"] < 2.0).astype(int)
    featured["event_attendance_log"] = np.log1p(featured["attendance"].clip(lower=0.0))
    return featured


def ensure_feature_contract(
    df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Ensure that all contract features exist with stable defaults."""
    feature_config = feature_config or FeatureConfig()
    featured = df.copy()
    for column in feature_config.numeric_features:
        if column not in featured.columns:
            featured[column] = np.nan
    for column in feature_config.categorical_features:
        if column not in featured.columns:
            featured[column] = "unknown"
        featured[column] = featured[column].astype("string").fillna("unknown")
    return featured


def add_cyclical_time_encoding(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Backward-compatible wrapper around the temporal feature builder."""
    featured = add_temporal_features(df=df, timestamp_column=timestamp_col)
    featured["sin_hour"] = featured["hour_sin"]
    featured["cos_hour"] = featured["hour_cos"]
    return featured


def engineer_training_dataset(
    bundle: DataBundle,
    training_config: TrainingConfig | None = None,
    ingestion_config: IngestionConfig | None = None,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline for offline training."""
    training_config = training_config or TrainingConfig()
    ingestion_config = ingestion_config or IngestionConfig()

    merged = merge_transport_weather(
        transport_df=bundle.transport,
        weather_df=bundle.weather,
        timestamp_column=training_config.timestamp_column,
    )
    merged = merge_transport_events(
        transport_df=merged,
        event_df=bundle.events,
        timestamp_column=training_config.timestamp_column,
        event_start_column=ingestion_config.event_start_column,
        event_end_column=ingestion_config.event_end_column,
    )
    merged = add_temporal_features(merged, timestamp_column=training_config.timestamp_column)
    merged = add_delay_history_features(
        merged,
        route_column="route_id",
        delay_column=training_config.target_column,
        timestamp_column=training_config.timestamp_column,
    )
    merged = add_weather_event_risk_features(merged)
    merged = ensure_feature_contract(merged, feature_config=training_config.feature_config)
    LOGGER.info(
        "Engineered training dataset.",
        extra={"rows": len(merged), "columns": len(merged.columns)},
    )
    return merged.sort_values(training_config.timestamp_column).reset_index(drop=True)


def _rolling_features_from_history(history: tuple[float, ...]) -> tuple[float, float, float]:
    """Compute lag and rolling features from a history tuple."""
    if not history:
        return 0.0, 0.0, 0.0
    history_array = np.asarray(history, dtype=float)
    prev_delay = float(history_array[-1])
    rolling_mean = float(history_array[-3:].mean())
    rolling_std = float(history_array[-3:].std(ddof=0))
    return prev_delay, rolling_mean, rolling_std


def build_inference_frame(
    context: PredictionContext,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Create a single-row model input frame from realtime request context."""
    feature_config = feature_config or FeatureConfig()
    transit = context.transit
    weather = context.weather
    event = context.event

    history = list(transit.historical_delay_minutes)
    if transit.observed_delay_minutes is not None:
        history.append(float(transit.observed_delay_minutes))

    prev_delay, rolling_mean, rolling_std = _rolling_features_from_history(tuple(history))
    timestamp = pd.Timestamp(transit.observed_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    attendance = event.attendance if event is not None and event.attendance is not None else 0.0

    record: dict[str, object] = {
        "timestamp": timestamp,
        "route_id": transit.route_id,
        "stop_id": transit.stop_id or "UNKNOWN_STOP",
        "city_zone": transit.city_zone or "unknown",
        "service_alert_level": transit.service_alert_level or "normal",
        "scheduled_headway_minutes": transit.scheduled_headway_minutes,
        "prev_delay_minutes": prev_delay,
        "rolling_delay_mean_3": rolling_mean,
        "rolling_delay_std_3": rolling_std,
        "precipitation_mm": weather.precipitation_mm if weather is not None else np.nan,
        "wind_speed_kph": weather.wind_speed_kph if weather is not None else np.nan,
        "temperature_c": weather.temperature_c if weather is not None else np.nan,
        "visibility_km": weather.visibility_km if weather is not None else np.nan,
        "attendance": attendance,
        "event_type": event.event_type if event is not None else "none",
        "event_severity": event.event_severity if event is not None else "none",
        "is_event_active": int(event.is_active) if event is not None else 0,
    }
    frame = pd.DataFrame([record])
    frame = add_temporal_features(frame)
    frame = add_weather_event_risk_features(frame)
    frame = ensure_feature_contract(frame, feature_config=feature_config)
    return frame.loc[:, list(feature_config.all_features)]
