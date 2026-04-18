"""Idempotent data ingestion pipeline for RainCheckAI."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from raincheckai.config import IngestionConfig
from raincheckai.contracts import DataBundle
from raincheckai.errors import DatasetValidationError

LOGGER = logging.getLogger(__name__)

_TRANSPORT_ALIASES: dict[str, str] = {
    "route": "route_id",
    "stop": "stop_id",
    "service_alert": "service_alert_level",
    "headway_minutes": "scheduled_headway_minutes",
}

_WEATHER_ALIASES: dict[str, str] = {
    "precipitation": "precipitation_mm",
    "wind_speed": "wind_speed_kph",
    "temperature": "temperature_c",
    "visibility": "visibility_km",
}

_EVENT_ALIASES: dict[str, str] = {
    "severity": "event_severity",
    "start_time": "start_timestamp",
    "end_time": "end_timestamp",
    "venue_capacity": "attendance",
}


def load_csv_frame(path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    LOGGER.info("Loading CSV file.", extra={"path": path})
    return pd.read_csv(path)


def _rename_known_columns(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Rename supported alias columns into the canonical RainCheckAI schema."""
    renamed = df.copy()
    columns_to_rename = {column: aliases[column] for column in renamed.columns if column in aliases}
    return renamed.rename(columns=columns_to_rename)


def _ensure_columns(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise DatasetValidationError(
            f"Missing required columns: {', '.join(missing_columns)}",
        )


def standardize_timestamp_column(
    df: pd.DataFrame,
    source_column: str,
    target_column: str = "timestamp",
) -> pd.DataFrame:
    """Parse a timestamp column into a canonical UTC timestamp column."""
    standardized = df.copy()
    standardized[target_column] = pd.to_datetime(
        standardized[source_column],
        errors="coerce",
        utc=True,
    )
    invalid_count = int(standardized[target_column].isna().sum())
    if invalid_count > 0:
        LOGGER.warning(
            "Timestamp parsing coerced invalid values to NaT.",
            extra={"source_column": source_column, "invalid_count": invalid_count},
        )
    return standardized


def _clip_numeric_column(
    df: pd.DataFrame,
    column: str,
    bounds: tuple[float, float] | None,
) -> pd.DataFrame:
    """Clip a numeric column to deterministic plausibility bounds."""
    clipped = df.copy()
    clipped[column] = pd.to_numeric(clipped[column], errors="coerce")
    if bounds is None:
        return clipped
    lower_bound, upper_bound = bounds
    clipped[column] = clipped[column].clip(lower=lower_bound, upper=upper_bound)
    return clipped


def _fill_numeric_column(
    df: pd.DataFrame,
    column: str,
    default_value: float,
) -> pd.DataFrame:
    """Fill missing numeric values with a deterministic median-or-default strategy."""
    filled = df.copy()
    filled[column] = pd.to_numeric(filled[column], errors="coerce")
    non_null_values = filled[column].dropna()
    if non_null_values.empty:
        fill_value = default_value
    else:
        fill_value = float(non_null_values.median())
    filled[column] = filled[column].fillna(fill_value)
    return filled


def _fill_categorical_column(
    df: pd.DataFrame,
    column: str,
    default_value: str,
) -> pd.DataFrame:
    """Fill missing categorical values with a stable sentinel."""
    filled = df.copy()
    filled[column] = (
        filled[column].astype("string").fillna(default_value).replace({"<NA>": default_value})
    )
    return filled


def _empty_event_frame(config: IngestionConfig) -> pd.DataFrame:
    """Return an empty events frame with the canonical schema."""
    return pd.DataFrame(
        {
            config.event_start_column: pd.Series(dtype="datetime64[ns, UTC]"),
            config.event_end_column: pd.Series(dtype="datetime64[ns, UTC]"),
            "event_type": pd.Series(dtype="string"),
            "event_severity": pd.Series(dtype="string"),
            "attendance": pd.Series(dtype="float64"),
        },
    )


def clean_transport_data(
    df: pd.DataFrame,
    config: IngestionConfig | None = None,
) -> pd.DataFrame:
    """Normalize and clean transport observations."""
    config = config or IngestionConfig()
    cleaned = _rename_known_columns(df, _TRANSPORT_ALIASES)
    _ensure_columns(cleaned, config.transport_required_columns)
    cleaned = standardize_timestamp_column(
        cleaned,
        config.timestamp_column,
        config.timestamp_column,
    )

    if "stop_id" not in cleaned.columns:
        cleaned["stop_id"] = "UNKNOWN_STOP"
    if "city_zone" not in cleaned.columns:
        cleaned["city_zone"] = "unknown"
    if "service_alert_level" not in cleaned.columns:
        cleaned["service_alert_level"] = "normal"
    if "scheduled_headway_minutes" not in cleaned.columns:
        cleaned["scheduled_headway_minutes"] = np.nan

    for column, default_value in (
        ("route_id", "unknown_route"),
        ("stop_id", "UNKNOWN_STOP"),
        ("city_zone", "unknown"),
        ("service_alert_level", "normal"),
    ):
        cleaned = _fill_categorical_column(cleaned, column, default_value)

    cleaned = _clip_numeric_column(
        cleaned,
        config.target_column,
        config.numeric_bounds.get(config.target_column),
    )
    cleaned = _clip_numeric_column(
        cleaned,
        "scheduled_headway_minutes",
        config.numeric_bounds.get("scheduled_headway_minutes"),
    )
    cleaned = _fill_numeric_column(cleaned, "scheduled_headway_minutes", default_value=10.0)
    cleaned = cleaned.drop_duplicates(subset=["route_id", "stop_id", config.timestamp_column])
    cleaned = cleaned.sort_values(config.timestamp_column).reset_index(drop=True)
    return cleaned


def clean_weather_data(
    df: pd.DataFrame,
    config: IngestionConfig | None = None,
) -> pd.DataFrame:
    """Normalize and clean weather observations."""
    config = config or IngestionConfig()
    cleaned = _rename_known_columns(df, _WEATHER_ALIASES)
    _ensure_columns(cleaned, config.weather_required_columns)
    cleaned = standardize_timestamp_column(
        cleaned,
        config.timestamp_column,
        config.timestamp_column,
    )

    for column in config.weather_optional_numeric_columns:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned = _clip_numeric_column(cleaned, column, config.numeric_bounds.get(column))

    default_values = {
        "precipitation_mm": 0.0,
        "wind_speed_kph": 15.0,
        "temperature_c": 15.0,
        "visibility_km": 10.0,
    }
    for column, default_value in default_values.items():
        cleaned = _fill_numeric_column(cleaned, column, default_value)

    cleaned = cleaned.drop_duplicates(subset=[config.timestamp_column], keep="last")
    cleaned = cleaned.sort_values(config.timestamp_column).reset_index(drop=True)
    return cleaned


def clean_event_data(
    df: pd.DataFrame | None,
    config: IngestionConfig | None = None,
) -> pd.DataFrame:
    """Normalize and clean optional event data."""
    config = config or IngestionConfig()
    if df is None or df.empty:
        return _empty_event_frame(config)

    cleaned = _rename_known_columns(df, _EVENT_ALIASES)
    if config.event_start_column not in cleaned.columns:
        raise DatasetValidationError("Event data must contain a start timestamp column.")

    cleaned = standardize_timestamp_column(
        cleaned,
        config.event_start_column,
        config.event_start_column,
    )

    if config.event_end_column not in cleaned.columns:
        cleaned[config.event_end_column] = cleaned[config.event_start_column] + pd.Timedelta(
            hours=2
        )
    cleaned = standardize_timestamp_column(
        cleaned,
        config.event_end_column,
        config.event_end_column,
    )
    cleaned[config.event_end_column] = cleaned[config.event_end_column].fillna(
        cleaned[config.event_start_column] + pd.Timedelta(hours=2),
    )

    if "event_type" not in cleaned.columns:
        cleaned["event_type"] = "none"
    if "event_severity" not in cleaned.columns:
        cleaned["event_severity"] = "none"
    if "attendance" not in cleaned.columns:
        cleaned["attendance"] = 0.0

    cleaned = _fill_categorical_column(cleaned, "event_type", "none")
    cleaned = _fill_categorical_column(cleaned, "event_severity", "none")
    cleaned = _clip_numeric_column(cleaned, "attendance", config.numeric_bounds.get("attendance"))
    cleaned = _fill_numeric_column(cleaned, "attendance", default_value=0.0)
    cleaned = cleaned.dropna(subset=[config.event_start_column]).copy()
    cleaned = cleaned.drop_duplicates(
        subset=[config.event_start_column, config.event_end_column, "event_type"],
    )
    cleaned = cleaned.sort_values(config.event_start_column).reset_index(drop=True)
    return cleaned


def load_data_bundle(
    transport_path: Path,
    weather_path: Path,
    events_path: Path | None = None,
    config: IngestionConfig | None = None,
) -> DataBundle:
    """Load and clean transport, weather, and optional event datasets."""
    config = config or IngestionConfig()
    transport_df = clean_transport_data(load_csv_frame(transport_path), config=config)
    weather_df = clean_weather_data(load_csv_frame(weather_path), config=config)
    events_df = clean_event_data(
        load_csv_frame(events_path) if events_path is not None else None,
        config=config,
    )
    LOGGER.info(
        "Loaded cleaned data bundle.",
        extra={
            "transport_rows": len(transport_df),
            "weather_rows": len(weather_df),
            "event_rows": len(events_df),
        },
    )
    return DataBundle(transport=transport_df, weather=weather_df, events=events_df)


def write_clean_data_bundle(
    bundle: DataBundle,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Persist a cleaned data bundle to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transport_path = output_dir / "transport_clean.csv"
    weather_path = output_dir / "weather_clean.csv"
    events_path = output_dir / "events_clean.csv"

    bundle.transport.to_csv(transport_path, index=False)
    bundle.weather.to_csv(weather_path, index=False)
    bundle.events.to_csv(events_path, index=False)
    LOGGER.info(
        "Persisted cleaned data bundle.",
        extra={
            "transport_path": transport_path,
            "weather_path": weather_path,
            "events_path": events_path,
        },
    )
    return transport_path, weather_path, events_path
