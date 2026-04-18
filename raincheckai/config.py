"""Configuration objects for RainCheckAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

DEFAULT_ARTIFACT_DIR: Final[Path] = Path("artifacts")
DEFAULT_DATA_DIR: Final[Path] = Path("data/processed")
DEFAULT_TIMESTAMP_COLUMN: Final[str] = "timestamp"
DEFAULT_TARGET_COLUMN: Final[str] = "delay_minutes"

NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "scheduled_headway_minutes",
    "prev_delay_minutes",
    "rolling_delay_mean_3",
    "rolling_delay_std_3",
    "precipitation_mm",
    "wind_speed_kph",
    "temperature_c",
    "visibility_km",
    "weather_risk_index",
    "is_heavy_rain",
    "is_low_visibility",
    "is_event_active",
    "event_attendance_log",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_peak_service_window",
)

CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "route_id",
    "stop_id",
    "city_zone",
    "service_alert_level",
    "event_type",
    "event_severity",
)


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Filesystem locations for persisted artifacts."""

    root_dir: Path = DEFAULT_ARTIFACT_DIR
    model_bundle_filename: str = "model_bundle.joblib"
    metadata_filename: str = "model_metadata.json"

    @property
    def model_bundle_path(self) -> Path:
        """Return the serialized model bundle path."""
        return self.root_dir / self.model_bundle_filename

    @property
    def metadata_path(self) -> Path:
        """Return the model metadata JSON path."""
        return self.root_dir / self.metadata_filename


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Feature contract shared across training and inference."""

    numeric_features: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURE_COLUMNS

    @property
    def all_features(self) -> tuple[str, ...]:
        """Return the ordered feature contract."""
        return self.numeric_features + self.categorical_features


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for idempotent data ingestion."""

    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    target_column: str = DEFAULT_TARGET_COLUMN
    event_start_column: str = "start_timestamp"
    event_end_column: str = "end_timestamp"
    numeric_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delay_minutes": (-10.0, 240.0),
            "scheduled_headway_minutes": (0.0, 180.0),
            "precipitation_mm": (0.0, 250.0),
            "wind_speed_kph": (0.0, 180.0),
            "temperature_c": (-40.0, 55.0),
            "visibility_km": (0.0, 30.0),
            "attendance": (0.0, 500_000.0),
        },
    )
    transport_required_columns: tuple[str, ...] = ("route_id", "timestamp", "delay_minutes")
    weather_required_columns: tuple[str, ...] = ("timestamp",)
    weather_optional_numeric_columns: tuple[str, ...] = (
        "precipitation_mm",
        "wind_speed_kph",
        "temperature_c",
        "visibility_km",
    )
    event_optional_columns: tuple[str, ...] = (
        "event_type",
        "event_severity",
        "attendance",
        "start_timestamp",
        "end_timestamp",
    )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for model training and validation."""

    random_seed: int = 42
    target_column: str = DEFAULT_TARGET_COLUMN
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    n_splits: int = 5
    search_iterations: int = 8
    model_name: str = "random_forest_regressor"
