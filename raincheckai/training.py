"""Model training and persistence for RainCheckAI."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from raincheckai.config import ArtifactPaths, TrainingConfig
from raincheckai.contracts import ModelBundle, ModelMetadata
from raincheckai.errors import ArtifactNotAvailableError

LOGGER = logging.getLogger(__name__)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features using training-set quantiles."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        """Initialize clipper quantiles."""
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantileClipper:
        """Learn per-column clipping bounds from training data."""
        del y
        array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip feature values to the fitted quantile bounds."""
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)


def load_engineered_dataset(
    path: Path,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Load a feature-engineered dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Engineered dataset not found: {path}")
    dataset = pd.read_csv(path, parse_dates=[timestamp_column])
    if dataset[timestamp_column].dt.tz is None:
        dataset[timestamp_column] = dataset[timestamp_column].dt.tz_localize("UTC")
    else:
        dataset[timestamp_column] = dataset[timestamp_column].dt.tz_convert("UTC")
    return dataset.sort_values(timestamp_column).reset_index(drop=True)


def build_training_frame(
    dataset: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered dataset into ordered feature and target frames."""
    config = config or TrainingConfig()
    feature_columns = list(config.feature_config.all_features)
    ordered = dataset.sort_values(config.timestamp_column).reset_index(drop=True)
    model_frame = ordered.dropna(subset=[config.target_column]).copy()
    X = model_frame.loc[:, feature_columns]
    y = pd.to_numeric(model_frame[config.target_column], errors="coerce")
    valid_mask = y.notna()
    return X.loc[valid_mask].reset_index(drop=True), y.loc[valid_mask].reset_index(drop=True)


def build_training_pipeline(config: TrainingConfig | None = None) -> Pipeline:
    """Construct the sklearn preprocessing and regression pipeline."""
    config = config or TrainingConfig()
    numeric_features = list(config.feature_config.numeric_features)
    categorical_features = list(config.feature_config.categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper()),
            ("scaler", StandardScaler()),
        ],
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    regressor = RandomForestRegressor(
        random_state=config.random_seed,
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ],
    )


def _parameter_distributions() -> dict[str, list[Any]]:
    """Return deterministic hyperparameter candidates for search."""
    return {
        "regressor__n_estimators": [200, 400, 600],
        "regressor__max_depth": [10, 16, 24, None],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", 0.7, 1.0],
    }


def _normalize_best_params(best_params: dict[str, Any]) -> dict[str, Any]:
    """Convert search parameters into JSON-serializable scalars."""
    normalized: dict[str, Any] = {}
    for key, value in best_params.items():
        if isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def _evaluate_time_series_cv(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_strategy: TimeSeriesSplit,
) -> tuple[float, float]:
    """Compute time-series CV metrics using explicit folds."""
    rmse_scores: list[float] = []
    mae_scores: list[float] = []

    for train_indices, validation_indices in split_strategy.split(X):
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_validation)
        rmse_scores.append(float(root_mean_squared_error(y_validation, predictions)))
        mae_scores.append(float(mean_absolute_error(y_validation, predictions)))

    return float(np.mean(rmse_scores)), float(np.mean(mae_scores))


def train_model_bundle(
    dataset: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> ModelBundle:
    """Train a reproducible model bundle from an engineered dataset."""
    config = config or TrainingConfig()
    X, y = build_training_frame(dataset, config=config)
    pipeline = build_training_pipeline(config=config)
    split_strategy = TimeSeriesSplit(n_splits=config.n_splits)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=_parameter_distributions(),
        n_iter=config.search_iterations,
        scoring="neg_root_mean_squared_error",
        cv=split_strategy,
        refit=True,
        random_state=config.random_seed,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    best_pipeline = search.best_estimator_
    evaluation_split_strategy = TimeSeriesSplit(n_splits=config.n_splits)
    cv_rmse, cv_mae = _evaluate_time_series_cv(
        estimator=best_pipeline,
        X=X,
        y=y,
        split_strategy=evaluation_split_strategy,
    )
    training_predictions = best_pipeline.predict(X)
    training_rmse = float(root_mean_squared_error(y, training_predictions))
    metadata = ModelMetadata(
        model_name=config.model_name,
        model_version=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        random_seed=config.random_seed,
        target_column=config.target_column,
        numeric_features=config.feature_config.numeric_features,
        categorical_features=config.feature_config.categorical_features,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        training_rmse=training_rmse,
        baseline_delay_minutes=float(y.median()),
        training_rows=int(len(X)),
        best_params=_normalize_best_params(search.best_params_),
    )
    LOGGER.info(
        "Trained model bundle.",
        extra={
            "model_version": metadata.model_version,
            "cv_rmse": cv_rmse,
            "cv_mae": cv_mae,
            "training_rows": metadata.training_rows,
        },
    )
    return ModelBundle(pipeline=best_pipeline, metadata=metadata)


def save_model_bundle(
    bundle: ModelBundle,
    artifact_paths: ArtifactPaths | None = None,
) -> tuple[Path, Path]:
    """Persist a trained model bundle and its JSON metadata."""
    artifact_paths = artifact_paths or ArtifactPaths()
    artifact_paths.root_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_paths.model_bundle_path)
    artifact_paths.metadata_path.write_text(
        json.dumps(bundle.metadata.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info(
        "Persisted model artifacts.",
        extra={
            "model_bundle_path": artifact_paths.model_bundle_path,
            "metadata_path": artifact_paths.metadata_path,
        },
    )
    return artifact_paths.model_bundle_path, artifact_paths.metadata_path


def load_model_bundle(
    artifact_paths: ArtifactPaths | None = None,
) -> ModelBundle:
    """Load a trained model bundle from disk."""
    artifact_paths = artifact_paths or ArtifactPaths()
    if not artifact_paths.model_bundle_path.exists():
        raise ArtifactNotAvailableError(
            f"Model bundle not found at {artifact_paths.model_bundle_path}",
        )
    bundle = joblib.load(artifact_paths.model_bundle_path)
    if not isinstance(bundle, ModelBundle):
        raise ArtifactNotAvailableError("Serialized artifact is not a RainCheckAI model bundle.")
    return bundle


def train_and_persist(
    engineered_dataset_path: Path,
    artifact_paths: ArtifactPaths | None = None,
    config: TrainingConfig | None = None,
) -> ModelBundle:
    """Train a bundle from disk and persist it to the artifact store."""
    config = config or TrainingConfig()
    dataset = load_engineered_dataset(
        path=engineered_dataset_path,
        timestamp_column=config.timestamp_column,
    )
    bundle = train_model_bundle(dataset=dataset, config=config)
    save_model_bundle(bundle=bundle, artifact_paths=artifact_paths)
    return bundle
