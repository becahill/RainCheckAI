import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from engineering import select_feature_columns


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_FEATURE_NAMES_PATH = Path("artifacts/feature_names.joblib")
OPTUNA_RANDOM_STATE = 42


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure basic logging for the module."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def load_engineered_data(
    path: Path,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Load engineered dataset from CSV.

    Parameters
    ----------
    path:
        Path to the engineered CSV file.
    timestamp_col:
        Name of the timestamp column to parse.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame sorted by timestamp.
    """
    LOGGER.info("Loading engineered data from %s", path)
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    LOGGER.info("Loaded %d rows and %d columns.", len(df), df.shape[1])
    return df


def build_feature_target_matrices(
    df: pd.DataFrame,
    target_col: str = "delay_minutes",
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare feature matrix X and target vector y for modelling.

    Parameters
    ----------
    df:
        Engineered DataFrame.
    target_col:
        Target column containing delay in minutes.
    timestamp_col:
        Timestamp column (excluded from X but used for splitting).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        Feature matrix, target vector, and the list of feature names.
    """
    ignore_cols = [timestamp_col, "time_key"]
    feature_cols = select_feature_columns(df, target_col=target_col, ignore_cols=ignore_cols)

    # Drop initial rows where lag feature might be NaN.
    LOGGER.info("Dropping rows with missing target or feature values.")
    model_df = df.dropna(subset=feature_cols + [target_col]).copy()

    X = model_df[feature_cols]
    y = model_df[target_col]

    LOGGER.info("Final modelling dataset has %d rows and %d features.", len(model_df), len(feature_cols))
    return X, y, feature_cols


def evaluate_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    params: Dict[str, float],
) -> float:
    """Evaluate XGBoost hyperparameters using TimeSeriesSplit RMSE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses: List[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            n_estimators=int(params["n_estimators"]),
            subsample=float(params["subsample"]),
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=OPTUNA_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        rmses.append(rmse)

    mean_rmse = float(np.mean(rmses))
    return mean_rmse


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_trials: int = 30,
    random_state: int = OPTUNA_RANDOM_STATE,
) -> Tuple[Dict[str, float], float]:
    """Use Optuna to tune XGBoost hyperparameters for lowest TimeSeries RMSE."""

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective that returns mean TimeSeries RMSE for a trial."""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        rmse = evaluate_hyperparameters(X=X, y=y, n_splits=n_splits, params=params)
        trial.report(rmse, step=0)
        return rmse

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    LOGGER.info("Starting Optuna hyperparameter search with %d trials.", n_trials)
    study.optimize(objective, n_trials=n_trials)

    best_params: Dict[str, float] = study.best_trial.params
    best_rmse: float = float(study.best_value)
    LOGGER.info("Optuna best RMSE: %.4f", best_rmse)
    LOGGER.info("Optuna best params: %s", best_params)
    return best_params, best_rmse


def log_feature_importance(
    model: XGBRegressor,
    feature_names: List[str],
    top_k: Optional[int] = 20,
) -> None:
    """Log feature importances learned by the XGBoost model.

    Parameters
    ----------
    model:
        Fitted XGBRegressor instance.
    feature_names:
        Names of the features used during training.
    top_k:
        Number of top features to log. If ``None``, log all.
    """
    importances = model.feature_importances_
    if importances is None or len(importances) == 0:
        LOGGER.warning("Model did not expose feature importances.")
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances},
    ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    LOGGER.info("Top feature importances:")
    for _, row in importance_df.iterrows():
        LOGGER.info("  %s: %.4f", row["feature"], row["importance"])


def save_model_artifacts(
    model: XGBRegressor,
    feature_names: List[str],
    model_path: Path,
    feature_names_path: Path,
) -> None:
    """Persist the trained model and its feature names using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving trained model to %s", model_path)
    joblib.dump(model, model_path)

    LOGGER.info("Saving feature names to %s", feature_names_path)
    joblib.dump(feature_names, feature_names_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model to predict public transport delays.",
    )
    parser.add_argument(
        "--engineered-csv",
        type=Path,
        required=True,
        help="Path to engineered dataset CSV (output of engineering.py).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="delay_minutes",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Name of the timestamp column.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--feature-names-path",
        type=Path,
        default=DEFAULT_FEATURE_NAMES_PATH,
        help="Path where the feature names list will be saved.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for hyperparameter search.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for model training pipeline."""
    configure_logging()
    args = parse_args()

    df = load_engineered_data(
        path=args.engineered_csv,
        timestamp_col=args.timestamp_col,
    )
    X, y, feature_names = build_feature_target_matrices(
        df=df,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
    )

    best_params, best_rmse = tune_hyperparameters(
        X=X,
        y=y,
        n_splits=5,
        n_trials=args.optuna_trials,
        random_state=OPTUNA_RANDOM_STATE,
    )
    LOGGER.info("Using best RMSE %.4f with tuned hyperparameters for final training.", best_rmse)

    model = XGBRegressor(
        learning_rate=float(best_params["learning_rate"]),
        max_depth=int(best_params["max_depth"]),
        n_estimators=int(best_params["n_estimators"]),
        subsample=float(best_params["subsample"]),
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=OPTUNA_RANDOM_STATE,
        n_jobs=-1,
    )
    LOGGER.info("Training final model on full dataset with tuned hyperparameters.")
    model.fit(X, y)

    log_feature_importance(model, feature_names, top_k=20)
    save_model_artifacts(
        model=model,
        feature_names=feature_names,
        model_path=args.model_path,
        feature_names_path=args.feature_names_path,
    )


if __name__ == "__main__":
    main()
