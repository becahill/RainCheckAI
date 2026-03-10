import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_FEATURE_NAMES_PATH = Path("artifacts/feature_names.joblib")


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure logging for prediction-time components."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def load_model_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    feature_names_path: Path = DEFAULT_FEATURE_NAMES_PATH,
) -> Tuple[XGBRegressor, List[str]]:
    """Load a trained XGBoost model and corresponding feature names.

    Parameters
    ----------
    model_path:
        Location of the persisted model.
    feature_names_path:
        Location of the persisted feature name list.

    Returns
    -------
    Tuple[XGBRegressor, List[str]]
        Loaded model and ordered list of feature names.
    """
    LOGGER.info("Loading trained model from %s", model_path)
    model = joblib.load(model_path)

    LOGGER.info("Loading feature names from %s", feature_names_path)
    feature_names: List[str] = joblib.load(feature_names_path)

    return model, feature_names


def build_feature_vector(
    event: Dict[str, Any],
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Construct a single-row feature matrix from a JSON-like event.

    The function assumes that the incoming JSON already contains
    model-ready numeric features with keys matching ``feature_names``.
    Missing features are filled with ``NaN`` and extra keys are ignored.

    Parameters
    ----------
    event:
        Dictionary representing a new transit event.
    feature_names:
        Names and ordering of features expected by the model.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame in the correct column order.
    """
    feature_names_list = list(feature_names)
    data = {name: event.get(name, np.nan) for name in feature_names_list}
    return pd.DataFrame([data], columns=feature_names_list)


def predict_delay_from_event(
    model: XGBRegressor,
    feature_names: List[str],
    event: Dict[str, Any],
) -> float:
    """Generate a delay prediction in minutes for a single event.

    Parameters
    ----------
    model:
        Trained XGBoost regressor.
    feature_names:
        Ordered feature names used during model training.
    event:
        JSON-like mapping containing numeric feature values.

    Returns
    -------
    float
        Predicted delay in minutes.
    """
    features_df = build_feature_vector(event, feature_names)
    prediction_array = model.predict(features_df)
    prediction = float(prediction_array[0])
    LOGGER.info("Generated prediction: %.4f minutes delay", prediction)
    return prediction


def main() -> None:
    """CLI entry point for local predictions.

    Usage examples
    --------------
    Predict from a JSON string:

        python -m src.predict --event-json \
          '{"sin_hour": 0.0, "cos_hour": 1.0, "prev_stop_delay_minutes": 2.5}'

    Or use the built-in example payload:

        python -m src.predict --example
    """
    parser = argparse.ArgumentParser(description="Local CLI for RainCheckAI predictions.")
    parser.add_argument(
        "--event-json",
        type=str,
        help="JSON string with model-ready feature values.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Use a built-in example feature payload.",
    )
    args = parser.parse_args()

    if not args.event_json and not args.example:
        parser.error("Provide --event-json or --example.")

    configure_logging()
    model, feature_names = load_model_artifacts()

    if args.example:
        event: Dict[str, Any] = {
            "sin_hour": 0.0,
            "cos_hour": 1.0,
            "prev_stop_delay_minutes": 5.0,
            "precipitation": 0.0,
            "wind_speed": 3.0,
            "temperature": 15.0,
        }
    else:
        try:
            event = json.loads(args.event_json or "")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --event-json: {exc}") from exc

    prediction = predict_delay_from_event(model, feature_names, event)
    print(prediction)


if __name__ == "__main__":
    main()

