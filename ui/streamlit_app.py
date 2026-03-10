import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st


LOGGER = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000/predict"
DEFAULT_FEATURE_NAMES_PATH = Path("artifacts/feature_names.joblib")


def configure_logging() -> None:
    """Configure basic logging for the Streamlit app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def compute_cyclical_hour(hour: int) -> Dict[str, float]:
    """Compute sine and cosine encodings for an hour of day."""
    radians = 2.0 * math.pi * float(hour) / 24.0
    return {
        "sin_hour": math.sin(radians),
        "cos_hour": math.cos(radians),
    }


def build_feature_payload(
    hour: int,
    prev_stop_delay: float,
    precipitation: float,
    wind_speed: float,
    temperature: float,
) -> Dict[str, Any]:
    """Build the feature dictionary expected by the backend model."""
    cyc = compute_cyclical_hour(hour)
    features: Dict[str, Any] = {
        "prev_stop_delay_minutes": prev_stop_delay,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "temperature": temperature,
        **cyc,
    }
    return features


def call_backend(features: Dict[str, Any]) -> float | None:
    """Send a prediction request to the FastAPI backend."""
    try:
        response = requests.post(
            BACKEND_URL,
            json={"features": features},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return float(data["delay_minutes"])
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Prediction request failed: %s", exc)
        st.error(f"Prediction request failed: {exc}")
        return None


def load_feature_names(path: Path = DEFAULT_FEATURE_NAMES_PATH) -> List[str] | None:
    """Load feature names to validate UI payloads against the trained model."""
    try:
        import joblib  # imported lazily to keep Streamlit import fast

        feature_names: List[str] = joblib.load(path)
        return feature_names
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load feature names from %s: %s", path, exc)
        return None


def main() -> None:
    """Streamlit entry point for the RainCheckAI dashboard."""
    configure_logging()

    st.title("RainCheckAI – Transit Delay Predictor")
    st.markdown(
        "Interactively explore how weather, time of day, and upstream delays "
        "influence public transport punctuality.",
    )

    st.sidebar.header("Input features")

    hour = st.sidebar.slider("Hour of day", min_value=0, max_value=23, value=8)
    prev_stop_delay = st.sidebar.slider(
        "Previous stop delay (minutes)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
    )
    precipitation = st.sidebar.slider(
        "Precipitation (mm/hr)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
    )
    wind_speed = st.sidebar.slider(
        "Wind speed (m/s)",
        min_value=0.0,
        max_value=30.0,
        value=3.0,
        step=0.5,
    )
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
    )

    feature_names = load_feature_names()

    if feature_names is not None:
        st.sidebar.caption(
            "Model expects the following features: "
            + ", ".join(feature_names),
        )

    if st.button("Predict delay"):
        features = build_feature_payload(
            hour=hour,
            prev_stop_delay=prev_stop_delay,
            precipitation=precipitation,
            wind_speed=wind_speed,
            temperature=temperature,
        )

        if feature_names is not None:
            missing = [name for name in feature_names if name not in features]
            if missing:
                st.warning(
                    "The following model features are missing from the UI payload "
                    f"and will be sent as NaN by the backend: {', '.join(missing)}",
                )

        st.write("Feature vector sent to model:", features)

        prediction = call_backend(features)
        if prediction is not None:
            st.success(f"Predicted delay: {prediction:.2f} minutes")


if __name__ == "__main__":
    main()

