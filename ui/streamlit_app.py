"""Streamlit dashboard for RainCheckAI."""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Any

import requests
import streamlit as st

from raincheckai.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
BACKEND_URL = "http://localhost:8000/predict"


def parse_historical_delays(raw_value: str) -> list[float]:
    """Parse a comma-separated history string into delay values."""
    if not raw_value.strip():
        return []
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def build_request_payload(
    observed_date: date,
    observed_time: time,
    route_id: str,
    stop_id: str,
    city_zone: str,
    observed_delay_minutes: float,
    historical_delay_minutes: list[float],
    scheduled_headway_minutes: float,
    service_alert_level: str,
    precipitation_mm: float,
    wind_speed_kph: float,
    temperature_c: float,
    visibility_km: float,
    event_type: str,
    event_severity: str,
    attendance: float,
    is_event_active: bool,
) -> dict[str, Any]:
    """Build a prediction payload in the API contract."""
    observed_at = datetime.combine(observed_date, observed_time).isoformat() + "Z"
    return {
        "transit": {
            "route_id": route_id,
            "observed_at": observed_at,
            "stop_id": stop_id,
            "city_zone": city_zone,
            "observed_delay_minutes": observed_delay_minutes,
            "historical_delay_minutes": historical_delay_minutes,
            "scheduled_headway_minutes": scheduled_headway_minutes,
            "service_alert_level": service_alert_level,
        },
        "weather": {
            "precipitation_mm": precipitation_mm,
            "wind_speed_kph": wind_speed_kph,
            "temperature_c": temperature_c,
            "visibility_km": visibility_km,
        },
        "event": {
            "event_type": event_type,
            "event_severity": event_severity,
            "attendance": attendance,
            "is_active": is_event_active,
        },
    }


def call_backend(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Call the FastAPI backend and return the decoded JSON response."""
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=10)
        response.raise_for_status()
        return dict(response.json())
    except requests.RequestException as exc:
        LOGGER.error("Prediction request failed.", extra={"error": str(exc)})
        st.error(f"Prediction request failed: {exc}")
        return None


def main() -> None:
    """Render the Streamlit UI."""
    configure_logging()

    st.set_page_config(page_title="RainCheckAI", layout="wide")
    st.title("RainCheckAI")
    st.caption("Transit delay forecasting with weather and event context.")

    with st.sidebar:
        st.header("Transit context")
        route_id = st.selectbox("Route", options=["BLUE_LINE", "GREEN_LINE", "RED_LINE"])
        stop_id = st.selectbox("Stop", options=["STOP_101", "STOP_102", "STOP_205", "STOP_410"])
        city_zone = st.selectbox("Zone", options=["downtown", "midtown", "suburban"])
        observed_date = st.date_input("Observed date", value=date.today())
        observed_time = st.time_input("Observed time", value=time(hour=8, minute=30))
        observed_delay_minutes = st.slider(
            "Observed upstream delay (minutes)",
            0.0,
            30.0,
            5.0,
            0.5,
        )
        scheduled_headway_minutes = st.slider(
            "Scheduled headway (minutes)",
            4.0,
            20.0,
            10.0,
            1.0,
        )
        service_alert_level = st.selectbox(
            "Service alert level", options=["normal", "minor", "major"]
        )
        history_raw = st.text_input(
            "Historical delays", value="2,4,5", help="Comma-separated minutes."
        )

        st.header("Weather context")
        precipitation_mm = st.slider("Precipitation (mm)", 0.0, 30.0, 4.0, 0.5)
        wind_speed_kph = st.slider("Wind speed (kph)", 0.0, 80.0, 18.0, 1.0)
        temperature_c = st.slider("Temperature (C)", -20.0, 40.0, 12.0, 0.5)
        visibility_km = st.slider("Visibility (km)", 0.0, 20.0, 8.0, 0.5)

        st.header("Event context")
        event_type = st.selectbox(
            "Event type", options=["none", "concert", "sports", "festival", "conference"]
        )
        event_severity = st.selectbox(
            "Event severity",
            options=["none", "low", "medium", "high"],
        )
        attendance = st.slider("Attendance", 0.0, 50000.0, 10000.0, 500.0)
        is_event_active = st.checkbox("Event active", value=False)

    if st.button("Predict delay", type="primary"):
        try:
            history = parse_historical_delays(history_raw)
        except ValueError as exc:
            st.error(f"Historical delays must be numeric: {exc}")
            return

        payload = build_request_payload(
            observed_date=observed_date,
            observed_time=observed_time,
            route_id=route_id,
            stop_id=stop_id,
            city_zone=city_zone,
            observed_delay_minutes=observed_delay_minutes,
            historical_delay_minutes=history,
            scheduled_headway_minutes=scheduled_headway_minutes,
            service_alert_level=service_alert_level,
            precipitation_mm=precipitation_mm,
            wind_speed_kph=wind_speed_kph,
            temperature_c=temperature_c,
            visibility_km=visibility_km,
            event_type=event_type,
            event_severity=event_severity,
            attendance=attendance,
            is_event_active=is_event_active,
        )
        st.code(str(payload), language="python")
        result = call_backend(payload)
        if result is not None:
            st.metric("Predicted delay (minutes)", f"{result['predicted_delay_minutes']:.2f}")
            st.write("Request ID:", result["request_id"])
            if result["degraded"]:
                st.warning(
                    "Prediction ran in degraded mode: " + ", ".join(result["degradation_reasons"])
                )
            else:
                st.success("Prediction served from the trained model bundle.")


if __name__ == "__main__":
    main()
