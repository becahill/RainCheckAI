"""Local CLI for RainCheckAI inference."""

from __future__ import annotations

import argparse
import json

from raincheckai.api.schemas import PredictRequest
from raincheckai.inference import InferenceService
from raincheckai.logging_utils import configure_logging
from raincheckai.telemetry import TelemetryCollector


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local inference."""
    parser = argparse.ArgumentParser(description="Run a local RainCheckAI prediction.")
    parser.add_argument("--request-json", type=str, help="Full prediction request payload as JSON.")
    parser.add_argument("--example", action="store_true", help="Run a built-in example request.")
    return parser.parse_args()


def _example_payload() -> dict[str, object]:
    """Return a representative prediction payload."""
    return {
        "transit": {
            "route_id": "BLUE_LINE",
            "observed_at": "2026-04-18T08:30:00Z",
            "stop_id": "STOP_101",
            "city_zone": "downtown",
            "observed_delay_minutes": 6.0,
            "historical_delay_minutes": [2.0, 4.0, 5.0],
            "scheduled_headway_minutes": 10.0,
            "service_alert_level": "minor",
        },
        "weather": {
            "precipitation_mm": 5.2,
            "wind_speed_kph": 24.0,
            "temperature_c": 11.0,
            "visibility_km": 5.5,
        },
        "event": {
            "event_type": "concert",
            "event_severity": "medium",
            "attendance": 12000.0,
            "is_active": True,
        },
    }


def main() -> None:
    """Execute the local inference CLI."""
    configure_logging()
    args = parse_args()

    if not args.request_json and not args.example:
        raise SystemExit("Provide --request-json or --example.")

    payload = _example_payload() if args.example else json.loads(args.request_json or "{}")
    request = PredictRequest.model_validate(payload)

    service = InferenceService(telemetry=TelemetryCollector())
    service.load()
    result = service.predict(context=request.to_domain(), request_id="cli-request")
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
