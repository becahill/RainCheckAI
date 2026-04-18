"""API integration tests for RainCheckAI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from raincheckai.api.app import create_app


def test_predict_endpoint_degrades_gracefully_without_model_artifacts() -> None:
    """The API should return a degraded prediction instead of failing hard."""
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "transit": {
                    "route_id": "BLUE_LINE",
                    "observed_at": "2026-04-18T08:30:00Z",
                    "historical_delay_minutes": [2.0, 4.0, 5.0],
                }
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["degraded"] is True
        assert "model_bundle_unavailable" in payload["degradation_reasons"]
        assert response.headers["x-request-id"]

        metrics_response = client.get("/metrics")
        metrics_payload = metrics_response.json()["metrics"]
        assert metrics_payload["counters"]["inference_requests_total|"] >= 1.0
