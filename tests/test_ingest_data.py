"""Tests for the RainCheckAI ingestion layer."""

from __future__ import annotations

import pandas as pd

from raincheckai.ingestion import (
    clean_event_data,
    clean_transport_data,
    standardize_timestamp_column,
)


def test_standardize_timestamp_column_parses_and_flags_invalid() -> None:
    """Invalid timestamps should be coerced to NaT in UTC."""
    df = pd.DataFrame({"raw_ts": ["2026-01-01T10:00:00Z", "bad-timestamp"]})

    result = standardize_timestamp_column(df, source_column="raw_ts", target_column="timestamp")

    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert pd.notna(result.loc[0, "timestamp"])
    assert pd.isna(result.loc[1, "timestamp"])


def test_clean_transport_data_is_idempotent_and_applies_defaults() -> None:
    """Transport cleaning should deduplicate rows and inject stable defaults."""
    df = pd.DataFrame(
        {
            "route_id": ["BLUE_LINE", "BLUE_LINE"],
            "timestamp": ["2026-04-18T08:00:00Z", "2026-04-18T08:00:00Z"],
            "delay_minutes": [500.0, 500.0],
        }
    )

    cleaned = clean_transport_data(df)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "delay_minutes"] == 240.0
    assert cleaned.loc[0, "stop_id"] == "UNKNOWN_STOP"
    assert cleaned.loc[0, "city_zone"] == "unknown"
    assert cleaned.loc[0, "service_alert_level"] == "normal"
    assert cleaned.loc[0, "scheduled_headway_minutes"] == 10.0


def test_clean_event_data_returns_empty_contract_when_absent() -> None:
    """Missing event inputs should still produce a valid empty event frame."""
    cleaned = clean_event_data(None)

    assert cleaned.empty
    assert list(cleaned.columns) == [
        "start_timestamp",
        "end_timestamp",
        "event_type",
        "event_severity",
        "attendance",
    ]
