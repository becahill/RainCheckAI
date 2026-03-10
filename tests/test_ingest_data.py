import pandas as pd

from ingest_data import standardise_timestamp


def test_standardise_timestamp_parses_and_flags_invalid() -> None:
    """standardise_timestamp should parse valid timestamps and coerce invalid ones."""
    df = pd.DataFrame(
        {
            "raw_ts": [
                "2024-01-01 10:00:00",
                "not-a-timestamp",
            ],
        },
    )

    result = standardise_timestamp(df, timestamp_col="raw_ts", new_col="ts", utc=True)

    assert "ts" in result.columns
    assert pd.api.types.is_datetime64tz_dtype(result["ts"])

    # First row should parse to a valid timestamp, second should be NaT
    assert pd.notna(result.loc[0, "ts"])
    assert pd.isna(result.loc[1, "ts"])

