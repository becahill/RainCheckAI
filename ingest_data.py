"""CLI entrypoint for the RainCheckAI ingestion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raincheckai.ingestion import (
    load_data_bundle,
    standardize_timestamp_column,
    write_clean_data_bundle,
)
from raincheckai.logging_utils import configure_logging


def standardise_timestamp(
    df: pd.DataFrame,
    timestamp_col: str,
    new_col: str = "timestamp",
    utc: bool = True,
) -> pd.DataFrame:
    """Backward-compatible wrapper around the canonical timestamp normalizer."""
    del utc
    return standardize_timestamp_column(
        df=df,
        source_column=timestamp_col,
        target_column=new_col,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset ingestion."""
    parser = argparse.ArgumentParser(description="Run RainCheckAI data ingestion.")
    parser.add_argument(
        "--transport-csv",
        type=Path,
        required=True,
        help="Raw transport CSV path.",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        required=True,
        help="Raw weather CSV path.",
    )
    parser.add_argument(
        "--events-csv", type=Path, default=None, help="Optional raw event CSV path."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for cleaned outputs."
    )
    return parser.parse_args()


def main() -> None:
    """Execute the ingestion CLI."""
    configure_logging()
    args = parse_args()
    bundle = load_data_bundle(
        transport_path=args.transport_csv,
        weather_path=args.weather_csv,
        events_path=args.events_csv,
    )
    write_clean_data_bundle(bundle=bundle, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
