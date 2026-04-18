"""CLI entrypoint for RainCheckAI feature engineering."""

from __future__ import annotations

import argparse
from pathlib import Path

from raincheckai.feature_engineering import add_cyclical_time_encoding, engineer_training_dataset
from raincheckai.ingestion import load_data_bundle
from raincheckai.logging_utils import configure_logging

__all__ = ["add_cyclical_time_encoding", "engineer_training_dataset"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature engineering."""
    parser = argparse.ArgumentParser(description="Run RainCheckAI feature engineering.")
    parser.add_argument(
        "--transport-csv",
        type=Path,
        required=True,
        help="Clean transport CSV path.",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        required=True,
        help="Clean weather CSV path.",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=None,
        help="Optional clean event CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output engineered CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the feature-engineering CLI."""
    configure_logging()
    args = parse_args()
    bundle = load_data_bundle(
        transport_path=args.transport_csv,
        weather_path=args.weather_csv,
        events_path=args.events_csv,
    )
    engineered = engineer_training_dataset(bundle=bundle)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
