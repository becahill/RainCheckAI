import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure basic logging for the module.

    Parameters
    ----------
    log_level:
        Logging level from the ``logging`` module.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path:
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    LOGGER.info("Loading CSV from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    LOGGER.info("Loaded %d rows and %d columns from %s", len(df), df.shape[1], path)
    return df


def standardise_timestamp(
    df: pd.DataFrame,
    timestamp_col: str,
    new_col: str = "timestamp",
    utc: bool = True,
) -> pd.DataFrame:
    """Convert a timestamp column to ``datetime64[ns]`` in a standard format.

    Parameters
    ----------
    df:
        Input DataFrame.
    timestamp_col:
        Name of the column containing timestamp strings.
    new_col:
        Name of the standardised timestamp column.
    utc:
        If ``True``, localise/convert timestamps to UTC.

    Returns
    -------
    pd.DataFrame
        DataFrame with a standardised timestamp column.
    """
    df = df.copy()

    LOGGER.info("Standardising timestamp column '%s' -> '%s'", timestamp_col, new_col)
    df[new_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=utc)

    invalid_count = df[new_col].isna().sum()
    if invalid_count > 0:
        LOGGER.warning(
            "Found %d invalid timestamps in column '%s' (coerced to NaT)",
            invalid_count,
            timestamp_col,
        )

    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy_numeric: str = "median",
    strategy_categorical: str = "mode",
) -> pd.DataFrame:
    """Handle missing values with simple, explicit strategies.

    Parameters
    ----------
    df:
        Input DataFrame.
    strategy_numeric:
        Strategy for numeric columns (``median`` or ``mean``).
    strategy_categorical:
        Strategy for categorical/object columns (currently supports ``mode``).

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    LOGGER.info(
        "Handling missing values: %d numeric, %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )

    for col in numeric_cols:
        if df[col].isna().any():
            if strategy_numeric == "median":
                fill_value = df[col].median()
            elif strategy_numeric == "mean":
                fill_value = df[col].mean()
            else:
                raise ValueError(f"Unsupported numeric strategy: {strategy_numeric}")
            LOGGER.debug("Imputing numeric column '%s' with %s", col, fill_value)
            df[col] = df[col].fillna(fill_value)

    if strategy_categorical != "mode":
        raise ValueError(f"Unsupported categorical strategy: {strategy_categorical}")

    for col in categorical_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "UNKNOWN"
            LOGGER.debug("Imputing categorical column '%s' with %s", col, fill_value)
            df[col] = df[col].fillna(fill_value)

    return df


def load_transport_and_weather(
    transport_path: Path,
    weather_path: Path,
    transport_timestamp_col: str = "timestamp",
    weather_timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean transport and weather datasets.

    This function is intentionally generic so it can be adapted to
    different timestamp column names by adjusting the arguments.

    Parameters
    ----------
    transport_path:
        Path to the transport CSV file.
    weather_path:
        Path to the weather CSV file.
    transport_timestamp_col:
        Name of the timestamp column in the transport CSV.
    weather_timestamp_col:
        Name of the timestamp column in the weather CSV.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Cleaned transport and weather DataFrames.
    """
    transport_df = load_csv(transport_path)
    weather_df = load_csv(weather_path)

    transport_df = standardise_timestamp(
        df=transport_df,
        timestamp_col=transport_timestamp_col,
        new_col="timestamp",
        utc=True,
    )
    weather_df = standardise_timestamp(
        df=weather_df,
        timestamp_col=weather_timestamp_col,
        new_col="timestamp",
        utc=True,
    )

    transport_df = handle_missing_values(transport_df)
    weather_df = handle_missing_values(weather_df)

    LOGGER.info("Finished loading and cleaning transport and weather data.")
    return transport_df, weather_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest and clean transport and weather CSV data for RainCheckAI.",
    )
    parser.add_argument(
        "--transport-csv",
        type=Path,
        required=True,
        help="Path to the transport CSV file.",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        required=True,
        help="Path to the weather CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cleaned CSVs will be written.",
    )
    parser.add_argument(
        "--transport-timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name in the transport CSV.",
    )
    parser.add_argument(
        "--weather-timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name in the weather CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution."""
    configure_logging()
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory set to %s", args.output_dir)

    transport_df, weather_df = load_transport_and_weather(
        transport_path=args.transport_csv,
        weather_path=args.weather_csv,
        transport_timestamp_col=args.transport_timestamp_col,
        weather_timestamp_col=args.weather_timestamp_col,
    )

    transport_out = args.output_dir / "transport_clean.csv"
    weather_out = args.output_dir / "weather_clean.csv"

    LOGGER.info("Writing cleaned transport data to %s", transport_out)
    transport_df.to_csv(transport_out, index=False)

    LOGGER.info("Writing cleaned weather data to %s", weather_out)
    weather_df.to_csv(weather_out, index=False)

    LOGGER.info("Ingestion complete.")


if __name__ == "__main__":
    main()

