"""CLI entrypoint for RainCheckAI model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from raincheckai.config import ArtifactPaths
from raincheckai.logging_utils import configure_logging
from raincheckai.training import train_and_persist


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train the RainCheckAI model bundle.")
    parser.add_argument(
        "--engineered-csv",
        type=Path,
        required=True,
        help="Engineered dataset CSV path.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where model artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the training CLI."""
    configure_logging()
    args = parse_args()
    train_and_persist(
        engineered_dataset_path=args.engineered_csv,
        artifact_paths=ArtifactPaths(root_dir=args.artifact_dir),
    )


if __name__ == "__main__":
    main()
