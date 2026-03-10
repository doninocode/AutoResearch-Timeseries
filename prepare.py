"""
Prepare futures market data for the autoresearch loop.

Examples:
    uv run prepare.py --synthetic-demo
    uv run prepare.py --input data/raw/my_futures.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.autoresearch_futures.data import (
    generate_demo_market_data,
    prepare_market_data,
)
from src.autoresearch_futures.settings import (
    DEFAULT_DEMO_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    METADATA_PATH,
    PREPARED_DATA_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare futures data for 3-day move classification.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to a CSV or parquet file with daily futures bars.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Generate a synthetic daily futures dataset before preparing the cache.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Fraction of timestamps allocated to the training split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of timestamps allocated to the validation split.",
    )
    parser.add_argument(
        "--move-threshold-sigma",
        type=float,
        default=0.50,
        help="Volatility-scaled threshold used to map future returns into down/flat/up classes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = args.input
    if args.synthetic_demo:
        input_path = generate_demo_market_data(DEFAULT_DEMO_DATA_PATH)
        print(f"Synthetic demo dataset written to {input_path}")

    metadata = prepare_market_data(
        input_path=input_path,
        output_path=PREPARED_DATA_PATH,
        metadata_path=METADATA_PATH,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        move_threshold_sigma=args.move_threshold_sigma,
    )

    print(f"Prepared data: {PREPARED_DATA_PATH}")
    print(f"Metadata:      {METADATA_PATH}")
    print(f"Input path:    {metadata.input_path}")
    print(f"Rows:          {metadata.rows}")
    print(f"Symbols:       {', '.join(metadata.symbols)}")
    print(f"Features:      {', '.join(metadata.feature_columns)}")
    print(f"Train end:     {metadata.train_end}")
    print(f"Val end:       {metadata.val_end}")
    print(f"Split counts:  {metadata.split_counts}")
    print(f"Label counts:  {metadata.label_counts}")


if __name__ == "__main__":
    main()
