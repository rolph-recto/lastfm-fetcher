#!/usr/bin/env python3
"""
Ephemeral script to convert listen.csv to listen.parquet
Run this once to migrate existing CSV data to Parquet format
"""

import argparse
import csv
import sys
from pathlib import Path

import polars as pl


def convert_csv_to_parquet(csv_file: str, parquet_file: str):
    """Convert listen.csv to listen.parquet using Polars"""
    csv_path = Path(csv_file)

    if not csv_path.exists():
        print(f"Error: {csv_file} not found")
        sys.exit(1)

    print(f"Reading {csv_file}...")

    # Read CSV with Python's csv module (more tolerant of malformed data)
    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) >= 3:
                try:
                    timestamp = int(row[0])
                    track = row[1]
                    artist = row[2]
                    data.append((timestamp, track, artist))
                except (ValueError, IndexError):
                    # Skip malformed rows
                    continue

    print(f"Loaded {len(data)} tracks from CSV")

    if not data:
        print("No data to convert!")
        sys.exit(1)

    # Create Polars DataFrame
    df = pl.DataFrame(
        data,
        schema=["timestamp", "track", "artist"],
        orient="row",
    )

    # Deduplicate by timestamp (keep first occurrence)
    df = df.unique(subset=["timestamp"], keep="first")

    # Sort by timestamp descending (most recent first)
    df = df.sort("timestamp", descending=True)

    print(f"After deduplication: {len(df)} unique tracks")

    # Write to Parquet
    df.write_parquet(parquet_file)
    print(f"Successfully converted to {parquet_file}")

    # Show sample data
    print("\nSample data:")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Last.fm listening history from CSV to Parquet"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="listen.csv",
        help="Path to input CSV file (default: listen.csv)",
    )
    parser.add_argument(
        "parquet_file",
        nargs="?",
        default="listen.parquet",
        help="Path to output Parquet file (default: listen.parquet)",
    )
    args = parser.parse_args()
    convert_csv_to_parquet(args.csv_file, args.parquet_file)
