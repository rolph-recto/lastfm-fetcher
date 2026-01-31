#!/usr/bin/env python3
"""
Last.fm listening history fetcher

Fetches scrobbled tracks from Last.fm API and saves to Parquet files.
Supports multiple users via JSON configuration.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from dotenv import load_dotenv
from matplotlib.patches import Polygon
from pydantic import BaseModel, Field
from tqdm import tqdm

# Load .env file for API_KEY
load_dotenv()

_api_key = os.getenv("API_KEY")
if not _api_key:
    print("Error: API_KEY not found in .env file")
    sys.exit(1)

API_KEY: str = _api_key


class UserConfig(BaseModel):
    """Configuration for a single Last.fm user"""

    user: str = Field(..., description="Last.fm username")
    page_limit: int = Field(default=200, description="Number of tracks per page")
    listen_file: str = Field(..., description="Path to Parquet file for this user")


class Config(BaseModel):
    """Root configuration containing multiple user configs"""

    users: List[UserConfig] = Field(..., description="List of user configurations")


def recent_tracks_url(
    api_key: str,
    user: str,
    page: int,
    limit: int = 200,
    from_ts: Optional[int] = None,
    to_ts: Optional[int] = None,
) -> str:
    """Generate URL for Last.fm recent tracks API"""
    url = (
        f"https://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks"
        f"&user={user}&limit={limit}&page={page}&api_key={api_key}&format=json&extended=0"
    )
    if from_ts is not None:
        url += f"&from={from_ts}"
    if to_ts is not None:
        url += f"&to={to_ts}"
    return url


def process_page(page: dict) -> List[tuple]:
    """Extract track data from API response page"""
    listen_map = []
    for track in page["recenttracks"]["track"]:
        if "date" in track:
            listen_map.append(
                (int(track["date"]["uts"]), track["name"], track["artist"]["#text"])
            )
    return listen_map


def fetch_tracks_for_user(
    user_config: UserConfig,
    rate_limit: float,
    existing_df: pl.DataFrame,
    listen_file: str,
    existing_timestamps: Optional[Set[int]] = None,
    from_ts: Optional[int] = None,
    to_ts: Optional[int] = None,
    max_retries: int = 3,
) -> pl.DataFrame:
    """
    Fetch tracks for a single user from Last.fm API.
    Saves combined dataframe after each page is processed.
    Returns DataFrame of all tracks (existing + new).
    """
    new_tracks = []
    page = 1
    should_continue = True

    if existing_timestamps is None:
        existing_timestamps = set()

    while should_continue:
        print(f"Fetching page {page} for user '{user_config.user}'...")

        retry_count = 0
        success = False

        while retry_count <= max_retries and not success:
            try:
                response = requests.get(
                    recent_tracks_url(
                        API_KEY,
                        user_config.user,
                        page,
                        user_config.page_limit,
                        from_ts,
                        to_ts,
                    )
                )
                response.raise_for_status()
                data = response.json()

                # Check for API-level errors (e.g., invalid API key)
                if "error" in data:
                    error_msg = data.get("message", "Unknown API error")
                    print(f"API error on page {page}: {error_msg}")
                    # Don't retry on authentication errors
                    if data.get("error") in [
                        6,
                        8,
                        9,
                        10,
                        26,
                    ]:  # Invalid API key, session, etc.
                        print("Authentication error - not retrying")
                        should_continue = False
                    break

                if "recenttracks" not in data or "track" not in data["recenttracks"]:
                    print(f"No tracks found on page {page}")
                    should_continue = False
                    break

                page_tracks = process_page(data)

                if not page_tracks:
                    print(f"No new tracks on page {page}")
                    should_continue = False
                    break

                # Filter out existing tracks
                page_new_tracks = [
                    track
                    for track in page_tracks
                    if track[0] not in existing_timestamps
                ]

                if len(page_new_tracks) == 0:
                    print(f"All tracks on page {page} already exist. Stopping.")
                    should_continue = False
                    break

                new_tracks.extend(page_new_tracks)
                print(f"Found {len(page_new_tracks)} new tracks on page {page}")

                # Print first and last track timestamps
                first_ts = page_new_tracks[0][0]
                last_ts = page_new_tracks[-1][0]
                first_dt = datetime.fromtimestamp(first_ts)
                last_dt = datetime.fromtimestamp(last_ts)
                print(f"  First track: {first_dt} (Unix: {first_ts})")
                print(f"  Last track:  {last_dt} (Unix: {last_ts})")

                # Save incremental progress after each page
                if new_tracks:
                    page_df = pl.DataFrame(
                        new_tracks,
                        schema=["timestamp", "track", "artist"],
                        orient="row",
                    )
                    combined_df = pl.concat([existing_df, page_df])
                    save_dataframe(combined_df, listen_file)
                    print(f"Saved progress: {len(combined_df)} total tracks")
                    # Update existing_df for next iteration
                    existing_df = combined_df
                    existing_timestamps = set(existing_df["timestamp"].to_list())
                    # Clear new_tracks since they're now saved
                    new_tracks = []

                # Check if we've reached the last page
                total_pages = int(data["recenttracks"]["@attr"]["totalPages"])
                if page >= total_pages:
                    should_continue = False
                    break

                success = True
                page += 1
                print(f"Waiting for {rate_limit} seconds...")
                for _ in tqdm(range(int(rate_limit))):
                    time.sleep(1)

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Error fetching page {page}: {e}")
                    print(f"Max retries ({max_retries}) exceeded. Stopping.")
                    should_continue = False
                    break
                else:
                    print(f"Error fetching page {page}: {e}")
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    print(f"Waiting for {rate_limit} seconds before retry...")
                    for _ in tqdm(range(int(rate_limit))):
                        time.sleep(1)
            except Exception as e:
                print(f"Unexpected error on page {page}: {e}")
                should_continue = False
                break

    # Return the final combined dataframe
    return existing_df


def load_or_create_dataframe(listen_file: str) -> pl.DataFrame:
    """Load existing Parquet file or create empty DataFrame"""
    file_path = Path(listen_file)

    if file_path.exists():
        print(f"Loading existing data from {listen_file}...")
        return pl.read_parquet(listen_file)
    else:
        print(f"No existing file found at {listen_file}. Will create new file.")
        return pl.DataFrame(
            schema={"timestamp": pl.Int64, "track": pl.Utf8, "artist": pl.Utf8}
        )


def save_dataframe(df: pl.DataFrame, listen_file: str):
    """Save DataFrame to Parquet file"""
    # Deduplicate by timestamp (keep first occurrence)
    df = df.unique(subset=["timestamp"], keep="first")

    # Sort by timestamp descending (most recent first)
    df = df.sort("timestamp", descending=True)

    df.write_parquet(listen_file)
    print(f"Saved {len(df)} tracks to {listen_file}")


def fetch_command(args):
    """Handle the fetch command"""
    # Load config
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            config_data = json.load(f)
        config = Config.model_validate(config_data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error validating config: {e}")
        sys.exit(1)

        print(f"Loaded configuration for {len(config.users)} user(s)")
        print(f"Rate limit: {args.rate_limit} second(s) between requests")
        if args.from_ts:
            print(f"From timestamp: {args.from_ts}")
        if args.to_ts:
            print(f"To timestamp: {args.to_ts}")
        print()

    for user_config in config.users:
        print(f"\n{'=' * 50}")
        print(f"Processing user: {user_config.user}")
        print(f"Output file: {user_config.listen_file}")
        print(f"{'=' * 50}\n")

        # Load existing data
        existing_df = load_or_create_dataframe(user_config.listen_file)
        existing_timestamps = set(existing_df["timestamp"].to_list())

        print(f"Existing tracks: {len(existing_df)}")
        if existing_timestamps:
            latest_ts = max(existing_timestamps)
            latest_dt = datetime.fromtimestamp(latest_ts)
            print(f"Latest track: {latest_dt}")

        # Fetch new tracks (saves incrementally after each page)
        final_df = fetch_tracks_for_user(
            user_config,
            args.rate_limit,
            existing_df,
            user_config.listen_file,
            existing_timestamps,
            args.from_ts,
            args.to_ts,
            args.max_retries,
        )

        new_count = len(final_df) - len(existing_df)
        if new_count > 0:
            print(f"Added {new_count} new tracks")
        else:
            print("No new tracks to add")

        print()


def detect_gaps_command(args):
    """Handle the detect-gaps command"""
    # Load parquet file
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found: {args.parquet_file}")
        sys.exit(1)

    print(f"Loading listening history from {args.parquet_file}...")
    df = pl.read_parquet(args.parquet_file)

    if len(df) == 0:
        print("Error: No listening history found in parquet file")
        sys.exit(1)

    # Sort by timestamp
    df = df.sort("timestamp")

    # Get timestamps as a list
    timestamps = df["timestamp"].to_list()

    # Convert gap days to seconds
    gap_threshold_seconds = args.days * 24 * 3600

    # Find gaps
    gaps = []
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > gap_threshold_seconds:
            gaps.append(
                {
                    "start_ts": timestamps[i - 1],
                    "end_ts": timestamps[i],
                    "days": gap / (24 * 3600),
                }
            )

    if not gaps:
        print(f"No gaps larger than {args.days} days found in listening history.")
        return

    print(f"\nFound {len(gaps)} gap(s) larger than {args.days} days:\n")
    for i, gap in enumerate(gaps, 1):
        start_dt = datetime.fromtimestamp(gap["start_ts"])
        end_dt = datetime.fromtimestamp(gap["end_ts"])
        print(f"{i}. Start: {start_dt} (Unix: {gap['start_ts']})")
        print(f"   End:   {end_dt} (Unix: {gap['end_ts']})")
        print(f"   Duration: {gap['days']:.2f} days")
        print(
            f"\n   Fetch command: python lastfm.py fetch --from {gap['start_ts']} --to {gap['end_ts']}\n"
        )


def plot_command(args):
    """Handle the plot command"""
    # Load parquet file
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found: {args.parquet_file}")
        sys.exit(1)

    print(f"Loading listening history from {args.parquet_file}...")
    df = pl.read_parquet(args.parquet_file)

    if len(df) == 0:
        print("Error: No listening history found in parquet file")
        sys.exit(1)

    # Convert timestamps to datetime
    df = df.with_columns(
        pl.from_epoch("timestamp", time_unit="s").alias("datetime")
    ).with_columns(pl.col("datetime").dt.date().alias("date"))

    # Parse date range if specified
    start_date = None
    end_date = None
    extended_start_date = None

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        # Calculate extended start date to include lookback data for rolling window
        extended_start_date = start_date - timedelta(days=args.window)
        print(
            f"Filtering from {args.start_date} (with {args.window} day lookback from {extended_start_date})"
        )

    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        print(f"Filtering until {args.end_date}")

    # Filter with extended start date to get lookback data for rolling window
    if extended_start_date:
        df = df.filter(pl.col("date") >= extended_start_date)
    if end_date:
        df = df.filter(pl.col("date") <= end_date)

    if len(df) == 0:
        print("Error: No data remaining after date filtering")
        sys.exit(1)

    # Aggregate daily listen counts per artist for ALL artists
    daily_counts_all = (
        df.group_by(["date", "artist"]).agg(pl.len().alias("listens")).sort("date")
    )

    # Create pivot table: dates as rows, artists as columns
    pivot_df = daily_counts_all.pivot(
        "artist",
        index="date",
        values="listens",
    )

    # Fill missing values with 0
    pivot_df = pivot_df.fill_null(0)

    # Get window size for smoothing
    dates = pivot_df["date"].to_list()
    window_size = min(args.window, len(dates))

    # Get artist columns (exclude date column)
    artist_cols_all = [col for col in pivot_df.columns if col != "date"]

    print(f"Found {len(artist_cols_all)} total artists")

    # Apply Gaussian-weighted rolling mean to each artist column
    if window_size > 1:
        print(f"Applying Gaussian rolling smooth (window={window_size})...")
        # Create Gaussian weights centered on the middle of the window
        x_weights = np.arange(window_size) - window_size // 2
        sigma = window_size / 4.3  # Equivalent to pandas std=7 for window=30
        gaussian_weights = np.exp(-(x_weights**2) / (2 * sigma**2))
        gaussian_weights = gaussian_weights / gaussian_weights.sum()
        weights_list = gaussian_weights.tolist()

        # Apply weighted rolling mean to each artist column
        for artist in artist_cols_all:
            pivot_df = pivot_df.with_columns(
                pl.col(artist)
                .rolling_mean(window_size=window_size, weights=weights_list)
                .fill_null(0)
                .alias(artist)
            )

    # Calculate total smoothed listens per artist and select top N
    print(f"Selecting top {args.top_n} artists by smoothed total listens...")

    # Calculate totals for each artist from the smoothed data
    artist_totals = []
    for artist in artist_cols_all:
        total = pivot_df[artist].sum()
        artist_totals.append((artist, total))

    # Sort by total and get top N
    artist_totals.sort(key=lambda x: x[1], reverse=True)
    top_artists = [artist for artist, _ in artist_totals[: args.top_n]]

    print(f"Selected top {len(top_artists)} artists for plotting")

    # Filter pivot_df to only top artists (keep date column)
    cols_to_keep = ["date"] + top_artists
    pivot_df = pivot_df.select(cols_to_keep)

    # Filter back to original date range (remove lookback data)
    if start_date:
        pivot_df = pivot_df.filter(pl.col("date") >= start_date)

    # Get dates and artist names for plotting
    dates = pivot_df["date"].to_list()
    artist_cols = [col for col in pivot_df.columns if col != "date"]

    if len(dates) < 2:
        print("Error: Need at least 2 days of data to create plot")
        sys.exit(1)

    # Convert to numpy arrays
    x = np.arange(len(dates))
    y_data = []
    for artist in artist_cols:
        y_data.append(pivot_df[artist].to_numpy())

    y_data = np.array(y_data)

    # Create plot with baseline calculation
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate baseline using stack layout (centered)
    n_artists, n_days = y_data.shape
    y_stack = np.zeros((n_artists, n_days))

    # Calculate cumulative sums for stacking
    cumulative = np.zeros(n_days)
    for i in range(n_artists):
        y_stack[i] = cumulative.copy()
        cumulative += y_data[i]

    # Center the plot around zero
    total_height = cumulative
    baseline_shift = total_height / 2

    # Create the plot with smooth curves
    for i in range(n_artists):
        y_bottom = y_stack[i] - baseline_shift
        y_top = y_stack[i] + y_data[i] - baseline_shift

        # Create polygon vertices
        x_extended = np.concatenate([[x[0]], x, [x[-1]]])
        y_top_extended = np.concatenate([[y_top[0]], y_top, [y_top[-1]]])
        y_bottom_extended = np.concatenate([[y_bottom[0]], y_bottom, [y_bottom[-1]]])

        vertices = list(zip(x_extended, y_top_extended)) + list(
            zip(x_extended[::-1], y_bottom_extended[::-1])
        )

        # Create filled polygon
        poly = Polygon(
            vertices, facecolor=plt.cm.tab20(i % 20), alpha=0.8, edgecolor="none"
        )
        ax.add_patch(poly)

    # Configure plot
    ax.set_xlim(x[0], x[-1])
    max_h = max(total_height)
    if np.isnan(max_h) or np.isinf(max_h) or max_h < 0.1:
        # Handle edge case where all data is zero, NaN, or Inf
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(-max_h / 2 * 1.1, max_h / 2 * 1.1)

    # Set x-axis labels (show ~10 dates)
    n_ticks = min(10, len(dates))
    tick_indices = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [dates[i].strftime("%Y-%m-%d") for i in tick_indices], rotation=45, ha="right"
    )

    ax.set_ylabel(f"Daily Listens ({window_size} day rolling average)", fontsize=12)
    ax.set_yticks([])
    ax.set_title(
        f"Listening History Streamgraph - Top {len(artist_cols)} Artists",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    ax.legend(
        [
            plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.tab20(i % 20), alpha=0.8)
            for i in range(len(artist_cols))
        ],
        artist_cols,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=9,
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Streamgraph saved to {args.output}")
    print(f"Artists shown: {', '.join(artist_cols)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Last.fm listening history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lastfm.py fetch                    # Use default config.json
  python lastfm.py fetch --config myconfig.json
  python lastfm.py fetch --rate-limit 0.5
  python lastfm.py fetch --from 1684165381 --to 1685022753
  python lastfm.py gaps history.parquet
  python lastfm.py gaps history.parquet --days 14
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch new listening history")
    fetch_parser.add_argument(
        "--config",
        default="config.json",
        help="Path to JSON config file (default: config.json)",
    )
    fetch_parser.add_argument(
        "--rate-limit",
        type=int,
        default=15,
        help="Seconds to wait between API requests (default: 15)",
    )
    fetch_parser.add_argument(
        "--from",
        dest="from_ts",
        type=int,
        help="Start timestamp (Unix seconds) for date range filtering",
    )
    fetch_parser.add_argument(
        "--to",
        dest="to_ts",
        type=int,
        help="End timestamp (Unix seconds) for date range filtering",
    )
    fetch_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per page on API failure (default: 3)",
    )

    # Streamgraph command
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plot plot of listening history"
    )
    plot_parser.add_argument(
        "parquet_file",
        help="Path to parquet file containing listening history",
    )
    plot_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to output PNG file",
    )
    plot_parser.add_argument(
        "--start-date",
        help="Start date for filtering (YYYY-MM-DD format)",
    )
    plot_parser.add_argument(
        "--end-date",
        help="End date for filtering (YYYY-MM-DD format)",
    )
    plot_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top artists to display (default: 10)",
    )
    plot_parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Rolling average window size in days (default: 7)",
    )

    # Detect-gaps command
    detect_gaps_parser = subparsers.add_parser(
        "gaps", help="Detect gaps in listening history"
    )
    detect_gaps_parser.add_argument(
        "parquet_file",
        help="Path to parquet file containing listening history",
    )
    detect_gaps_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Minimum gap size in days to report (default: 7)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "fetch":
        fetch_command(args)
    elif args.command == "gaps":
        detect_gaps_command(args)
    elif args.command == "plot":
        plot_command(args)


if __name__ == "__main__":
    main()
