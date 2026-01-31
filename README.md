# Last.fm Listening History Fetcher and Visualizer

A Python tool to fetch and visualize your Last.fm listening history.

## Configuration

1. **Get a Last.fm API key**:
   - Visit https://www.last.fm/api/account/create
   - Create an API account to get your API key

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   API_KEY=your_lastfm_api_key_here
   ```

3. **Configure users**:
   Create a `config.json` file:
   ```json
   {
     "users": [
       {
         "user": "your_lastfm_username",
         "page_limit": 200,
         "listen_file": "listen.parquet"
       }
     ]
   }
   ```

## Usage

### `fetch` - Download Listening History

Fetches scrobbled tracks from Last.fm and saves to Parquet files.

```bash
# Fetch using default config.json
python lastfm.py fetch

# Specify custom config file
python lastfm.py fetch --config myconfig.json

# Adjust rate limiting (seconds between API requests)
python lastfm.py fetch --rate-limit 30

# Fetch specific date range (Unix timestamps)
python lastfm.py fetch --from 1684165381 --to 1685022753

# Configure retry attempts on API failures
python lastfm.py fetch --max-retries 5
```

### `gaps` - Detect Missing Data

Identifies gaps in your listening history larger than a specified threshold.

```bash
# Find gaps larger than 7 days (default)
python lastfm.py gaps listen.parquet

# Find gaps larger than 14 days
python lastfm.py gaps listen.parquet --days 14
```

### `plot` - Generate Visualization

Creates a streamgraph visualization of your listening history.

```bash
# Basic plot with top 10 artists
python lastfm.py plot listen.parquet -o output.png

# Specify date range
python lastfm.py plot listen.parquet -o output.png --start-date 2024-01-01 --end-date 2024-12-31

# Show top 20 artists with 14-day rolling average
python lastfm.py plot listen.parquet -o output.png --top-n 20 --window 14
```

