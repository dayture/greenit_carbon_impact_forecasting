"""
fetch_entsoe_data.py

Fetches day-ahead prices, total load, and/or generation-by-type data
from the ENTSO-E Transparency Platform for a given bidding zone, and
writes each to CSV under data/.

Requires an ENTSO-E API key, read from the ENTSOE_API_KEY environment
variable (set it in a .env file at the project root - see .env.example).
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

try:
    from entsoe import EntsoePandasClient
except ImportError:
    print("Error: entsoe-py is not installed. Run: pip install entsoe-py")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_api_key() -> str:
    """Retrieve and validate the ENTSO-E API key from environment variables."""
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key or api_key.strip() == "YOUR_API_KEY_HERE":
        print(f"\nENTSOE_API_KEY is not set. Add it to '{ENV_PATH}':")
        print("   ENTSOE_API_KEY=your_actual_api_key\n")
        sys.exit(1)
    return api_key.strip()


def fetch_data(country_code: str, data_type: str, start_date: str, end_date: str, output_dir: Path):
    """Fetch the requested data type(s) from ENTSO-E and save each to CSV."""
    api_key = get_api_key()
    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching ENTSO-E data: country={country_code}, type={data_type}, {start} -> {end}")

    results = {}
    try:
        if data_type in ["prices", "all"]:
            prices = client.query_day_ahead_prices(country_code, start=start, end=end)
            results["day_ahead_prices"] = prices
            csv_path = output_dir / f"{country_code}_day_ahead_prices_{start_date}_{end_date}.csv"
            prices.to_csv(csv_path)
            print(f"Saved: {csv_path}")

        if data_type in ["load", "all"]:
            load = client.query_load(country_code, start=start, end=end)
            results["load"] = load
            csv_path = output_dir / f"{country_code}_total_load_{start_date}_{end_date}.csv"
            load.to_csv(csv_path)
            print(f"Saved: {csv_path}")

        if data_type in ["generation", "all"]:
            generation = client.query_generation(country_code, start=start, end=end)
            results["generation"] = generation
            csv_path = output_dir / f"{country_code}_generation_{start_date}_{end_date}.csv"
            generation.to_csv(csv_path)
            print(f"Saved: {csv_path}")

        return results

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        print("Check that your ENTSO-E API key is active and valid.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ENTSO-E Transparency Platform data fetcher")
    parser.add_argument("--country", type=str, default="DE_LU",
                         help="Bidding zone code (e.g. DE_LU, BE, FR, TR, NL)")
    parser.add_argument("--type", type=str, choices=["prices", "load", "generation", "all"],
                         default="prices", help="Data type to fetch")
    parser.add_argument("--start", type=str,
                         default=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                         help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str,
                         default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
                         help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    fetch_data(
        country_code=args.country,
        data_type=args.type,
        start_date=args.start,
        end_date=args.end,
        output_dir=PROJECT_ROOT / "data",
    )


if __name__ == "__main__":
    main()
