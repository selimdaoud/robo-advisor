import os
import requests

# Clé Alpha Vantage lue depuis l'environnement (ALPHAVANTAGE_API_KEY).
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"


def fetch_last_10_cotations(symbol: str) -> None:
    params = {
        "function": "TIME_SERIES_WEEKLY",
        "symbol": symbol,
        "apikey": API_KEY
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if "Weekly Time Series" not in data:
        raise ValueError(f"No weekly time series returned: {data}")

    time_series = data["Weekly Time Series"]

    # Sort dates descending and take the last 12 weeks
    last_12_dates = sorted(time_series.keys(), reverse=True)[:12]

    print(f"\nLast 4 weekly quotations for {symbol}:\n")

    for date in last_12_dates:
        d = time_series[date]
        print(
            f" {date} | "
            f"Close: {d['4. close']} | "
        )


if __name__ == "__main__":
    symbol = "SP5.PAR"  # example
    fetch_last_10_cotations(symbol)
