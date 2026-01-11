import os
import sys
import time
import requests

BASE_URL = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


def fetch_last_close(symbol: str) -> float:
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": API_KEY}
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "Global Quote" not in data or "05. price" not in data["Global Quote"]:
        raise ValueError(f"Réponse invalide pour {symbol}: {data}")
    return float(data["Global Quote"]["05. price"])


def main():
    if not API_KEY:
        print("ALPHAVANTAGE_API_KEY manquant dans l'environnement", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: python fetch_quotes_total.py SYM1 SYM2 ...", file=sys.stderr)
        sys.exit(1)

    symbols = sys.argv[1:]
    total = 0.0
    for idx, sym in enumerate(symbols, 1):
        try:
            price = fetch_last_close(sym)
            total += price
            print(f"{sym}: {price:.4f}")
        except Exception as exc:
            print(f"{sym}: ERREUR {exc}")
        # Alpha Vantage free tier : petite pause pour éviter le throttling
        if idx < len(symbols):
            time.sleep(1.2)

    print(f"\nTotal: {total:.4f}")


if __name__ == "__main__":
    main()
