import os
import time
import requests
from statistics import mean

# Clé API lue depuis l'environnement (ALPHAVANTAGE_API_KEY). Aucun fallback en dur.
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"
SYMBOL = "SP5.PAR"


class AlphaVantageRateLimit(Exception):
    pass


def _is_throttle_message(payload: dict) -> bool:
    # Alpha Vantage can return these keys when throttling / errors occur
    return any(k in payload for k in ("Information", "Note", "Error Message"))


def fetch_time_series(function: str, symbol: str, key_name: str, *, max_retries: int = 5) -> dict:
    """
    Fetch a time series from Alpha Vantage with retry/backoff to handle free-tier rate limits.
    """
    params = {"function": function, "symbol": symbol, "apikey": API_KEY}

    # Free tier guidance: avoid bursts; practical minimum is ~1 request / second.
    # We'll enforce a small delay before each call.
    base_sleep = 1.2

    for attempt in range(1, max_retries + 1):
        time.sleep(base_sleep)

        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if _is_throttle_message(data):
            msg = data.get("Information") or data.get("Note") or data.get("Error Message") or "Unknown API message"

            # Exponential backoff: 2s, 4s, 8s, ...
            backoff = min(2 ** attempt, 30)
            if attempt == max_retries:
                raise AlphaVantageRateLimit(
                    f"Alpha Vantage throttling / error after {max_retries} attempts: {msg}"
                )
            print(f"[AlphaVantage] Throttled (attempt {attempt}/{max_retries}). Backing off {backoff}s...")
            time.sleep(backoff)
            continue

        if key_name not in data:
            raise ValueError(f"Missing '{key_name}' in response: {data}")

        return data[key_name]

    raise AlphaVantageRateLimit("Unexpected retry loop exit.")


def moving_average(values, window: int) -> float:
    if len(values) < window:
        raise ValueError(f"Not enough data points for MA({window}). Got {len(values)}.")
    return mean(values[:window])


def build_medium_term_view(symbol: str) -> None:
    # --- Weekly data ---
    weekly = fetch_time_series("TIME_SERIES_WEEKLY", symbol, "Weekly Time Series")
    weekly_dates = sorted(weekly.keys(), reverse=True)
    weekly_closes = [float(weekly[d]["4. close"]) for d in weekly_dates]

    last_week_close = weekly_closes[0]
    ma_26w = moving_average(weekly_closes, 26)

    if len(weekly_closes) < 14:
        raise ValueError("Not enough weekly points to compute 13-week momentum.")
    momentum_13w = (weekly_closes[0] / weekly_closes[13]) - 1

    # --- Monthly data ---
    monthly = fetch_time_series("TIME_SERIES_MONTHLY", symbol, "Monthly Time Series")
    monthly_dates = sorted(monthly.keys(), reverse=True)
    monthly_closes = [float(monthly[d]["4. close"]) for d in monthly_dates]

    last_month_close = monthly_closes[0]
    ma_10m = moving_average(monthly_closes, 10)

    # --- Interpretation (medium-term) ---
    regime = "HAUSSIER" if last_month_close > ma_10m else "BAISSIER"
    trend = "POSITIVE" if last_week_close > ma_26w else "NEGATIVE"

    # Simple decision grid
    if regime == "HAUSSIER" and trend == "POSITIVE" and momentum_13w > 0:
        interpretation = "Tendance haussière moyen terme confirmée (régime + tendance + momentum alignés)."
        stance = "Biais pro-risque (exposition maintenue/renforcée selon contraintes)."
    elif regime == "BAISSIER" and trend == "NEGATIVE" and momentum_13w < 0:
        interpretation = "Tendance baissière moyen terme confirmée (régime + tendance + momentum alignés)."
        stance = "Biais défensif (réduction/hedge/rotation selon contraintes)."
    else:
        interpretation = "Zone de transition : signaux mixtes (risque de faux départ / range)."
        stance = "Neutralité ou gestion prudente (réduction du risque, attentes de confirmation)."

    # --- Display ---
    print(f"\n=== VUE TENDANCE MOYEN TERME — {symbol} ===\n")

    print("Mensuel (régime):")
    print(f"  Dernier close      : {last_month_close:.4f}")
    print(f"  MA 10 mois         : {ma_10m:.4f}")
    print(f"  Régime             : {regime}\n")

    print("Hebdomadaire (tendance & momentum):")
    print(f"  Dernier close      : {last_week_close:.4f}")
    print(f"  MA 26 semaines     : {ma_26w:.4f}")
    print(f"  Momentum ~3 mois   : {momentum_13w*100:.2f}%")
    print(f"  Tendance           : {trend}\n")

    print("Interprétation:")
    print(f"  {interpretation}")
    print(f"  Posture suggérée   : {stance}\n")


if __name__ == "__main__":
    build_medium_term_view(SYMBOL)
