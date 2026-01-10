import os
import requests

# Clé Alpha Vantage lue depuis l'environnement (ALPHAVANTAGE_API_KEY).
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

def symbol_search(etf_name: str) -> dict:
    """
    Execute Alpha Vantage SYMBOL_SEARCH for a given ETF name.
    Returns the JSON response as a Python dict.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": etf_name,
        "apikey": API_KEY
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()  # raise exception if HTTP error

    return response.json()


if __name__ == "__main__":
    etf_name = "Amundi EURO STOXX 50"

    result = symbol_search(etf_name)

    # affichage lisible
    for match in result.get("bestMatches", []):
        print(
            match.get('1. symbol'),",'",
            match.get('2. name'),"',",
            match.get('3. type'),",",
            match.get('4. region'),",",
            match.get('8. currency')
        )
