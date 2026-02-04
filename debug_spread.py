from dotenv import load_dotenv
import os
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.exceptions import V20Error

load_dotenv()

ACCESS_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
INSTRUMENTS = ["USD_JPY", "GBP_USD", "EUR_USD", "GBP_JPY", "XAU_USD"]

def pip_size_for(instrument: str) -> float:
    """
    Match bot.py:
      - XAU pip size = 0.1 (i.e., $0.10 = 1 pip)
      - JPY pip size = 0.01
      - Others = 0.0001
    """
    if "XAU" in instrument:
        return 0.1
    if instrument.endswith("_JPY"):
        return 0.01
    return 0.0001

def debug_spreads():
    if not ACCESS_TOKEN or not ACCOUNT_ID:
        print("[!] Missing OANDA_TOKEN or OANDA_ACCOUNT_ID in env.")
        return

    client = oandapyV20.API(access_token=ACCESS_TOKEN)
    params = {"instruments": ",".join(INSTRUMENTS)}

    try:
        r = pricing.PricingInfo(accountID=ACCOUNT_ID, params=params)
        response = client.request(r)

        print(f"{'PAIR':<10} | {'BID':<12} | {'ASK':<12} | {'DIFF':<12} | {'PIP_SIZE':<8} | {'CALC SPREAD (Pips)':<20} | SOURCE")
        print("-" * 105)

        for price in response.get("prices", []):
            instrument = price.get("instrument", "UNKNOWN")

            # Prefer raw top-of-book bids/asks if present
            bid = None
            ask = None
            source = "raw"

            try:
                if price.get("bids") and len(price["bids"]) > 0:
                    bid = float(price["bids"][0]["price"])
                if price.get("asks") and len(price["asks"]) > 0:
                    ask = float(price["asks"][0]["price"])
            except Exception:
                bid, ask = None, None

            # Fallback to closeout if raw is missing
            if bid is None or ask is None:
                source = "closeout"
                try:
                    bid = float(price.get("closeoutBid"))
                    ask = float(price.get("closeoutAsk"))
                except Exception:
                    bid, ask = None, None

            if bid is None or ask is None:
                print(f"{instrument:<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<8} | {'N/A':<20} | missing")
                continue

            diff = ask - bid
            pip_size = pip_size_for(instrument)
            spread_pips = diff / pip_size

            print(f"{instrument:<10} | {bid:<12.5f} | {ask:<12.5f} | {diff:<12.5f} | {pip_size:<8.4f} | {spread_pips:<20.2f} | {source}")

    except V20Error as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    print("\n--- OANDA SPREAD DEBUGGER (BOT-ALIGNED) ---\n")
    debug_spreads()
    print("\n------------------------------------------\n")
