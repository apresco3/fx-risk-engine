from dotenv import load_dotenv
import os
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.exceptions import V20Error

load_dotenv()

# --- CONFIGURATION (Update these or import from your config file) ---
ACCESS_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
INSTRUMENTS = ["AUD_USD", "USD_JPY", "XAU_USD"]
# -------------------------------------------------------------------

def get_pip_multiplier(instrument):
    """Returns the multiplier to convert price difference to pips."""
    if "_JPY" in instrument or "XAU" in instrument: # JPY pairs and Gold usually 2 decimal places
        return 100
    return 10000

def debug_spreads():
    client = oandapyV20.API(access_token=ACCESS_TOKEN)
    params = {"instruments": ",".join(INSTRUMENTS)}
    
    try:
        r = pricing.PricingInfo(accountID=ACCOUNT_ID, params=params)
        response = client.request(r)
        
        print(f"{'PAIR':<10} | {'BID':<10} | {'ASK':<10} | {'DIFF':<10} | {'CALC SPREAD (Pips)':<20}")
        print("-" * 75)

        for price in response['prices']:
            instrument = price['instrument']
            
            # OANDA returns prices as strings, convert to float
            bid = float(price['bids'][0]['price'])
            ask = float(price['asks'][0]['price'])
            
            # Raw difference
            diff = ask - bid
            
            # Calculate Pips
            multiplier = get_pip_multiplier(instrument)
            spread_pips = diff * multiplier
            
            print(f"{instrument:<10} | {bid:<10.5f} | {ask:<10.5f} | {diff:<10.5f} | {spread_pips:<20.2f}")

    except V20Error as e:
        print("Error fetching data: {}".format(e))

if __name__ == "__main__":
    print("\n--- OANDA SPREAD DEBUGGER ---\n")
    debug_spreads()
    print("\n-----------------------------\n")