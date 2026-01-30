import os
import requests
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

print("-" * 40)
print(f"TESTING OANDA CONNECTION")
print(f"Mode:       {MODE}")
print(f"Account ID: {ACCOUNT_ID}")
print(f"Token:      {TOKEN[:5]}...{TOKEN[-5:]} (Masked)")
print("-" * 40)

if MODE == "PRACTICE":
    BASE_URL = "https://api-fxpractice.oanda.com/v3"
else:
    BASE_URL = "https://api-fxtrade.oanda.com/v3"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 1. Test Summary Endpoint (Lightweight)
url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/summary"
print(f"\n[1] Pinging: {url} ...")

try:
    r = requests.get(url, headers=headers, timeout=10)
    
    if r.status_code == 200:
        data = r.json()
        nav = data['account']['NAV']
        print(f"✅ SUCCESS! Connection Established.")
        print(f"   Account NAV: {nav} {data['account']['currency']}")
        print(f"   Open Positions: {data['account']['openPositionCount']}")
    else:
        print(f"❌ FAILED. HTTP Status: {r.status_code}")
        print(f"   Error Message: {r.text}")
        
        if r.status_code == 401:
            print("\n👉 HINT: 401 means 'Unauthorized'. Your Token is wrong or does not belong to this Account ID.")
        elif r.status_code == 404:
            print("\n👉 HINT: 404 means 'Not Found'. Your Account ID is invalid or does not exist in this Mode (Practice/Live).")
            
except Exception as e:
    print(f"❌ NETWORK ERROR: {e}")

print("-" * 40)