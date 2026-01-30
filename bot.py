from flask import Flask, request
import requests
import datetime
import os

app = Flask(__name__)

OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE")

if not OANDA_TOKEN or not ACCOUNT_ID:
    raise Exception("Missing OANDA credentials. Check .env file.")

BASE_URL = "https://api-fxpractice.oanda.com/v3" if MODE == "PRACTICE" else "https://api-fxtrade.oanda.com/v3"

MAX_UNITS = 1000
TRADING_ENABLED = True   # Kill switch

def log(msg):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.datetime.now()} | {msg}\n")

def risk_check(units):
    return abs(units) <= MAX_UNITS

@app.route("/webhook", methods=["POST"])
def webhook():
    if not TRADING_ENABLED:
        return {"status": "DISABLED"}

    data = request.json

    side = data["side"]
    pair = data["pair"]
    units = int(data["units"])

    if not risk_check(units):
        log("RISK BLOCKED")
        return {"status": "RISK_BLOCKED"}

    if side == "sell":
        units = -units

    order = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT"
        }
    }

    headers = {
        "Authorization": f"Bearer {OANDA_TOKEN}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders",
        json=order,
        headers=headers
    )

    log(f"{pair} {side} {units} | {r.status_code}")
    return {"status": r.status_code, "response": r.json()}

app.run(port=5000)

