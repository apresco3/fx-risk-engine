from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# =====================
# Config
# =====================
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

WEBHOOK_KEY = os.getenv("WEBHOOK_KEY", "")
MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

if not OANDA_TOKEN or not ACCOUNT_ID:
    raise RuntimeError("Missing OANDA credentials. Check .env file.")

BASE_URL = (
    "https://api-fxpractice.oanda.com/v3"
    if MODE == "PRACTICE"
    else "https://api-fxtrade.oanda.com/v3"
)

# =====================
# Helpers
# =====================
def log(msg: str):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")


def risk_check(units: int) -> bool:
    return abs(units) <= MAX_UNITS


def validate_payload(data: dict):
    side = str(data.get("side", "")).lower()
    pair = str(data.get("pair", "")).upper()
    units = data.get("units")

    if side not in {"buy", "sell"}:
        return None, "Invalid side"

    if "_" not in pair:
        return None, "Invalid pair format (use EUR_USD)"

    try:
        units = int(units)
    except Exception:
        return None, "Units must be integer"

    if units <= 0:
        return None, "Units must be positive"

    return {"side": side, "pair": pair, "units": units}, None

# =====================
# Routes
# =====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mode": MODE,
        "trading_enabled": TRADING_ENABLED
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    # --- URL key auth (TradingView-compatible) ---
    key = request.args.get("key", "")
    if key != WEBHOOK_KEY:
        log("AUTH BLOCKED | invalid webhook key")
        return {"status": "UNAUTHORIZED"}, 401

    if not TRADING_ENABLED:
        return {"status": "DISABLED"}, 200

    data = request.get_json(silent=True) or {}
    payload, err = validate_payload(data)
    if err:
        log(f"BAD REQUEST | {err} | {data}")
        return {"status": "BAD_REQUEST", "error": err}, 400

    side = payload["side"]
    pair = payload["pair"]
    units = payload["units"]

    if not risk_check(units):
        log(f"RISK BLOCKED | {pair} {side} {units}")
        return {"status": "RISK_BLOCKED"}, 403

    signed_units = -units if side == "sell" else units

    order = {
        "order": {
            "instrument": pair,
            "units": str(signed_units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT"
        }
    }

    headers = {
        "Authorization": f"Bearer {OANDA_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders",
            json=order,
            headers=headers,
            timeout=10
        )

        body = r.json()
        log(f"ORDER | {pair} {side} {signed_units} | HTTP {r.status_code}")

        return {
            "status": "OK" if r.ok else "OANDA_ERROR",
            "http_status": r.status_code,
            "response": body
        }, (200 if r.ok else 502)

    except requests.RequestException as e:
        log(f"NETWORK ERROR | {repr(e)}")
        return {"status": "NETWORK_ERROR"}, 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
