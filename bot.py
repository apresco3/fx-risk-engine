from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os
import hmac
import hashlib

load_dotenv()

app = Flask(__name__)

OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

if not OANDA_TOKEN or not ACCOUNT_ID:
    raise RuntimeError("Missing OANDA credentials. Ensure OANDA_TOKEN and OANDA_ACCOUNT_ID are set.")

BASE_URL = "https://api-fxpractice.oanda.com/v3" if MODE == "PRACTICE" else "https://api-fxtrade.oanda.com/v3"


def log(msg: str) -> None:
    with open("trades.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")


def risk_check(units: int) -> bool:
    return abs(units) <= MAX_UNITS


def verify_signature(raw_body: bytes, provided_sig: str) -> bool:
    """
    Simple HMAC auth:
      TradingView (or your middleware) sends header: X-Signature = hex(hmac_sha256(secret, raw_body))
    """
    if not WEBHOOK_SECRET:
        return False
    expected = hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided_sig or "")


def parse_payload(data: dict):
    # Required fields
    side = str(data.get("side", "")).lower().strip()
    pair = str(data.get("pair", "")).upper().strip()
    units_raw = data.get("units", None)

    if side not in {"buy", "sell"}:
        return None, "Invalid 'side'. Must be 'buy' or 'sell'."

    if not pair or "_" not in pair:
        # OANDA instruments look like EUR_USD, USD_JPY, etc.
        return None, "Invalid 'pair'. Expected format like 'EUR_USD'."

    try:
        units = int(units_raw)
    except Exception:
        return None, "Invalid 'units'. Must be an integer."

    if units <= 0:
        return None, "'units' must be > 0."

    return {"side": side, "pair": pair, "units": units}, None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "mode": MODE, "trading_enabled": TRADING_ENABLED})


@app.route("/webhook", methods=["POST"])
def webhook():
    if not TRADING_ENABLED:
        return jsonify({"status": "DISABLED"}), 200

    # 1) Authenticate caller (highly recommended if this is exposed to the internet)
    raw = request.get_data()  # raw bytes
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(raw, sig):
        log("AUTH BLOCKED | missing/invalid signature")
        return jsonify({"status": "UNAUTHORIZED"}), 401

    # 2) Parse JSON safely
    data = request.get_json(silent=True) or {}
    payload, err = parse_payload(data)
    if err:
        log(f"BAD REQUEST | {err} | data={data}")
        return jsonify({"status": "BAD_REQUEST", "error": err}), 400

    side = payload["side"]
    pair = payload["pair"]
    units = payload["units"]

    # 3) Risk gate
    if not risk_check(units):
        log(f"RISK BLOCKED | pair={pair} side={side} units={units} max={MAX_UNITS}")
        return jsonify({"status": "RISK_BLOCKED"}), 403

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
            timeout=10,
        )
        ok = 200 <= r.status_code < 300
        body = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {"raw": r.text}

        log(f"ORDER | pair={pair} side={side} units={signed_units} http={r.status_code} ok={ok}")

        return jsonify({
            "status": "OK" if ok else "OANDA_ERROR",
            "http_status": r.status_code,
            "oanda": body
        }), (200 if ok else 502)

    except requests.RequestException as e:
        log(f"NETWORK ERROR | {pair} {side} {signed_units} | {repr(e)}")
        return jsonify({"status": "NETWORK_ERROR"}), 502


if __name__ == "__main__":
    # For local dev. In production use gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)
