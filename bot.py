from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os
import time
from collections import OrderedDict
from decimal import Decimal, ROUND_DOWN

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# =====================
# Config
# =====================
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

WEBHOOK_KEY = os.getenv("WEBHOOK_KEY", "")
MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))                 # per-order cap
MAX_NET_UNITS = int(os.getenv("MAX_NET_UNITS", "2000"))         # per-instrument net position cap
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# Basic default bracket (you can override per alert if you want)
SL_PIPS_DEFAULT = int(os.getenv("SL_PIPS", "20"))
TP_PIPS_DEFAULT = int(os.getenv("TP_PIPS", "40"))

if not OANDA_TOKEN or not ACCOUNT_ID:
    raise RuntimeError("Missing OANDA credentials. Check .env file.")

BASE_URL = (
    "https://api-fxpractice.oanda.com/v3"
    if MODE == "PRACTICE"
    else "https://api-fxtrade.oanda.com/v3"
)

HEADERS = {
    "Authorization": f"Bearer {OANDA_TOKEN}",
    "Content-Type": "application/json",
}

# =====================
# Duplicate protection
# =====================
SEEN = OrderedDict()
SEEN_TTL_SECONDS = 60   # ignore duplicate alert_id within 60 seconds
SEEN_MAX_SIZE = 5000

def seen_before(alert_id: str) -> bool:
    now = time.time()

    # Prune old entries
    while SEEN:
        oldest_id, ts = next(iter(SEEN.items()))
        if now - ts > SEEN_TTL_SECONDS:
            SEEN.popitem(last=False)
        else:
            break

    if alert_id in SEEN:
        return True

    SEEN[alert_id] = now
    if len(SEEN) > SEEN_MAX_SIZE:
        SEEN.popitem(last=False)

    return False

# =====================
# Helpers
# =====================
def log(msg: str):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")

def risk_check(units: int) -> bool:
    # per-order cap
    return abs(units) <= MAX_UNITS

def pip_size_for(instrument: str) -> Decimal:
    # Simple FX convention: JPY pairs pip is 0.01, most others 0.0001
    return Decimal("0.01") if instrument.endswith("_JPY") else Decimal("0.0001")

def fmt_price_decimal(x: Decimal, instrument: str) -> str:
    # Round down to a sensible number of decimals for the instrument
    # (JPY pairs often 3 decimals; others 5 is common—distance can be fine with 5)
    places = Decimal("0.001") if instrument.endswith("_JPY") else Decimal("0.00001")
    return str(x.quantize(places, rounding=ROUND_DOWN))

def sl_tp_distance(instrument: str, sl_pips: int, tp_pips: int):
    pip = pip_size_for(instrument)
    sl_dist = (pip * Decimal(sl_pips)) if sl_pips and sl_pips > 0 else None
    tp_dist = (pip * Decimal(tp_pips)) if tp_pips and tp_pips > 0 else None
    return sl_dist, tp_dist

def oanda_get_open_positions():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions"
    r = requests.get(url, headers=HEADERS, timeout=10)
    return r

def get_net_units_for_instrument(instrument: str) -> int:
    """
    Returns current net units for instrument (long positive, short negative).
    If instrument has no open position, returns 0.
    """
    r = oanda_get_open_positions()
    if not r.ok:
        # If we can't verify exposure, fail safe (block trading).
        raise RuntimeError(f"Failed to fetch open positions: HTTP {r.status_code} {r.text}")

    data = r.json()
    for pos in data.get("positions", []):
        if pos.get("instrument") == instrument:
            long_units = int(pos.get("long", {}).get("units", "0"))
            short_units = int(pos.get("short", {}).get("units", "0"))  # typically negative or 0
            return long_units + short_units

    return 0

def validate_payload(data: dict):
    alert_id = str(data.get("alert_id", "")).strip()
    side = str(data.get("side", "")).lower().strip()
    pair = str(data.get("pair", "")).upper().strip()
    units = data.get("units")

    # Optional per-alert overrides
    sl_pips = data.get("sl_pips", SL_PIPS_DEFAULT)
    tp_pips = data.get("tp_pips", TP_PIPS_DEFAULT)

    if not alert_id:
        return None, "Missing alert_id"

    if side not in {"buy", "sell"}:
        return None, "Invalid side (must be 'buy' or 'sell')"

    if "_" not in pair:
        return None, "Invalid pair format (use EUR_USD)"

    try:
        units = int(units)
    except Exception:
        return None, "Units must be integer"

    if units <= 0:
        return None, "Units must be positive"

    try:
        sl_pips = int(sl_pips)
        tp_pips = int(tp_pips)
    except Exception:
        return None, "sl_pips / tp_pips must be integers"

    if sl_pips < 0 or tp_pips < 0:
        return None, "sl_pips / tp_pips must be >= 0"

    return {
        "alert_id": alert_id,
        "side": side,
        "pair": pair,
        "units": units,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
    }, None

# =====================
# Routes
# =====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mode": MODE,
        "trading_enabled": TRADING_ENABLED,
        "max_units": MAX_UNITS,
        "max_net_units": MAX_NET_UNITS,
        "sl_pips_default": SL_PIPS_DEFAULT,
        "tp_pips_default": TP_PIPS_DEFAULT,
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

    alert_id = payload["alert_id"]
    if seen_before(alert_id):
        log(f"DUPLICATE IGNORED | alert_id={alert_id}")
        return {"status": "DUPLICATE_IGNORED"}, 200

    side = payload["side"]
    pair = payload["pair"]
    units = payload["units"]
    sl_pips = payload["sl_pips"]
    tp_pips = payload["tp_pips"]

    if not risk_check(units):
        log(f"RISK BLOCKED (MAX_UNITS) | alert_id={alert_id} | {pair} {side} {units}")
        return {"status": "RISK_BLOCKED_MAX_UNITS"}, 403

    signed_units = -units if side == "sell" else units

    # --- Net exposure gate (MAX_NET_UNITS per instrument) ---
    try:
        current_net = get_net_units_for_instrument(pair)
    except Exception as e:
        log(f"RISK BLOCKED (POS_FETCH_FAIL) | alert_id={alert_id} | {pair} | {repr(e)}")
        return {"status": "RISK_BLOCKED_POS_FETCH_FAIL"}, 503

    projected_net = current_net + signed_units
    if abs(projected_net) > MAX_NET_UNITS:
        log(
            f"RISK BLOCKED (MAX_NET_UNITS) | alert_id={alert_id} | {pair} "
            f"current_net={current_net} new={signed_units} projected={projected_net} cap={MAX_NET_UNITS}"
        )
        return {
            "status": "RISK_BLOCKED_MAX_NET_UNITS",
            "current_net": current_net,
            "projected_net": projected_net,
            "cap": MAX_NET_UNITS
        }, 403

    # --- Bracket orders (SL/TP) via distance ---
    sl_dist, tp_dist = sl_tp_distance(pair, sl_pips, tp_pips)

    order = {
        "order": {
            "instrument": pair,
            "units": str(signed_units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }

    if sl_dist is not None:
        order["order"]["stopLossOnFill"] = {"distance": fmt_price_decimal(sl_dist, pair)}
    if tp_dist is not None:
        order["order"]["takeProfitOnFill"] = {"distance": fmt_price_decimal(tp_dist, pair)}

    try:
        r = requests.post(
            f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders",
            json=order,
            headers=HEADERS,
            timeout=10
        )

        body = r.json()
        log(
            f"ORDER | alert_id={alert_id} | {pair} {side} {signed_units} | "
            f"net={current_net}->{projected_net} | SLpips={sl_pips} TPpips={tp_pips} | HTTP {r.status_code}"
        )

        return {
            "status": "OK" if r.ok else "OANDA_ERROR",
            "http_status": r.status_code,
            "current_net": current_net,
            "projected_net": projected_net,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "response": body
        }, (200 if r.ok else 502)

    except requests.RequestException as e:
        log(f"NETWORK ERROR | alert_id={alert_id} | {pair} {side} {signed_units} | {repr(e)}")
        return {"status": "NETWORK_ERROR"}, 502

if __name__ == "__main__":
    # debug=False to avoid Flask auto-reloader weirdness during trading
    app.run(host="0.0.0.0", port=8080, debug=False)
