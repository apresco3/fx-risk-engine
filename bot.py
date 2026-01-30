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

# Safety switches
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# Risk limits
MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))          # per-order cap
MAX_NET_UNITS = int(os.getenv("MAX_NET_UNITS", "2000"))  # per-instrument net cap

# Bracket defaults (pips)
SL_PIPS_DEFAULT = int(os.getenv("SL_PIPS", "20"))
TP_PIPS_DEFAULT = int(os.getenv("TP_PIPS", "40"))

# Spread filter (pips). 0 disables.
MAX_SPREAD_PIPS = int(os.getenv("MAX_SPREAD_PIPS", "2"))

# Duplicate protection
SEEN_TTL_SECONDS = int(os.getenv("SEEN_TTL_SECONDS", "60"))
SEEN_MAX_SIZE = int(os.getenv("SEEN_MAX_SIZE", "5000"))

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

def seen_before(alert_id: str) -> bool:
    now = time.time()

    # prune old entries
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


def pip_size_for(instrument: str) -> Decimal:
    return Decimal("0.01") if instrument.endswith("_JPY") else Decimal("0.0001")


def fmt_price_decimal(x: Decimal, instrument: str) -> str:
    # JPY pairs typically 3 decimals; most others 5 decimals is safe
    places = Decimal("0.001") if instrument.endswith("_JPY") else Decimal("0.00001")
    return str(x.quantize(places, rounding=ROUND_DOWN))


def risk_check_order_units(units: int) -> bool:
    return abs(units) <= MAX_UNITS


def sl_tp_distance(instrument: str, sl_pips: int, tp_pips: int):
    pip = pip_size_for(instrument)
    sl_dist = (pip * Decimal(sl_pips)) if sl_pips and sl_pips > 0 else None
    tp_dist = (pip * Decimal(tp_pips)) if tp_pips and tp_pips > 0 else None
    return sl_dist, tp_dist


def oanda_get_open_positions():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions"
    return requests.get(url, headers=HEADERS, timeout=10)


def get_net_units_for_instrument(instrument: str) -> int:
    r = oanda_get_open_positions()
    if not r.ok:
        raise RuntimeError(f"Failed to fetch open positions: HTTP {r.status_code} {r.text}")

    data = r.json()
    for pos in data.get("positions", []):
        if pos.get("instrument") == instrument:
            long_units = int(pos.get("long", {}).get("units", "0"))
            short_units = int(pos.get("short", {}).get("units", "0"))  # usually negative or 0
            return long_units + short_units
    return 0


def oanda_get_pricing(instrument: str):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing"
    params = {"instruments": instrument}
    r = requests.get(url, headers=HEADERS, params=params, timeout=10)
    return r


def get_spread_pips(instrument: str) -> float:
    """
    Fetch current pricing and compute spread in pips.
    """
    r = oanda_get_pricing(instrument)
    if not r.ok:
        raise RuntimeError(f"Failed to fetch pricing: HTTP {r.status_code} {r.text}")

    data = r.json()
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError("No pricing data returned")

    p = prices[0]
    bids = p.get("bids", [])
    asks = p.get("asks", [])
    if not bids or not asks:
        raise RuntimeError("Missing bid/ask in pricing data")

    bid = Decimal(bids[0]["price"])
    ask = Decimal(asks[0]["price"])
    spread = ask - bid
    pip = pip_size_for(instrument)

    spread_pips = float(spread / pip)
    return spread_pips


def place_market_order(
    *,
    alert_id: str,
    instrument: str,
    signed_units: int,
    sl_pips: int,
    tp_pips: int
):
    # Attach SL/TP via distance
    sl_dist, tp_dist = sl_tp_distance(instrument, sl_pips, tp_pips)

    order = {
        "order": {
            "instrument": instrument,
            "units": str(signed_units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }

    if sl_dist is not None:
        order["order"]["stopLossOnFill"] = {"distance": fmt_price_decimal(sl_dist, instrument)}
    if tp_dist is not None:
        order["order"]["takeProfitOnFill"] = {"distance": fmt_price_decimal(tp_dist, instrument)}

    r = requests.post(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders",
        json=order,
        headers=HEADERS,
        timeout=10
    )
    return r


def close_position(instrument: str):
    """
    Close any open long/short for instrument.
    OANDA: PUT /accounts/{id}/positions/{instrument}/close
    """
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/positions/{instrument}/close"
    payload = {"longUnits": "ALL", "shortUnits": "ALL"}
    r = requests.put(url, json=payload, headers=HEADERS, timeout=10)
    return r


def validate_payload(data: dict):
    """
    Supported payloads:

    1) Open:
      {
        "alert_id": "...",
        "action": "open",        # optional, defaults to "open"
        "side": "buy"|"sell",
        "pair": "EUR_USD",
        "units": 10,
        "sl_pips": 20,           # optional
        "tp_pips": 40            # optional
      }

    2) Close:
      {
        "alert_id": "...",
        "action": "close",
        "pair": "EUR_USD"
      }

    3) Reverse:
      {
        "alert_id": "...",
        "action": "reverse",
        "side": "buy"|"sell",
        "pair": "EUR_USD",
        "units": 10,
        "sl_pips": 20,           # optional
        "tp_pips": 40            # optional
      }
    """
    alert_id = str(data.get("alert_id", "")).strip()
    action = str(data.get("action", "open")).lower().strip()
    pair = str(data.get("pair", "")).upper().strip()

    if not alert_id:
        return None, "Missing alert_id"

    if "_" not in pair:
        return None, "Invalid pair format (use EUR_USD)"

    if action not in {"open", "close", "reverse"}:
        return None, "Invalid action (open|close|reverse)"

    if action == "close":
        return {"alert_id": alert_id, "action": action, "pair": pair}, None

    side = str(data.get("side", "")).lower().strip()
    if side not in {"buy", "sell"}:
        return None, "Invalid side (must be 'buy' or 'sell')"

    units = data.get("units")
    try:
        units = int(units)
    except Exception:
        return None, "Units must be integer"

    if units <= 0:
        return None, "Units must be positive"

    sl_pips = data.get("sl_pips", SL_PIPS_DEFAULT)
    tp_pips = data.get("tp_pips", TP_PIPS_DEFAULT)
    try:
        sl_pips = int(sl_pips)
        tp_pips = int(tp_pips)
    except Exception:
        return None, "sl_pips / tp_pips must be integers"

    if sl_pips < 0 or tp_pips < 0:
        return None, "sl_pips / tp_pips must be >= 0"

    return {
        "alert_id": alert_id,
        "action": action,
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
        "max_spread_pips": MAX_SPREAD_PIPS,
        "seen_ttl_seconds": SEEN_TTL_SECONDS,
    })


@app.route("/positions", methods=["GET"])
def positions():
    """
    Debug endpoint: shows open positions from OANDA.
    Protect with the same key so it’s not public.
    """
    key = request.args.get("key", "")
    if key != WEBHOOK_KEY:
        return {"status": "UNAUTHORIZED"}, 401

    try:
        r = oanda_get_open_positions()
        if not r.ok:
            return {"status": "OANDA_ERROR", "http_status": r.status_code, "response": r.text}, 502
        return jsonify({"status": "OK", "response": r.json()})
    except requests.RequestException as e:
        return {"status": "NETWORK_ERROR", "error": repr(e)}, 502


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

    action = payload["action"]
    pair = payload["pair"]

    # ===== CLOSE =====
    if action == "close":
        try:
            r = close_position(pair)
            body = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {"raw": r.text}
            log(f"CLOSE | alert_id={alert_id} | {pair} | HTTP {r.status_code}")
            return {
                "status": "OK" if r.ok else "OANDA_ERROR",
                "http_status": r.status_code,
                "response": body
            }, (200 if r.ok else 502)
        except requests.RequestException as e:
            log(f"NETWORK ERROR | CLOSE | alert_id={alert_id} | {pair} | {repr(e)}")
            return {"status": "NETWORK_ERROR"}, 502

    # From here: OPEN or REVERSE
    side = payload["side"]
    units = payload["units"]
    sl_pips = payload["sl_pips"]
    tp_pips = payload["tp_pips"]
    signed_units = -units if side == "sell" else units

    # Per-order units cap
    if not risk_check_order_units(units):
        log(f"RISK BLOCKED (MAX_UNITS) | alert_id={alert_id} | {pair} {side} {units}")
        return {"status": "RISK_BLOCKED_MAX_UNITS"}, 403

    # Spread filter (optional)
    if MAX_SPREAD_PIPS > 0:
        try:
            spread_pips = get_spread_pips(pair)
            if spread_pips > MAX_SPREAD_PIPS:
                log(f"RISK BLOCKED (SPREAD) | alert_id={alert_id} | {pair} spread_pips={spread_pips:.2f} cap={MAX_SPREAD_PIPS}")
                return {
                    "status": "RISK_BLOCKED_SPREAD",
                    "spread_pips": spread_pips,
                    "cap": MAX_SPREAD_PIPS
                }, 403
        except Exception as e:
            # Fail safe: if you can't price-check, block
            log(f"RISK BLOCKED (PRICE_FETCH_FAIL) | alert_id={alert_id} | {pair} | {repr(e)}")
            return {"status": "RISK_BLOCKED_PRICE_FETCH_FAIL"}, 503

    # Reverse means: close first, then open opposite
    if action == "reverse":
        try:
            r_close = close_position(pair)
            log(f"REVERSE-CLOSE | alert_id={alert_id} | {pair} | HTTP {r_close.status_code}")
            if not r_close.ok:
                body = r_close.json() if r_close.headers.get("Content-Type", "").startswith("application/json") else {"raw": r_close.text}
                return {"status": "OANDA_ERROR", "step": "close", "http_status": r_close.status_code, "response": body}, 502
        except requests.RequestException as e:
            log(f"NETWORK ERROR | REVERSE-CLOSE | alert_id={alert_id} | {pair} | {repr(e)}")
            return {"status": "NETWORK_ERROR", "step": "close"}, 502

        # small delay to let position state settle (practice env can be slightly laggy)
        time.sleep(0.2)

    # Net exposure gate
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

    # Place order with SL/TP
    try:
        r = place_market_order(
            alert_id=alert_id,
            instrument=pair,
            signed_units=signed_units,
            sl_pips=sl_pips,
            tp_pips=tp_pips
        )

        body = r.json()
        log(
            f"{action.upper()} | alert_id={alert_id} | {pair} {side} {signed_units} | "
            f"net={current_net}->{projected_net} | SLpips={sl_pips} TPpips={tp_pips} | HTTP {r.status_code}"
        )

        return {
            "status": "OK" if r.ok else "OANDA_ERROR",
            "action": action,
            "http_status": r.status_code,
            "current_net": current_net,
            "projected_net": projected_net,
            "spread_filter_pips": MAX_SPREAD_PIPS,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "response": body
        }, (200 if r.ok else 502)

    except requests.RequestException as e:
        log(f"NETWORK ERROR | {action.upper()} | alert_id={alert_id} | {pair} {side} {signed_units} | {repr(e)}")
        return {"status": "NETWORK_ERROR"}, 502


if __name__ == "__main__":
    # debug=False to avoid Flask auto-reloader weirdness during trading
    app.run(host="0.0.0.0", port=8080, debug=False)
