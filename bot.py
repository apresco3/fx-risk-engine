from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os
import time
import sqlite3
from decimal import Decimal, ROUND_DOWN, getcontext

getcontext().prec = 28

# Load env
load_dotenv()

app = Flask(__name__)

# =====================
# Config
# =====================
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

WEBHOOK_KEY = os.getenv("WEBHOOK_KEY", "")
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# Risk caps
MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))              # per-order hard cap
MAX_NET_UNITS = int(os.getenv("MAX_NET_UNITS", "2000"))      # per-instrument net cap

# Bracket defaults (pips)
SL_PIPS_DEFAULT = int(os.getenv("SL_PIPS", "20"))
TP_PIPS_DEFAULT = int(os.getenv("TP_PIPS", "40"))

# Spread filter (float lets you use 1.5 pips if desired)
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "2"))   # 0 disables

# Equity sizing
# Risk per trade as a fraction of NAV (e.g. 0.0025 = 0.25%)
RISK_PCT = Decimal(os.getenv("RISK_PCT", "0.0025"))
# If payload includes "units", we can either respect it or override with risk sizing.
# Set true to ALWAYS compute units from risk sizing.
FORCE_RISK_SIZING = os.getenv("FORCE_RISK_SIZING", "true").lower() == "true"
# If account currency != instrument quote currency, sizing needs conversion.
# Default: block to be safe (no silent wrong sizing).
ALLOW_NONQUOTE_HOME_SIZING = os.getenv("ALLOW_NONQUOTE_HOME_SIZING", "false").lower() == "true"

# Persistent state
DB_PATH = os.getenv("DB_PATH", "bot.db")
ALERT_TTL_SECONDS = int(os.getenv("ALERT_TTL_SECONDS", "60"))  # dedupe window

# Server
PORT = int(os.getenv("PORT", "8080"))

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
# DB setup
# =====================
def db_conn():
    # check_same_thread=False is fine here because we do simple short transactions
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            first_seen_ts INTEGER NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            alert_id TEXT NOT NULL,
            action TEXT NOT NULL,
            instrument TEXT NOT NULL,
            side TEXT,
            units INTEGER,
            sl_pips INTEGER,
            tp_pips INTEGER,
            current_net INTEGER,
            projected_net INTEGER,
            spread_pips REAL,
            oanda_http INTEGER,
            status TEXT NOT NULL,
            oanda_response TEXT
        );
    """)

    conn.commit()
    conn.close()

db_init()

# =====================
# Logging
# =====================
def log(msg: str):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")

def db_record_execution(**kwargs):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO executions (
            ts, alert_id, action, instrument, side, units, sl_pips, tp_pips,
            current_net, projected_net, spread_pips, oanda_http, status, oanda_response
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(time.time()),
        kwargs.get("alert_id"),
        kwargs.get("action"),
        kwargs.get("instrument"),
        kwargs.get("side"),
        kwargs.get("units"),
        kwargs.get("sl_pips"),
        kwargs.get("tp_pips"),
        kwargs.get("current_net"),
        kwargs.get("projected_net"),
        kwargs.get("spread_pips"),
        kwargs.get("oanda_http"),
        kwargs.get("status"),
        kwargs.get("oanda_response"),
    ))
    conn.commit()
    conn.close()

# =====================
# Dedupe (SQLite-backed)
# =====================
def seen_before(alert_id: str) -> bool:
    now = int(time.time())
    cutoff = now - ALERT_TTL_SECONDS

    conn = db_conn()
    cur = conn.cursor()

    # prune old rows occasionally (cheap + simple)
    cur.execute("DELETE FROM alerts WHERE first_seen_ts < ?", (cutoff,))

    # check existence
    cur.execute("SELECT first_seen_ts FROM alerts WHERE alert_id = ?", (alert_id,))
    row = cur.fetchone()
    if row is not None:
        conn.commit()
        conn.close()
        return True

    # insert new
    cur.execute("INSERT INTO alerts (alert_id, first_seen_ts) VALUES (?, ?)", (alert_id, now))
    conn.commit()
    conn.close()
    return False

# =====================
# OANDA helpers
# =====================
def pip_size_for(instrument: str) -> Decimal:
    return Decimal("0.01") if instrument.endswith("_JPY") else Decimal("0.0001")

def fmt_price_decimal(x: Decimal, instrument: str) -> str:
    places = Decimal("0.001") if instrument.endswith("_JPY") else Decimal("0.00001")
    return str(x.quantize(places, rounding=ROUND_DOWN))

def oanda_get_open_positions():
    return requests.get(f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions", headers=HEADERS, timeout=10)

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
    return requests.get(url, headers=HEADERS, params=params, timeout=10)

def get_spread_pips(instrument: str) -> float:
    r = oanda_get_pricing(instrument)
    if not r.ok:
        raise RuntimeError(f"Failed to fetch pricing: HTTP {r.status_code} {r.text}")

    data = r.json()
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError("No pricing data returned")

    p = prices[0]
    bid = Decimal(p["bids"][0]["price"])
    ask = Decimal(p["asks"][0]["price"])
    spread = ask - bid
    pip = pip_size_for(instrument)
    return float(spread / pip)

def oanda_get_account_summary():
    return requests.get(f"{BASE_URL}/accounts/{ACCOUNT_ID}/summary", headers=HEADERS, timeout=10)

def get_account_nav_and_currency():
    r = oanda_get_account_summary()
    if not r.ok:
        raise RuntimeError(f"Failed to fetch account summary: HTTP {r.status_code} {r.text}")
    acct = r.json().get("account", {})
    nav = Decimal(acct.get("NAV", acct.get("balance", "0")))
    currency = str(acct.get("currency", "USD")).upper()
    return nav, currency

def close_position(instrument: str):
    """
    Close out the open Position for an instrument.
    IMPORTANT: Only specify longUnits/shortUnits if that side exists.
    """
    # Use openPositions so we can see long vs short amounts.
    r = oanda_get_open_positions()
    if not r.ok:
        raise RuntimeError(f"Failed to fetch open positions: HTTP {r.status_code} {r.text}")

    long_units = 0
    short_units = 0
    for pos in r.json().get("positions", []):
        if pos.get("instrument") == instrument:
            long_units = int(pos.get("long", {}).get("units", "0"))
            short_units = int(pos.get("short", {}).get("units", "0"))
            break

    payload = {}
    if long_units > 0:
        payload["longUnits"] = "ALL"
    if short_units < 0:  # OANDA short units are typically negative
        payload["shortUnits"] = "ALL"

    if not payload:
        # Already flat, nothing to close
        class Dummy:
            ok = True
            status_code = 200
            text = '{"skipped": true, "reason": "already_flat"}'
            headers = {"Content-Type": "application/json"}
            def json(self): return {"skipped": True, "reason": "already_flat"}
        return Dummy()

    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/positions/{instrument}/close"
    return requests.put(url, json=payload, headers=HEADERS, timeout=10)

def close_all_positions():
    """
    Closes ALL open positions across ALL instruments in the account.
    """
    r = oanda_get_open_positions()
    if not r.ok:
        raise RuntimeError(f"Failed to fetch open positions: HTTP {r.status_code} {r.text}")

    instruments = [p.get("instrument") for p in r.json().get("positions", []) if p.get("instrument")]
    results = []
    for inst in instruments:
        rc = close_position(inst)
        results.append({
            "instrument": inst,
            "ok": bool(rc.ok),
            "http_status": rc.status_code,
            "response": (rc.json() if rc.headers.get("Content-Type", "").startswith("application/json") else {"raw": rc.text})
        })
    return results

def place_market_order(*, instrument: str, signed_units: int, sl_pips: int, tp_pips: int):
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

    return requests.post(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders",
        json=order,
        headers=HEADERS,
        timeout=10
    )

# =====================
# Risk sizing
# =====================
def sl_tp_distance(instrument: str, sl_pips: int, tp_pips: int):
    pip = pip_size_for(instrument)
    sl_dist = (pip * Decimal(sl_pips)) if sl_pips and sl_pips > 0 else None
    tp_dist = (pip * Decimal(tp_pips)) if tp_pips and tp_pips > 0 else None
    return sl_dist, tp_dist

def compute_units_from_risk(
    *,
    instrument: str,
    sl_pips: int,
    nav: Decimal,
    account_ccy: str
) -> int:
    """
    Units from risk budget:
      risk_budget_home = nav * RISK_PCT
      loss_per_unit_home ≈ sl_dist_in_quote * quote_to_home
      units = risk_budget_home / loss_per_unit_home

    SAFETY:
    - If quote currency != account currency and we can't safely convert, we block unless allowed.
    """
    base, quote = instrument.split("_")
    pip = pip_size_for(instrument)
    sl_dist_quote = pip * Decimal(sl_pips)  # SL distance in quote currency per 1 unit (1 base unit)

    if sl_pips <= 0:
        raise RuntimeError("sl_pips must be > 0 for risk sizing")

    risk_budget_home = nav * RISK_PCT

    # Case 1: account currency == quote currency (common for USD accounts with EUR_USD, GBP_USD, etc.)
    if quote == account_ccy:
        loss_per_unit_home = sl_dist_quote

    else:
        # We need quote->home conversion.
        if not ALLOW_NONQUOTE_HOME_SIZING:
            raise RuntimeError(
                f"Cannot risk-size safely: quote={quote} != account_ccy={account_ccy}. "
                f"Set ALLOW_NONQUOTE_HOME_SIZING=true to allow approximate conversion."
            )

        quote_home = f"{quote}_{account_ccy}"
        home_quote = f"{account_ccy}_{quote}"

        conv = None
        # Try quote_home directly
        try:
            r = oanda_get_pricing(quote_home)
            if r.ok and r.json().get("prices"):
                p = r.json()["prices"][0]
                bid = Decimal(p["bids"][0]["price"])
                ask = Decimal(p["asks"][0]["price"])
                conv = (bid + ask) / Decimal(2)  # 1 quote = conv home
        except Exception:
            conv = None

        if conv is None:
            # Try inverse via home_quote
            r2 = oanda_get_pricing(home_quote)
            if not r2.ok or not r2.json().get("prices"):
                raise RuntimeError(f"Failed to obtain conversion rate {quote}<->{account_ccy}")
            p2 = r2.json()["prices"][0]
            bid2 = Decimal(p2["bids"][0]["price"])
            ask2 = Decimal(p2["asks"][0]["price"])
            mid2 = (bid2 + ask2) / Decimal(2)  # 1 home = mid2 quote
            conv = Decimal(1) / mid2           # 1 quote = conv home

        loss_per_unit_home = sl_dist_quote * conv

    if loss_per_unit_home <= 0:
        raise RuntimeError("Computed invalid loss_per_unit_home")

    units = int((risk_budget_home / loss_per_unit_home).to_integral_value(rounding=ROUND_DOWN))

    # hard caps
    units = max(1, units)
    units = min(units, MAX_UNITS)
    return units

# =====================
# Validation
# =====================
def validate_payload(data: dict):
    """
    Supported payloads:

    OPEN:
      {"alert_id":"...", "action":"open", "side":"buy|sell", "pair":"EUR_USD",
       "sl_pips":20, "tp_pips":40, "units":10 (optional)}

    CLOSE (single instrument):
      {"alert_id":"...", "action":"close", "pair":"EUR_USD"}

    CLOSE_ALL:
      {"alert_id":"...", "action":"close_all"}

    REVERSE:
      {"alert_id":"...", "action":"reverse", "side":"buy|sell", "pair":"EUR_USD",
       "sl_pips":20, "tp_pips":40, "units":10 (optional)}
    """
    alert_id = str(data.get("alert_id", "")).strip()
    action = str(data.get("action", "open")).lower().strip()
    pair = str(data.get("pair", "")).upper().strip()

    if not alert_id:
        return None, "Missing alert_id"
    if action not in {"open", "close", "close_all", "reverse"}:
        return None, "Invalid action (open|close|close_all|reverse)"

    if action == "close_all":
        return {"alert_id": alert_id, "action": action}, None

    if action == "close":
        if "_" not in pair:
            return None, "Invalid pair format (use EUR_USD)"
        return {"alert_id": alert_id, "action": action, "pair": pair}, None

    # open / reverse
    if "_" not in pair:
        return None, "Invalid pair format (use EUR_USD)"

    side = str(data.get("side", "")).lower().strip()
    if side not in {"buy", "sell"}:
        return None, "Invalid side (buy|sell)"

    sl_pips = data.get("sl_pips", SL_PIPS_DEFAULT)
    tp_pips = data.get("tp_pips", TP_PIPS_DEFAULT)
    try:
        sl_pips = int(sl_pips)
        tp_pips = int(tp_pips)
    except Exception:
        return None, "sl_pips/tp_pips must be integers"
    if sl_pips <= 0:
        return None, "sl_pips must be > 0 (required for risk sizing)"
    if tp_pips < 0:
        return None, "tp_pips must be >= 0"

    # units optional (risk sizing will override by default)
    units = data.get("units", None)
    if units is not None:
        try:
            units = int(units)
        except Exception:
            return None, "units must be integer"
        if units <= 0:
            return None, "units must be > 0"

    return {
        "alert_id": alert_id,
        "action": action,
        "side": side,
        "pair": pair,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "units": units
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
        "risk_pct": str(RISK_PCT),
        "force_risk_sizing": FORCE_RISK_SIZING,
        "db_path": DB_PATH,
        "alert_ttl_seconds": ALERT_TTL_SECONDS
    })

@app.route("/positions", methods=["GET"])
def positions():
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

@app.route("/executions", methods=["GET"])
def executions():
    key = request.args.get("key", "")
    if key != WEBHOOK_KEY:
        return {"status": "UNAUTHORIZED"}, 401

    limit = int(request.args.get("limit", "50"))
    limit = max(1, min(limit, 500))

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM executions ORDER BY ts DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"status": "OK", "executions": rows})

@app.route("/webhook", methods=["POST"])
def webhook():
    # auth
    key = request.args.get("key", "")
    if key != WEBHOOK_KEY:
        log("AUTH BLOCKED | invalid webhook key")
        return {"status": "UNAUTHORIZED"}, 401

    # optional: log raw body for debugging TradingView formatting
    raw = request.get_data(as_text=True)
    if raw:
        log(f"RAW BODY | {raw}")

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
        db_record_execution(
            alert_id=alert_id, action=payload["action"], instrument=payload.get("pair", "N/A"),
            side=payload.get("side"), units=payload.get("units"),
            sl_pips=payload.get("sl_pips"), tp_pips=payload.get("tp_pips"),
            status="DUPLICATE_IGNORED", oanda_http=None, oanda_response=None
        )
        return {"status": "DUPLICATE_IGNORED"}, 200

    action = payload["action"]

    # CLOSE ALL
    if action == "close_all":
        try:
            results = close_all_positions()
            log(f"CLOSE_ALL | alert_id={alert_id} | closed={len(results)}")
            for item in results:
                db_record_execution(
                    alert_id=alert_id,
                    action="close",
                    instrument=item["instrument"],
                    side=None, units=None, sl_pips=None, tp_pips=None,
                    current_net=None, projected_net=None, spread_pips=None,
                    oanda_http=item["http_status"],
                    status=("OK" if item["ok"] else "OANDA_ERROR"),
                    oanda_response=str(item["response"])
                )
            return {"status": "OK", "results": results}, 200
        except Exception as e:
            log(f"ERROR | CLOSE_ALL | alert_id={alert_id} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="close_all", instrument="ALL",
                status="ERROR", oanda_response=repr(e), oanda_http=None
            )
            return {"status": "ERROR", "error": str(e)}, 502

    pair = payload["pair"]

    # CLOSE (single)
    if action == "close":
        try:
            r = close_position(pair)
            body = r.text
            log(f"CLOSE | alert_id={alert_id} | {pair} | HTTP {r.status_code}")
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair,
                side=None, units=None, sl_pips=None, tp_pips=None,
                current_net=None, projected_net=None, spread_pips=None,
                oanda_http=r.status_code, status=("OK" if r.ok else "OANDA_ERROR"),
                oanda_response=body
            )
            return {"status": "OK" if r.ok else "OANDA_ERROR",
                    "http_status": r.status_code,
                    "response": (r.json() if r.headers.get("Content-Type","").startswith("application/json") else {"raw": r.text})
                    }, (200 if r.ok else 502)
        except requests.RequestException as e:
            log(f"NETWORK ERROR | CLOSE | alert_id={alert_id} | {pair} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair, status="NETWORK_ERROR",
                oanda_response=repr(e), side=None, units=None, sl_pips=None, tp_pips=None,
                current_net=None, projected_net=None, spread_pips=None, oanda_http=None
            )
            return {"status": "NETWORK_ERROR"}, 502

    # OPEN / REVERSE
    side = payload["side"]
    sl_pips = payload["sl_pips"]
    tp_pips = payload["tp_pips"]

    # Spread filter
    spread_pips_val = None
    if MAX_SPREAD_PIPS > 0:
        try:
            spread_pips_val = get_spread_pips(pair)
            if spread_pips_val > MAX_SPREAD_PIPS:
                log(f"RISK BLOCKED (SPREAD) | alert_id={alert_id} | {pair} spread={spread_pips_val:.2f} cap={MAX_SPREAD_PIPS}")
                db_record_execution(
                    alert_id=alert_id, action=action, instrument=pair, side=side,
                    units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                    spread_pips=spread_pips_val, status="RISK_BLOCKED_SPREAD",
                    oanda_http=None, oanda_response=None
                )
                return {"status": "RISK_BLOCKED_SPREAD", "spread_pips": spread_pips_val, "cap": MAX_SPREAD_PIPS}, 403
        except Exception as e:
            log(f"RISK BLOCKED (PRICE_FETCH_FAIL) | alert_id={alert_id} | {pair} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action=action, instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                status="RISK_BLOCKED_PRICE_FETCH_FAIL", oanda_http=None, oanda_response=repr(e)
            )
            return {"status": "RISK_BLOCKED_PRICE_FETCH_FAIL"}, 503

    # Reverse: close first
    if action == "reverse":
        try:
            r_close = close_position(pair)
            log(f"REVERSE-CLOSE | alert_id={alert_id} | {pair} | HTTP {r_close.status_code}")
            if not r_close.ok:
                body = r_close.text
                db_record_execution(
                    alert_id=alert_id, action="reverse_close", instrument=pair,
                    status="OANDA_ERROR", oanda_http=r_close.status_code, oanda_response=body,
                    side=side, units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                    current_net=None, projected_net=None, spread_pips=spread_pips_val
                )
                return {"status": "OANDA_ERROR", "step": "close",
                        "http_status": r_close.status_code,
                        "response": (r_close.json() if r_close.headers.get("Content-Type","").startswith("application/json") else {"raw": r_close.text})
                        }, 502
        except requests.RequestException as e:
            log(f"NETWORK ERROR | REVERSE-CLOSE | alert_id={alert_id} | {pair} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="reverse_close", instrument=pair,
                status="NETWORK_ERROR", oanda_response=repr(e), oanda_http=None,
                side=side, units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=None, projected_net=None, spread_pips=spread_pips_val
            )
            return {"status": "NETWORK_ERROR", "step": "close"}, 502
        time.sleep(0.2)

    # Net exposure gate
    try:
        current_net = get_net_units_for_instrument(pair)
    except Exception as e:
        log(f"RISK BLOCKED (POS_FETCH_FAIL) | alert_id={alert_id} | {pair} | {repr(e)}")
        db_record_execution(
            alert_id=alert_id, action=action, instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=tp_pips,
            status="RISK_BLOCKED_POS_FETCH_FAIL", oanda_response=repr(e)
        )
        return {"status": "RISK_BLOCKED_POS_FETCH_FAIL"}, 503

    # Determine units (risk sizing)
    try:
        nav, acct_ccy = get_account_nav_and_currency()
        if FORCE_RISK_SIZING or payload.get("units") is None:
            units = compute_units_from_risk(
                instrument=pair, sl_pips=sl_pips, nav=nav, account_ccy=acct_ccy
            )
        else:
            units = int(payload["units"])
            units = min(units, MAX_UNITS)
            units = max(1, units)
    except Exception as e:
        log(f"RISK BLOCKED (SIZING_FAIL) | alert_id={alert_id} | {pair} | {repr(e)}")
        db_record_execution(
            alert_id=alert_id, action=action, instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_SIZING_FAIL", oanda_response=repr(e)
        )
        return {"status": "RISK_BLOCKED_SIZING_FAIL", "error": str(e)}, 403

    signed_units = -units if side == "sell" else units
    projected_net = current_net + signed_units

    if abs(projected_net) > MAX_NET_UNITS:
        log(f"RISK BLOCKED (MAX_NET_UNITS) | alert_id={alert_id} | {pair} current={current_net} new={signed_units} proj={projected_net} cap={MAX_NET_UNITS}")
        db_record_execution(
            alert_id=alert_id, action=action, instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_MAX_NET_UNITS", oanda_response=None
        )
        return {"status": "RISK_BLOCKED_MAX_NET_UNITS", "current_net": current_net, "projected_net": projected_net, "cap": MAX_NET_UNITS}, 403

    # Place order
    try:
        r = place_market_order(instrument=pair, signed_units=signed_units, sl_pips=sl_pips, tp_pips=tp_pips)
        body_text = r.text
        ok = r.ok
        log(f"{action.upper()} | alert_id={alert_id} | {pair} {side} {signed_units} | net={current_net}->{projected_net} | SL={sl_pips} TP={tp_pips} | spread={spread_pips_val} | HTTP {r.status_code}")

        db_record_execution(
            alert_id=alert_id, action=action, instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            oanda_http=r.status_code, status=("OK" if ok else "OANDA_ERROR"),
            oanda_response=body_text
        )

        return {
            "status": "OK" if ok else "OANDA_ERROR",
            "action": action,
            "http_status": r.status_code,
            "instrument": pair,
            "side": side,
            "units": units,
            "signed_units": signed_units,
            "current_net": current_net,
            "projected_net": projected_net,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "spread_pips": spread_pips_val,
            "response": (r.json() if r.headers.get("Content-Type","").startswith("application/json") else {"raw": r.text})
        }, (200 if ok else 502)

    except requests.RequestException as e:
        log(f"NETWORK ERROR | {action.upper()} | alert_id={alert_id} | {pair} | {repr(e)}")
        db_record_execution(
            alert_id=alert_id, action=action, instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="NETWORK_ERROR", oanda_response=repr(e)
        )
        return {"status": "NETWORK_ERROR"}, 502


if __name__ == "__main__":
    # Dev only. Production uses gunicorn.
    app.run(host="0.0.0.0", port=PORT, debug=False)
