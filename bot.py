from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os
import time
import sqlite3
import json
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Optional, Tuple, Dict, Any

getcontext().prec = 28

# Load env
load_dotenv()
app = Flask(__name__)

# =====================
# OpenAI (GPT gatekeeper)
# =====================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GPT_ENABLED = os.getenv("GPT_ENABLED", "false").lower() == "true"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5.2")
GPT_MIN_CONFIDENCE = float(os.getenv("GPT_MIN_CONFIDENCE", "0.55"))
GPT_FAIL_OPEN = os.getenv("GPT_FAIL_OPEN", "false").lower() == "true"

# Safety: default disallows scaling up (1.0 => only reduce or unchanged)
GPT_MAX_UNITS_MULT = float(os.getenv("GPT_MAX_UNITS_MULT", "1.0"))

# Optional: OpenAI client timeout (seconds)
GPT_TIMEOUT_SECONDS = float(os.getenv("GPT_TIMEOUT_SECONDS", "12"))

openai_client = None
if OpenAI and OPENAI_API_KEY:
    try:
        # OpenAI Python SDK: Responses API, supports tools/function calling
        openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=GPT_TIMEOUT_SECONDS)
    except Exception:
        openai_client = None

# Function tool schema (strict mode = schema-locked)
# PATCH: use valid JSON Schema anyOf instead of enum containing None
DECIDE_TRADE_TOOL = [
    {
        "type": "function",
        "name": "decide_trade",
        "description": "Given market features and risk limits/state, decide HOLD/OPEN/CLOSE and (if OPEN) side and bracket.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["HOLD", "OPEN", "CLOSE"]},
                "side": {
                    "anyOf": [
                        {"type": "string", "enum": ["buy", "sell"]},
                        {"type": "null"}
                    ]
                },
                "sl_pips": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "null"}
                    ]
                },
                "tp_pips": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"}
                    ]
                },
                "units_mult": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"}
            },
            # strict mode requires all fields required (use null for optional values)
            "required": ["action", "side", "sl_pips", "tp_pips", "units_mult", "confidence", "reason"],
        }
    }
]

def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def gpt_decide_trade(context: dict) -> dict:
    """
    Returns dict:
      {
        "action": "HOLD|OPEN|CLOSE",
        "side": "buy|sell|null",
        "sl_pips": int|null,
        "tp_pips": int|null,
        "units_mult": float,
        "confidence": float,
        "reason": str
      }

    Fail behavior:
      - GPT_FAIL_OPEN=true  -> HOLD (still conservative; you can change later)
      - GPT_FAIL_OPEN=false -> fail-closed => HOLD
    """
    safe_hold = {
        "action": "HOLD",
        "side": None,
        "sl_pips": None,
        "tp_pips": None,
        "units_mult": 1.0,
        "confidence": 0.0,
        "reason": "GPT_DISABLED_OR_UNAVAILABLE"
    }

    if not GPT_ENABLED:
        safe_hold["confidence"] = 1.0
        safe_hold["reason"] = "GPT_DISABLED"
        return safe_hold

    if not openai_client:
        # No SDK or no key configured (or client init failed)
        if GPT_FAIL_OPEN:
            safe_hold["reason"] = "GPT_NO_CLIENT_FAIL_OPEN_BUT_HOLD"
            return safe_hold
        return safe_hold

    # Your tuned policy (as you posted)
    instructions = (
        "You are an FX execution gatekeeper. Output ONLY the decide_trade tool call.\n\n"

        "Interpretation of inputs:\n"
        "- features.adx: trend strength\n"
        "- features.ema_sep_pips: separation between fast/slow EMAs (small => chop)\n"
        "- features.atr_pips: recent volatility in pips\n"
        "- features.range_pips: current bar range in pips (very small => noise)\n"
        "- features.buyCross / sellCross: EMA cross events\n"
        "- features.trendUp / trendDown: fast EMA above/below slow EMA\n"
        "- features.fast_slope_pips / slow_slope_pips: EMA slope direction\n\n"

        "Hard constraints (must obey):\n"
        "- If state.spread_pips is null, or (limits.max_spread_pips>0 and state.spread_pips>limits.max_spread_pips): action=HOLD.\n"
        "- units_mult must be <= limits.gpt_max_units_mult.\n"
        "- sl_pips must be within [limits.min_sl_pips, limits.max_sl_pips] when action=OPEN.\n"
        "- tp_pips must be within [0, limits.max_tp_pips] when action=OPEN.\n\n"

        "Extra noise filter using range_pips:\n"
        "- If features.range_pips < 2.0 AND features.adx < 18: action=HOLD.\n\n"

        "Chop filter (avoid whipsaw):\n"
        "- If features.adx < 15 AND features.ema_sep_pips < 1.2: action=HOLD.\n\n"

        "OPEN rules (trade relatively often but not random):\n"
        "- If state.current_net == 0:\n"
        "    * If features.buyCross is true AND (features.adx >= 16 OR features.ema_sep_pips >= 1.5): action=OPEN, side=buy.\n"
        "    * Else if features.sellCross is true AND (features.adx >= 16 OR features.ema_sep_pips >= 1.5): action=OPEN, side=sell.\n"
        "    * Else if features.adx >= 22 AND features.trendUp is true AND features.fast_slope_pips > 0 AND features.slow_slope_pips > 0: action=OPEN, side=buy.\n"
        "    * Else if features.adx >= 22 AND features.trendDown is true AND features.fast_slope_pips < 0 AND features.slow_slope_pips < 0: action=OPEN, side=sell.\n"
        "    * Else action=HOLD.\n\n"

        "CLOSE rules (GPT is allowed to close):\n"
        "- If state.current_net > 0 (currently long):\n"
        "    * If features.sellCross is true: action=CLOSE.\n"
        "    * Else if features.adx < 14: action=CLOSE.\n"
        "    * Else action=HOLD.\n"
        "- If state.current_net < 0 (currently short):\n"
        "    * If features.buyCross is true: action=CLOSE.\n"
        "    * Else if features.adx < 14: action=CLOSE.\n"
        "    * Else action=HOLD.\n\n"

        "SL/TP selection (use ATR when available):\n"
        "- If features.atr_pips is present and > 0:\n"
        "    * sl_pips = round(clamp(1.4 * atr_pips, limits.min_sl_pips, limits.max_sl_pips)).\n"
        "    * tp_pips = round(clamp(2.0 * sl_pips, 0, limits.max_tp_pips)).\n"
        "- Else:\n"
        "    * sl_pips = clamp(hint.sl_pips, limits.min_sl_pips, limits.max_sl_pips).\n"
        "    * tp_pips = clamp(hint.tp_pips, 0, limits.max_tp_pips).\n\n"

        "Sizing modifier (only reduce size):\n"
        "- Default units_mult = 1.0.\n"
        "- If features.adx is between 15 and 17 or ema_sep_pips is between 1.2 and 1.5: units_mult = 0.5.\n\n"

        "Confidence:\n"
        "- If OPEN via buyCross/sellCross with adx>=18: confidence>=0.70.\n"
        "- If OPEN via trend continuation rule (adx>=22): confidence>=0.75.\n"
        "- If HOLD due to chop/spread/noise: confidence can be 0.6-0.9.\n"
        "- If unsure: action=HOLD, confidence<=0.55.\n"
    )

    try:
        resp = openai_client.responses.create(
            model=GPT_MODEL,
            instructions=instructions,
            tools=DECIDE_TRADE_TOOL,
            tool_choice={"type": "function", "name": "decide_trade"},
            parallel_tool_calls=False,
            input=[{"role": "user", "content": json.dumps(context)}],
        )

        for item in _get_attr(resp, "output", []) or []:
            if _get_attr(item, "type") == "function_call" and _get_attr(item, "name") == "decide_trade":
                args_raw = _get_attr(item, "arguments", "{}")
                out = json.loads(args_raw) if isinstance(args_raw, str) else args_raw

                action = str(out.get("action", "HOLD")).upper().strip()
                if action not in {"HOLD", "OPEN", "CLOSE"}:
                    action = "HOLD"

                side = out.get("side", None)
                if side is not None:
                    side = str(side).lower().strip()
                    if side not in {"buy", "sell"}:
                        side = None

                sl_pips = out.get("sl_pips", None)
                tp_pips = out.get("tp_pips", None)
                sl_pips = int(sl_pips) if sl_pips is not None else None
                tp_pips = int(tp_pips) if tp_pips is not None else None

                units_mult = float(out.get("units_mult", 1.0))
                units_mult = max(0.1, min(units_mult, GPT_MAX_UNITS_MULT))

                confidence = float(out.get("confidence", 0.0))
                confidence = max(0.0, min(confidence, 1.0))

                reason = str(out.get("reason", "")).strip()[:500]

                return {
                    "action": action,
                    "side": side,
                    "sl_pips": sl_pips,
                    "tp_pips": tp_pips,
                    "units_mult": units_mult,
                    "confidence": confidence,
                    "reason": reason
                }

        raise RuntimeError("No decide_trade tool call returned")

    except Exception as e:
        safe_hold["reason"] = f"GPT_ERROR_FAIL_CLOSED: {repr(e)}"
        return safe_hold

# =====================
# Config
# =====================
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

WEBHOOK_KEY = os.getenv("WEBHOOK_KEY", "")
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

# Risk caps
MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))
MAX_NET_UNITS = int(os.getenv("MAX_NET_UNITS", "2000"))

# Stale alert guard (0 disables)
MAX_ALERT_AGE_SECONDS = int(os.getenv("MAX_ALERT_AGE_SECONDS", "0"))

# Bracket defaults (pips)
SL_PIPS_DEFAULT = int(os.getenv("SL_PIPS", "20"))
TP_PIPS_DEFAULT = int(os.getenv("TP_PIPS", "40"))

# Hard caps for brackets (pips)
MAX_SL_PIPS = int(os.getenv("MAX_SL_PIPS", str(SL_PIPS_DEFAULT)))
MAX_TP_PIPS = int(os.getenv("MAX_TP_PIPS", str(TP_PIPS_DEFAULT)))
MIN_SL_PIPS = int(os.getenv("MIN_SL_PIPS", "1"))

# Spread filter
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "2"))   # 0 disables

# Equity sizing
RISK_PCT = Decimal(os.getenv("RISK_PCT", "0.0025"))
MAX_RISK_USD = Decimal(os.getenv("MAX_RISK_USD", "100"))
DAILY_LOSS_LIMIT_USD = Decimal(os.getenv("DAILY_LOSS_LIMIT_USD", "0"))

FORCE_RISK_SIZING = os.getenv("FORCE_RISK_SIZING", "true").lower() == "true"
ALLOW_NONQUOTE_HOME_SIZING = os.getenv("ALLOW_NONQUOTE_HOME_SIZING", "false").lower() == "true"

# Persistent state
DB_PATH = os.getenv("DB_PATH", "bot.db")
ALERT_TTL_SECONDS = int(os.getenv("ALERT_TTL_SECONDS", "60"))

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
# DB setup + migration
# =====================
def db_conn():
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
            oanda_response TEXT,
            meta TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_state (
            day TEXT PRIMARY KEY,
            nav_start TEXT NOT NULL
        );
    """)

    cur.execute("PRAGMA table_info(executions);")
    cols = {row["name"] for row in cur.fetchall()}
    if "meta" not in cols:
        cur.execute("ALTER TABLE executions ADD COLUMN meta TEXT;")

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
            current_net, projected_net, spread_pips, oanda_http, status, oanda_response, meta
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        kwargs.get("meta"),
    ))
    conn.commit()
    conn.close()

# =====================
# Dedupe
# =====================
def seen_before(alert_id: str) -> bool:
    now = int(time.time())
    cutoff = now - ALERT_TTL_SECONDS

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM alerts WHERE first_seen_ts < ?", (cutoff,))
    cur.execute("SELECT first_seen_ts FROM alerts WHERE alert_id = ?", (alert_id,))
    row = cur.fetchone()
    if row is not None:
        conn.commit()
        conn.close()
        return True

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
            short_units = int(pos.get("short", {}).get("units", "0"))
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

def get_day_start_nav(nav_now: Decimal) -> Decimal:
    day = datetime.datetime.utcnow().date().isoformat()
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT nav_start FROM daily_state WHERE day = ?", (day,))
    row = cur.fetchone()
    if row is not None:
        conn.close()
        return Decimal(row["nav_start"])
    cur.execute("INSERT INTO daily_state (day, nav_start) VALUES (?, ?)", (day, str(nav_now)))
    conn.commit()
    conn.close()
    return nav_now

def close_position(instrument: str):
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
    if short_units < 0:
        payload["shortUnits"] = "ALL"

    if not payload:
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

def sl_tp_distance(instrument: str, sl_pips: int, tp_pips: int):
    pip = pip_size_for(instrument)
    sl_dist = (pip * Decimal(sl_pips)) if sl_pips and sl_pips > 0 else None
    tp_dist = (pip * Decimal(tp_pips)) if tp_pips and tp_pips > 0 else None
    return sl_dist, tp_dist

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
def loss_per_unit_home(*, instrument: str, sl_pips: int, account_ccy: str) -> Decimal:
    base, quote = instrument.split("_")
    pip = pip_size_for(instrument)
    sl_dist_quote = pip * Decimal(sl_pips)

    if sl_pips <= 0:
        raise RuntimeError("sl_pips must be > 0 for risk sizing")

    if quote == account_ccy:
        return sl_dist_quote

    if not ALLOW_NONQUOTE_HOME_SIZING:
        raise RuntimeError(
            f"Cannot risk-size safely: quote={quote} != account_ccy={account_ccy}. "
            f"Set ALLOW_NONQUOTE_HOME_SIZING=true to allow approximate conversion."
        )

    quote_home = f"{quote}_{account_ccy}"
    home_quote = f"{account_ccy}_{quote}"

    conv = None
    try:
        r = oanda_get_pricing(quote_home)
        if r.ok and r.json().get("prices"):
            p = r.json()["prices"][0]
            bid = Decimal(p["bids"][0]["price"])
            ask = Decimal(p["asks"][0]["price"])
            conv = (bid + ask) / Decimal(2)
    except Exception:
        conv = None

    if conv is None:
        r2 = oanda_get_pricing(home_quote)
        if not r2.ok or not r2.json().get("prices"):
            raise RuntimeError(f"Failed to obtain conversion rate {quote}<->{account_ccy}")
        p2 = r2.json()["prices"][0]
        bid2 = Decimal(p2["bids"][0]["price"])
        ask2 = Decimal(p2["asks"][0]["price"])
        mid2 = (bid2 + ask2) / Decimal(2)
        conv = Decimal(1) / mid2

    loss_unit_home = sl_dist_quote * conv
    if loss_unit_home <= 0:
        raise RuntimeError("Computed invalid loss_per_unit_home")

    return loss_unit_home

def compute_units_from_risk(*, instrument: str, sl_pips: int, nav: Decimal, account_ccy: str) -> int:
    loss_unit = loss_per_unit_home(instrument=instrument, sl_pips=sl_pips, account_ccy=account_ccy)
    risk_budget_home = min(nav * RISK_PCT, MAX_RISK_USD, nav)
    units = int((risk_budget_home / loss_unit).to_integral_value(rounding=ROUND_DOWN))
    return max(1, units)

# =====================
# Validation
# =====================
def validate_payload(data: dict):
    alert_id = str(data.get("alert_id", "")).strip()
    action = str(data.get("action", "observe")).lower().strip()
    pair = str(data.get("pair", "")).upper().strip()

    if not alert_id:
        return None, "Missing alert_id"
    if action not in {"observe", "open", "close", "close_all"}:
        return None, "Invalid action (observe|open|close|close_all)"

    if action == "close_all":
        return {"alert_id": alert_id, "action": action}, None

    if action in {"observe", "close", "open"}:
        if "_" not in pair:
            return None, "Invalid pair format (use USD_JPY)"

    if action == "close":
        return {"alert_id": alert_id, "action": action, "pair": pair}, None

    sl_pips = data.get("sl_pips", SL_PIPS_DEFAULT)
    tp_pips = data.get("tp_pips", TP_PIPS_DEFAULT)
    try:
        sl_pips = int(sl_pips)
        tp_pips = int(tp_pips)
    except Exception:
        return None, "sl_pips/tp_pips must be integers"

    sl_pips = max(MIN_SL_PIPS, min(sl_pips if sl_pips > 0 else SL_PIPS_DEFAULT, MAX_SL_PIPS))
    tp_pips = max(0, min(tp_pips if tp_pips >= 0 else TP_PIPS_DEFAULT, MAX_TP_PIPS))

    features = data.get("features", {}) or {}
    if not isinstance(features, dict):
        return None, "features must be an object"

    if action == "observe":
        return {
            "alert_id": alert_id,
            "action": "observe",
            "pair": pair,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "features": features
        }, None

    # open requires side
    side = str(data.get("side", "")).lower().strip()
    if side not in {"buy", "sell"}:
        return None, "Invalid side (buy|sell)"

    return {
        "alert_id": alert_id,
        "action": "open",
        "pair": pair,
        "side": side,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "features": features
    }, None

# =====================
# Core execution helper
# =====================
def execute_open_trade(
    *,
    alert_id: str,
    pair: str,
    side: str,
    sl_pips: int,
    tp_pips: int,
    units_mult: float,
    spread_pips_val: Optional[float],
    current_net: int,
    nav: Decimal,
    acct_ccy: str,
    meta: dict
) -> Tuple[Dict[str, Any], int]:

    # Spread gate (OPEN only)
    if MAX_SPREAD_PIPS > 0:
        if spread_pips_val is None:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=None,
                status="RISK_BLOCKED_PRICE_FETCH_FAIL", oanda_http=None, oanda_response=None,
                meta=json.dumps(meta)
            )
            return {"status": "RISK_BLOCKED_PRICE_FETCH_FAIL"}, 503

        if spread_pips_val > MAX_SPREAD_PIPS:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_SPREAD", oanda_http=None, oanda_response=None,
                meta=json.dumps(meta)
            )
            return {"status": "RISK_BLOCKED_SPREAD", "spread_pips": spread_pips_val, "cap": MAX_SPREAD_PIPS}, 403

    # Daily loss cap
    if DAILY_LOSS_LIMIT_USD > 0:
        day_start_nav = get_day_start_nav(nav)
        if day_start_nav - nav >= DAILY_LOSS_LIMIT_USD:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_DAILY_LOSS_LIMIT", oanda_http=None, oanda_response=None,
                meta=json.dumps(meta)
            )
            return {
                "status": "RISK_BLOCKED_DAILY_LOSS_LIMIT",
                "nav_start": str(day_start_nav),
                "nav_now": str(nav),
                "limit": str(DAILY_LOSS_LIMIT_USD)
            }, 403

    # Risk sizing based on sl_pips
    units = compute_units_from_risk(instrument=pair, sl_pips=sl_pips, nav=nav, account_ccy=acct_ccy)

    # Apply GPT reduction
    units_mult = max(0.1, min(float(units_mult), GPT_MAX_UNITS_MULT))
    units = max(1, int(units * units_mult))
    units = min(units, MAX_UNITS)

    # Absolute max-risk gate
    loss_unit = loss_per_unit_home(instrument=pair, sl_pips=sl_pips, account_ccy=acct_ccy)
    expected_loss = loss_unit * Decimal(units)
    max_loss = min(nav, MAX_RISK_USD)
    if expected_loss > max_loss:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_MAX_RISK_USD", oanda_http=None, oanda_response=None,
            meta=json.dumps(meta)
        )
        return {"status": "RISK_BLOCKED_MAX_RISK_USD", "expected_loss": str(expected_loss), "cap": str(max_loss)}, 403

    signed_units = -units if side == "sell" else units
    projected_net = current_net + signed_units

    # Net exposure gate
    if abs(projected_net) > MAX_NET_UNITS:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_MAX_NET_UNITS", oanda_http=None, oanda_response=None,
            meta=json.dumps(meta)
        )
        return {"status": "RISK_BLOCKED_MAX_NET_UNITS", "current_net": current_net, "projected_net": projected_net, "cap": MAX_NET_UNITS}, 403

    # Place order
    try:
        r = place_market_order(instrument=pair, signed_units=signed_units, sl_pips=sl_pips, tp_pips=tp_pips)
        ok = r.ok

        log(f"OPEN | alert_id={alert_id} | {pair} {side} {signed_units} | net={current_net}->{projected_net} | SL={sl_pips} TP={tp_pips} | spread={spread_pips_val} | HTTP {r.status_code}")

        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            oanda_http=r.status_code, status=("OK" if ok else "OANDA_ERROR"),
            oanda_response=r.text,
            meta=json.dumps(meta)
        )

        return {
            "status": "OK" if ok else "OANDA_ERROR",
            "action": "open",
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
            "meta": meta,
            "response": (r.json() if r.headers.get("Content-Type","").startswith("application/json") else {"raw": r.text})
        }, (200 if ok else 502)

    except requests.RequestException as e:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="NETWORK_ERROR", oanda_http=None, oanda_response=repr(e),
            meta=json.dumps(meta)
        )
        return {"status": "NETWORK_ERROR", "error": str(e)}, 502

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
        "min_sl_pips": MIN_SL_PIPS,
        "max_sl_pips": MAX_SL_PIPS,
        "max_tp_pips": MAX_TP_PIPS,
        "max_spread_pips": MAX_SPREAD_PIPS,
        "risk_pct": str(RISK_PCT),
        "max_risk_usd": str(MAX_RISK_USD),
        "daily_loss_limit_usd": str(DAILY_LOSS_LIMIT_USD),
        "force_risk_sizing": FORCE_RISK_SIZING,
        "db_path": DB_PATH,
        "alert_ttl_seconds": ALERT_TTL_SECONDS,
        "gpt_enabled": GPT_ENABLED,
        "gpt_model": GPT_MODEL,
        "gpt_min_confidence": GPT_MIN_CONFIDENCE,
        "gpt_fail_open": GPT_FAIL_OPEN,
        "gpt_max_units_mult": GPT_MAX_UNITS_MULT,
        "gpt_timeout_seconds": GPT_TIMEOUT_SECONDS
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
    key = request.args.get("key", "")
    if key != WEBHOOK_KEY:
        log("AUTH BLOCKED | invalid webhook key")
        return {"status": "UNAUTHORIZED"}, 401

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
            side=payload.get("side"), units=None,
            sl_pips=payload.get("sl_pips"), tp_pips=payload.get("tp_pips"),
            current_net=None, projected_net=None, spread_pips=None,
            status="DUPLICATE_IGNORED", oanda_http=None, oanda_response=None, meta=None
        )
        return {"status": "DUPLICATE_IGNORED"}, 200

    action = payload["action"]

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
                    oanda_response=str(item["response"]),
                    meta=None
                )
            return {"status": "OK", "results": results}, 200
        except Exception as e:
            log(f"ERROR | CLOSE_ALL | alert_id={alert_id} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="close_all", instrument="ALL",
                status="ERROR", oanda_response=repr(e), oanda_http=None, meta=None
            )
            return {"status": "ERROR", "error": str(e)}, 502

    pair = payload["pair"]

    if action == "close":
        try:
            r = close_position(pair)
            log(f"CLOSE | alert_id={alert_id} | {pair} | HTTP {r.status_code}")
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair,
                side=None, units=None, sl_pips=None, tp_pips=None,
                current_net=None, projected_net=None, spread_pips=None,
                oanda_http=r.status_code, status=("OK" if r.ok else "OANDA_ERROR"),
                oanda_response=r.text, meta=None
            )
            return {"status": "OK" if r.ok else "OANDA_ERROR",
                    "http_status": r.status_code,
                    "response": (r.json() if r.headers.get("Content-Type","").startswith("application/json") else {"raw": r.text})
                    }, (200 if r.ok else 502)
        except requests.RequestException as e:
            log(f"NETWORK ERROR | CLOSE | alert_id={alert_id} | {pair} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair,
                status="NETWORK_ERROR", oanda_response=repr(e), oanda_http=None,
                side=None, units=None, sl_pips=None, tp_pips=None,
                current_net=None, projected_net=None, spread_pips=None,
                meta=None
            )
            return {"status": "NETWORK_ERROR"}, 502

    # =====================
    # OBSERVE -> GPT DECIDES
    # =====================
    if action == "observe":
        features = payload.get("features", {}) or {}
        hint_sl = payload.get("sl_pips", SL_PIPS_DEFAULT)
        hint_tp = payload.get("tp_pips", TP_PIPS_DEFAULT)

        # NEW: Stale alert guard (uses TradingView bar time_ms)
        if MAX_ALERT_AGE_SECONDS > 0:
            try:
                tv_time_ms = features.get("time_ms", None)
                if tv_time_ms is not None:
                    tv_time_ms = int(tv_time_ms)
                    now_ms = int(time.time() * 1000)
                    age_sec = (now_ms - tv_time_ms) / 1000.0
                    if age_sec > MAX_ALERT_AGE_SECONDS:
                        log(f"STALE_ALERT_IGNORED | alert_id={alert_id} | pair={pair} | age_sec={age_sec:.1f} > {MAX_ALERT_AGE_SECONDS}")
                        db_record_execution(
                            alert_id=alert_id, action="observe", instrument=pair,
                            side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                            current_net=None, projected_net=None, spread_pips=None,
                            status="STALE_ALERT_IGNORED", oanda_http=None, oanda_response=None,
                            meta=json.dumps({"age_sec": age_sec, "max_age_sec": MAX_ALERT_AGE_SECONDS, "time_ms": tv_time_ms})
                        )
                        return {"status": "STALE_ALERT_IGNORED", "age_sec": age_sec, "max_age_sec": MAX_ALERT_AGE_SECONDS}, 200
            except Exception as e:
                # If parsing fails, ignore the guard and continue
                log(f"STALE_GUARD_PARSE_FAIL | alert_id={alert_id} | {repr(e)}")

        # Spread best effort
        spread_pips_val = None
        if MAX_SPREAD_PIPS > 0:
            try:
                spread_pips_val = get_spread_pips(pair)
            except Exception as e:
                log(f"PRICE_FETCH_FAIL | alert_id={alert_id} | {pair} | {repr(e)}")
                spread_pips_val = None

        # Current net
        try:
            current_net = get_net_units_for_instrument(pair)
        except Exception as e:
            log(f"POS_FETCH_FAIL | alert_id={alert_id} | {pair} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                current_net=None, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_POS_FETCH_FAIL", oanda_http=None, oanda_response=repr(e),
                meta=None
            )
            return {"status": "RISK_BLOCKED_POS_FETCH_FAIL"}, 503

        # NAV (for sizing)
        try:
            nav, acct_ccy = get_account_nav_and_currency()
        except Exception as e:
            log(f"ACCT_FETCH_FAIL | alert_id={alert_id} | {repr(e)}")
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_ACCT_FETCH_FAIL", oanda_http=None, oanda_response=repr(e),
                meta=None
            )
            return {"status": "RISK_BLOCKED_ACCT_FETCH_FAIL"}, 503

        gpt_ctx = {
            "pair": pair,
            "features": features,
            "hint": {"sl_pips": hint_sl, "tp_pips": hint_tp},
            "limits": {
                "min_sl_pips": MIN_SL_PIPS,
                "max_sl_pips": MAX_SL_PIPS,
                "max_tp_pips": MAX_TP_PIPS,
                "max_units": MAX_UNITS,
                "max_net_units": MAX_NET_UNITS,
                "max_spread_pips": MAX_SPREAD_PIPS,
                "gpt_max_units_mult": GPT_MAX_UNITS_MULT
            },
            "state": {
                "spread_pips": spread_pips_val,
                "current_net": current_net
            },
            "risk": {
                "risk_pct": str(RISK_PCT),
                "max_risk_usd": str(MAX_RISK_USD),
                "daily_loss_limit_usd": str(DAILY_LOSS_LIMIT_USD)
            }
        }

        gpt_decision = gpt_decide_trade(gpt_ctx)

        # Confidence gate
        if gpt_decision["confidence"] < GPT_MIN_CONFIDENCE:
            log(f"GPT_HOLD_LOW_CONF | alert_id={alert_id} | {pair} | {gpt_decision}")
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="GPT_HOLD_LOW_CONF", oanda_http=None, oanda_response=None,
                meta=json.dumps(gpt_decision)
            )
            return {"status": "GPT_HOLD_LOW_CONF", "decision": gpt_decision}, 200

        if gpt_decision["action"] == "HOLD":
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="GPT_HOLD", oanda_http=None, oanda_response=None,
                meta=json.dumps(gpt_decision)
            )
            return {"status": "GPT_HOLD", "decision": gpt_decision}, 200

        if gpt_decision["action"] == "CLOSE":
            try:
                r = close_position(pair)
                db_record_execution(
                    alert_id=alert_id, action="close", instrument=pair,
                    side=None, units=None, sl_pips=None, tp_pips=None,
                    current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                    oanda_http=r.status_code, status=("OK" if r.ok else "OANDA_ERROR"),
                    oanda_response=r.text,
                    meta=json.dumps(gpt_decision)
                )
                return {
                    "status": "OK" if r.ok else "OANDA_ERROR",
                    "action": "close",
                    "http_status": r.status_code,
                    "instrument": pair,
                    "decision": gpt_decision,
                    "response": (r.json() if r.headers.get("Content-Type","").startswith("application/json") else {"raw": r.text})
                }, (200 if r.ok else 502)
            except Exception as e:
                db_record_execution(
                    alert_id=alert_id, action="close", instrument=pair,
                    side=None, units=None, sl_pips=None, tp_pips=None,
                    current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                    status="ERROR", oanda_http=None, oanda_response=repr(e),
                    meta=json.dumps(gpt_decision)
                )
                return {"status": "ERROR", "action": "close", "error": str(e), "decision": gpt_decision}, 502

        # PATCH: One-position-at-a-time.
        # If you already have a position, do NOT allow OPEN (prevents stacking/pyramiding).
        if gpt_decision["action"] == "OPEN" and current_net != 0:
            log(f"GPT_HOLD_ALREADY_IN_POSITION | alert_id={alert_id} | {pair} | net={current_net} | {gpt_decision}")
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="GPT_HOLD_ALREADY_IN_POSITION", oanda_http=None, oanda_response=None,
                meta=json.dumps(gpt_decision)
            )
            return {"status": "GPT_HOLD_ALREADY_IN_POSITION", "decision": gpt_decision}, 200

        # OPEN (only when flat)
        if gpt_decision["action"] == "OPEN":
            side = gpt_decision["side"]
            if side not in ("buy", "sell"):
                db_record_execution(
                    alert_id=alert_id, action="observe", instrument=pair,
                    side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                    current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                    status="GPT_HOLD_BAD_SIDE", oanda_http=None, oanda_response=None,
                    meta=json.dumps(gpt_decision)
                )
                return {"status": "GPT_HOLD_BAD_SIDE", "decision": gpt_decision}, 200

            sl_pips = gpt_decision["sl_pips"] if gpt_decision["sl_pips"] is not None else hint_sl
            tp_pips = gpt_decision["tp_pips"] if gpt_decision["tp_pips"] is not None else hint_tp
            sl_pips = int(max(MIN_SL_PIPS, min(sl_pips, MAX_SL_PIPS)))
            tp_pips = int(max(0, min(tp_pips, MAX_TP_PIPS)))

            meta = {"gpt": gpt_decision, "features": features}
            resp, status = execute_open_trade(
                alert_id=alert_id,
                pair=pair,
                side=side,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                units_mult=gpt_decision["units_mult"],
                spread_pips_val=spread_pips_val,
                current_net=current_net,
                nav=nav,
                acct_ccy=acct_ccy,
                meta=meta
            )
            return resp, status

        db_record_execution(
            alert_id=alert_id, action="observe", instrument=pair,
            side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="GPT_HOLD_UNKNOWN", oanda_http=None, oanda_response=None,
            meta=json.dumps(gpt_decision)
        )
        return {"status": "GPT_HOLD_UNKNOWN", "decision": gpt_decision}, 200

    # =====================
    # Manual OPEN (optional)
    # PATCH: also enforce one-position-at-a-time for manual opens
    # =====================
    if action == "open":
        side = payload["side"]
        sl_pips = payload["sl_pips"]
        tp_pips = payload["tp_pips"]
        features = payload.get("features", {}) or {}

        # Spread best-effort
        spread_pips_val = None
        if MAX_SPREAD_PIPS > 0:
            try:
                spread_pips_val = get_spread_pips(pair)
            except Exception:
                spread_pips_val = None

        try:
            current_net = get_net_units_for_instrument(pair)
            nav, acct_ccy = get_account_nav_and_currency()
        except Exception as e:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=None, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_STATE_FETCH_FAIL", oanda_http=None, oanda_response=repr(e),
                meta=None
            )
            return {"status": "RISK_BLOCKED_STATE_FETCH_FAIL", "error": str(e)}, 503

        if current_net != 0:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="MANUAL_BLOCKED_ALREADY_IN_POSITION", oanda_http=None, oanda_response=None,
                meta=json.dumps({"current_net": current_net})
            )
            return {"status": "MANUAL_BLOCKED_ALREADY_IN_POSITION", "current_net": current_net}, 200

        meta = {"manual": True, "features": features}
        resp, status = execute_open_trade(
            alert_id=alert_id,
            pair=pair,
            side=side,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            units_mult=1.0,
            spread_pips_val=spread_pips_val,
            current_net=current_net,
            nav=nav,
            acct_ccy=acct_ccy,
            meta=meta
        )
        return resp, status

    return {"status": "BAD_REQUEST", "error": "Unhandled action"}, 400


if __name__ == "__main__":
    # Dev only. Production uses gunicorn.
    app.run(host="0.0.0.0", port=PORT, debug=False)
