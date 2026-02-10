from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os
import time
import sqlite3
import json
import random
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Optional, Tuple, Dict, Any, List

getcontext().prec = 28

# Load env
load_dotenv()
app = Flask(__name__)

# ============================================================
# OpenAI (GPT gatekeeper)
# ============================================================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GPT_ENABLED = os.getenv("GPT_ENABLED", "false").lower() == "true"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini")
GPT_MIN_CONFIDENCE = float(os.getenv("GPT_MIN_CONFIDENCE", "0.55"))

# Safety: default disallows scaling up (1.0 => only reduce or unchanged)
GPT_MAX_UNITS_MULT = float(os.getenv("GPT_MAX_UNITS_MULT", "1.0"))
GPT_TIMEOUT_SECONDS = float(os.getenv("GPT_TIMEOUT_SECONDS", "30"))
GPT_ENTRY_ADX = int(os.getenv("GPT_ENTRY_ADX", "25"))

openai_client = None
if OpenAI and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=GPT_TIMEOUT_SECONDS)
    except Exception:
        openai_client = None

# Function tool schema
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
                # NOTE: We keep tp_pips for schema compatibility, but we disable TP in execution if TAKE_PROFIT_ENABLED=false.
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
            "required": ["action", "side", "sl_pips", "tp_pips", "units_mult", "confidence", "reason"],
        }
    }
]


def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def log(msg: str):
    with open("trades.log", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")


def gpt_decide_trade(context: dict) -> dict:
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
        safe_hold["reason"] = "GPT_NO_CLIENT"
        return safe_hold

    instructions = (
        "You are an FX scalper. Output ONLY the decide_trade tool call.\n\n"
        "Hard rules:\n"
        f"- MACRO BUFFER: You must respect a neutral zone of +/- {MACRO_BUFFER_PIPS} pips around the EMA 200.\n" 
        "- Prefer HOLD unless entry is high quality.\n"
        "- If recommending OPEN: choose side; provide sl_pips; set tp_pips=0 (let engine use default).\n\n"
        "Strategy (CASH FLOW / SCALPING):\n"
        "1) MACRO FILTER: Respect the EMA 200 bias as described above.\n"
        "2) ENTRY: Look for short-term momentum. We want 'base hits' (10-15 pips). Speed is key.\n"
        "3) EXIT: The engine has a hard 15-pip TP. However, if the trade stalls or structure breaks, signal CLOSE immediately. Do not wait for a homerun.\n"
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

                reason = str(out.get("reason", "")).strip()[:1200]

                # Keep DB clean
                if action != "OPEN":
                    side = None
                    sl_pips = None
                    tp_pips = None
                    units_mult = 1.0

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


# ============================================================
# Config
# ============================================================
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()

WEBHOOK_KEY = os.getenv("WEBHOOK_KEY", "")
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"

MAX_UNITS = int(os.getenv("MAX_UNITS", "5000"))
MAX_NET_UNITS = int(os.getenv("MAX_NET_UNITS", "15000"))
GLOBAL_MAX_UNITS = int(os.getenv("GLOBAL_MAX_UNITS", "30000"))

MAX_ALERT_AGE_SECONDS = int(os.getenv("MAX_ALERT_AGE_SECONDS", "1200"))

SL_PIPS_DEFAULT = int(os.getenv("SL_PIPS", "30"))
TP_PIPS_DEFAULT = int(os.getenv("TP_PIPS", "60"))
MAX_SL_PIPS = int(os.getenv("MAX_SL_PIPS", "100"))
MAX_TP_PIPS = int(os.getenv("MAX_TP_PIPS", "200"))

TP_ATR_MULTIPLIER = float(os.getenv("TP_ATR_MULTIPLIER", "3.0"))
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", "2.5"))
MIN_SL_PIPS = int(os.getenv("MIN_SL_PIPS", "15"))

MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "4.0"))
MIN_EMA_SEP_PIPS = float(os.getenv("MIN_EMA_SEP_PIPS", "1.5"))
MACRO_BUFFER_PIPS = float(os.getenv("MACRO_BUFFER_PIPS", "8.0"))

# Pair allowlist (comma-separated)
PAIRS_CSV = os.getenv("PAIRS", "EUR_USD,USD_JPY,GBP_USD,AUD_USD,EUR_GBP")
ALLOWED_PAIRS = {p.strip().upper() for p in PAIRS_CSV.split(",") if p.strip()}

# ============================================================
# FIFO safeguard (OANDA FIFO accounts)
# ============================================================
FIFO_ENFORCE_UNIQUE_UNITS = os.getenv("FIFO_ENFORCE_UNIQUE_UNITS", "true").lower() == "true"
FIFO_BUMP_LIMIT = int(os.getenv("FIFO_BUMP_LIMIT", "50"))

# ============================================================
# Rollover / spread policy
# ============================================================
ROLLOVER_BLOCK_OPENS = os.getenv("ROLLOVER_BLOCK_OPENS", "true").lower() == "true"
ROLLOVER_START_UTC = os.getenv("ROLLOVER_START_UTC", "21:55")  # HH:MM (UTC)
ROLLOVER_END_UTC = os.getenv("ROLLOVER_END_UTC", "22:45")      # HH:MM (UTC)

EXIT_SPREAD_MULT = float(os.getenv("EXIT_SPREAD_MULT", "1.5"))
EXIT_SPREAD_MIN_ADD = float(os.getenv("EXIT_SPREAD_MIN_ADD", "0.0"))

# ============================================================
# Session/time-of-day filter (ENTRY ONLY)
# ============================================================
SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "true").lower() == "true"
SESSION_START_UTC = os.getenv("SESSION_START_UTC", "13:00")   # e.g., 13:00 UTC
SESSION_END_UTC = os.getenv("SESSION_END_UTC", "17:00")       # e.g., 17:00 UTC

# ============================================================
# Trade Degradation Exit (stall + profit-protect)
# ============================================================
DEGRADE_EXIT_ENABLED = os.getenv("DEGRADE_EXIT_ENABLED", "true").lower() == "true"
DEGRADE_MIN_MINUTES = int(os.getenv("DEGRADE_MIN_MINUTES", "30"))
DEGRADE_STALL_PIPS = float(os.getenv("DEGRADE_STALL_PIPS", "2.0"))
DEGRADE_ADX_MAX = float(os.getenv("DEGRADE_ADX_MAX", "18.0"))
DEGRADE_EMA_SEP_PIPS = float(os.getenv("DEGRADE_EMA_SEP_PIPS", "2.0"))
DEGRADE_PROFIT_PROTECT_MIN_MINUTES = int(os.getenv("DEGRADE_PROFIT_PROTECT_MIN_MINUTES", "10"))
DEGRADE_PROFIT_PROTECT_PIPS = float(os.getenv("DEGRADE_PROFIT_PROTECT_PIPS", "12.0"))

# Pair-specific profit-protect override (optional envs)
DEGRADE_PROFIT_PROTECT_PIPS_USDJPY = float(os.getenv("DEGRADE_PROFIT_PROTECT_PIPS_USDJPY", "15.0"))
DEGRADE_PROFIT_PROTECT_PIPS_DEFAULT = float(os.getenv("DEGRADE_PROFIT_PROTECT_PIPS_DEFAULT", str(DEGRADE_PROFIT_PROTECT_PIPS)))

# ============================================================
# Trailing exit (Chandelier-style)
# ============================================================
TAKE_PROFIT_ENABLED = os.getenv("TAKE_PROFIT_ENABLED", "false").lower() == "true"
TRAIL_ENABLED = os.getenv("TRAIL_ENABLED", "true").lower() == "true"
TRAIL_LOOKBACK_BARS = int(os.getenv("TRAIL_LOOKBACK_BARS", "30"))   # use last N candles from features.lookback
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "3.0"))          # chandelier ATR multiplier
TRAIL_UPDATE_SECONDS = int(os.getenv("TRAIL_UPDATE_SECONDS", "60")) # throttle updates per instrument

BE_TRIGGER_PIPS = float(os.getenv("BE_TRIGGER_PIPS", "10.0"))       # move to breakeven+offset once >= this
BE_OFFSET_PIPS = float(os.getenv("BE_OFFSET_PIPS", "1.0"))          # offset beyond entry to cover costs

# ============================================================
# USD basket / correlation cap
# ============================================================
USD_BASKET_MAX_POS = int(os.getenv("USD_BASKET_MAX_POS", "2"))  # max simultaneous "long USD" or "short USD" positions

# ============================================================
# Risk caps
# ============================================================
RISK_PCT = Decimal(os.getenv("RISK_PCT", "0.0025"))
MAX_RISK_USD = Decimal(os.getenv("MAX_RISK_USD", "150"))
DAILY_LOSS_LIMIT_USD = Decimal(os.getenv("DAILY_LOSS_LIMIT_USD", "500"))
ALLOW_NONQUOTE_HOME_SIZING = os.getenv("ALLOW_NONQUOTE_HOME_SIZING", "true").lower() == "true"

DB_PATH = os.getenv("DB_PATH", "bot.db")
ALERT_TTL_SECONDS = int(os.getenv("ALERT_TTL_SECONDS", "60"))
PORT = int(os.getenv("PORT", "8080"))

MIN_TRADE_UNITS = int(os.getenv("MIN_TRADE_UNITS", "10"))

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


# ============================================================
# Time helpers
# ============================================================
def _parse_hhmm(s: str) -> Tuple[int, int]:
    s = (s or "").strip()
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM: {s}")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid HH:MM: {s}")
    return hh, mm


def in_window_utc(dt_utc: datetime.datetime, start_hhmm: str, end_hhmm: str) -> bool:
    try:
        sh, sm = _parse_hhmm(start_hhmm)
        eh, em = _parse_hhmm(end_hhmm)
    except Exception:
        return False

    start = dt_utc.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = dt_utc.replace(hour=eh, minute=em, second=0, microsecond=0)

    if end < start:
        return (dt_utc >= start) or (dt_utc <= end)
    return start <= dt_utc <= end


def in_rollover_window(dt_utc: datetime.datetime) -> bool:
    return in_window_utc(dt_utc, ROLLOVER_START_UTC, ROLLOVER_END_UTC)


def in_session_window(dt_utc: datetime.datetime) -> bool:
    return in_window_utc(dt_utc, SESSION_START_UTC, SESSION_END_UTC)


# ============================================================
# DB setup + migration
# ============================================================
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")

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
        CREATE TABLE IF NOT EXISTS trade_state (
            instrument TEXT PRIMARY KEY,
            is_open INTEGER NOT NULL,
            side TEXT,
            units INTEGER,
            entry_price TEXT,
            entry_time_ms INTEGER,
            last_mark_price TEXT,
            unrealized_pl_home TEXT,
            realized_pl_home TEXT,
            last_update_ts INTEGER
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_state (
            day TEXT PRIMARY KEY,
            nav_start TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trailing_state (
            instrument TEXT PRIMARY KEY,
            last_trail_ts INTEGER NOT NULL
        );
    """)

    conn.commit()
    conn.close()


db_init()


# ============================================================
# DB helpers
# ============================================================
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
        kwargs.get("alert_id") or "",
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


def get_trade_state(instrument: str) -> dict:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trade_state WHERE instrument = ?", (instrument,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def upsert_trade_state(
    *,
    instrument: str,
    is_open: bool,
    side: Optional[str],
    units: Optional[int],
    entry_price: Optional[Decimal],
    entry_time_ms: Optional[int],
    last_mark_price: Optional[Decimal],
    unrealized_pl_home: Optional[Decimal],
    realized_pl_home: Optional[Decimal]
):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trade_state (
            instrument, is_open, side, units, entry_price, entry_time_ms,
            last_mark_price, unrealized_pl_home, realized_pl_home, last_update_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument) DO UPDATE SET
            is_open=excluded.is_open,
            side=excluded.side,
            units=excluded.units,
            entry_price=excluded.entry_price,
            entry_time_ms=excluded.entry_time_ms,
            last_mark_price=excluded.last_mark_price,
            unrealized_pl_home=excluded.unrealized_pl_home,
            realized_pl_home=excluded.realized_pl_home,
            last_update_ts=excluded.last_update_ts;
    """, (
        instrument,
        1 if is_open else 0,
        side,
        units,
        (str(entry_price) if entry_price is not None else None),
        entry_time_ms,
        (str(last_mark_price) if last_mark_price is not None else None),
        (str(unrealized_pl_home) if unrealized_pl_home is not None else None),
        (str(realized_pl_home) if realized_pl_home is not None else None),
        int(time.time())
    ))
    conn.commit()
    conn.close()


def get_last_trail_ts(instrument: str) -> int:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT last_trail_ts FROM trailing_state WHERE instrument = ?", (instrument,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return 0
    try:
        return int(row["last_trail_ts"])
    except Exception:
        return 0


def set_last_trail_ts(instrument: str, ts: int) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trailing_state (instrument, last_trail_ts)
        VALUES (?, ?)
        ON CONFLICT(instrument) DO UPDATE SET last_trail_ts=excluded.last_trail_ts;
    """, (instrument, int(ts)))
    conn.commit()
    conn.close()


# ============================================================
# Dedupe
# ============================================================
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


# ============================================================
# OANDA helpers
# ============================================================
def pip_size_for(instrument: str) -> Decimal:
    if instrument.endswith("_JPY"):
        return Decimal("0.01")
    return Decimal("0.0001")


def fmt_price_decimal(x: Decimal, instrument: str) -> str:
    if instrument.endswith("_JPY"):
        places = Decimal("0.001")
    else:
        places = Decimal("0.00001")
    return str(x.quantize(places, rounding=ROUND_DOWN))


def oanda_get_open_positions():
    return requests.get(f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions", headers=HEADERS, timeout=10)


def oanda_get_open_trades(instrument: str):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/trades"
    params = {"state": "OPEN", "instrument": instrument}
    return requests.get(url, headers=HEADERS, params=params, timeout=10)


def oanda_set_trade_orders(trade_id: str, payload: dict):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/trades/{trade_id}/orders"
    return requests.put(url, headers=HEADERS, json=payload, timeout=10)


def get_open_trade_unit_sizes(instrument: str) -> Optional[set]:
    try:
        r = oanda_get_open_trades(instrument)
        if not r.ok:
            return set()
        data = r.json() or {}
        trades = data.get("trades", []) or []
        sizes = set()
        for t in trades:
            try:
                sizes.add(abs(int(t.get("currentUnits", "0"))))
            except Exception:
                pass
        return sizes
    except Exception:
        return None


def fifo_make_units_unique(
    instrument: str,
    desired_units: int,
    *,
    min_units: int,
    max_units: int,
    bump_limit: int = 50
) -> int:
    try:
        desired_units = int(desired_units)
    except Exception:
        desired_units = 1

    min_units = max(1, int(min_units))
    max_units = max(min_units, int(max_units))
    desired_units = max(min_units, min(desired_units, max_units))

    existing = get_open_trade_unit_sizes(instrument)

    if existing is not None and desired_units not in existing:
        return desired_units

    u = desired_units
    for _ in range(max(0, int(bump_limit))):
        offset = random.randint(1, 15)

        u_up = u + offset
        if u_up <= max_units and (existing is None or u_up not in existing):
            return u_up

        u_down = u - offset
        if u_down >= min_units and (existing is None or u_down not in existing):
            return u_down

    return desired_units


def oanda_get_pricing(instrument: str):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing"
    params = {"instruments": instrument}
    return requests.get(url, headers=HEADERS, params=params, timeout=10)


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


def get_price_snapshot(instrument: str) -> Tuple[Decimal, Decimal, Decimal]:
    r = oanda_get_pricing(instrument)
    if not r.ok:
        raise RuntimeError(f"Failed to fetch pricing: HTTP {r.status_code} {r.text}")
    data = r.json()
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError("No pricing data returned")
    p = prices[0]

    bid_s = (p["bids"][0]["price"] if ("bids" in p and p["bids"]) else p.get("closeoutBid"))
    ask_s = (p["asks"][0]["price"] if ("asks" in p and p["asks"]) else p.get("closeoutAsk"))

    bid = Decimal(str(bid_s))
    ask = Decimal(str(ask_s))
    mid = (bid + ask) / Decimal(2)
    return bid, ask, mid


def get_net_and_avg_entry_from_oanda(instrument: str) -> Tuple[int, Optional[str], Optional[Decimal]]:
    r = oanda_get_open_positions()
    if not r.ok:
        raise RuntimeError(f"Failed to fetch open positions: HTTP {r.status_code} {r.text}")
    data = r.json()
    for pos in data.get("positions", []):
        if pos.get("instrument") != instrument:
            continue
        long_units = int(pos.get("long", {}).get("units", "0"))
        short_units = int(pos.get("short", {}).get("units", "0"))
        net = long_units + short_units

        if net > 0:
            side = "buy"
            ap = pos.get("long", {}).get("averagePrice")
        elif net < 0:
            side = "sell"
            ap = pos.get("short", {}).get("averagePrice")
        else:
            side = None
            ap = None

        avg_price = Decimal(str(ap)) if ap is not None else None
        return net, side, avg_price

    return 0, None, None


def get_total_exposure_units() -> int:
    r = oanda_get_open_positions()
    if not r.ok:
        return 0
    data = r.json()
    total = 0
    for pos in data.get("positions", []):
        long_u = int(pos.get("long", {}).get("units", "0"))
        short_u = int(pos.get("short", {}).get("units", "0"))
        total += abs(long_u) + abs(short_u)
    return total


def quote_to_home_mid(quote_ccy: str, home_ccy: str) -> Optional[Decimal]:
    if quote_ccy == home_ccy:
        return Decimal(1)
    if not ALLOW_NONQUOTE_HOME_SIZING:
        return None

    quote_home = f"{quote_ccy}_{home_ccy}"
    home_quote = f"{home_ccy}_{quote_ccy}"

    try:
        r = oanda_get_pricing(quote_home)
        if r.ok and r.json().get("prices"):
            p = r.json()["prices"][0]
            bid_s = p.get("closeoutBid") or p["bids"][0]["price"]
            ask_s = p.get("closeoutAsk") or p["asks"][0]["price"]
            bid = Decimal(str(bid_s))
            ask = Decimal(str(ask_s))
            return (bid + ask) / Decimal(2)
    except Exception:
        pass

    r2 = oanda_get_pricing(home_quote)
    if not r2.ok or not r2.json().get("prices"):
        return None
    p2 = r2.json()["prices"][0]
    bid2_s = p2.get("closeoutBid") or p2["bids"][0]["price"]
    ask2_s = p2.get("closeoutAsk") or p2["asks"][0]["price"]
    bid2 = Decimal(str(bid2_s))
    ask2 = Decimal(str(ask2_s))
    mid2 = (bid2 + ask2) / Decimal(2)  # 1 home = mid2 quote
    return Decimal(1) / mid2           # 1 quote = 1/mid2 home


def compute_unrealized_pl_home(instrument: str, side: str, units: int, entry_price: Decimal, mark_price: Decimal, account_ccy: str) -> Optional[Decimal]:
    _, quote = instrument.split("_")
    u = abs(int(units))
    if u == 0:
        return Decimal(0)
    if side == "buy":
        pl_quote = (mark_price - entry_price) * Decimal(u)
    else:
        pl_quote = (entry_price - mark_price) * Decimal(u)
    conv = quote_to_home_mid(quote, account_ccy)
    if conv is None:
        return None
    return pl_quote * conv


def pips_from_entry(instrument: str, side: str, entry: Decimal, mark: Decimal) -> float:
    pip = pip_size_for(instrument)
    if side == "buy":
        return float((mark - entry) / pip)
    return float((entry - mark) / pip)


def loss_per_unit_home(*, instrument: str, sl_pips: int, account_ccy: str) -> Decimal:
    _, quote = instrument.split("_")
    pip = pip_size_for(instrument)
    sl_dist_quote = pip * Decimal(sl_pips)
    if sl_pips <= 0:
        raise RuntimeError("sl_pips must be > 0 for risk sizing")
    if quote == account_ccy:
        return sl_dist_quote
    conv = quote_to_home_mid(quote, account_ccy)
    if conv is None:
        raise RuntimeError(
            f"Cannot risk-size safely: quote={quote} != account_ccy={account_ccy}. "
            f"Market conversion failed."
        )
    return sl_dist_quote * conv


def compute_units_from_risk(*, instrument: str, sl_pips: int, nav: Decimal, account_ccy: str) -> int:
    loss_unit = loss_per_unit_home(instrument=instrument, sl_pips=sl_pips, account_ccy=account_ccy)
    risk_budget_home = min(nav * RISK_PCT, MAX_RISK_USD, nav)
    units = int((risk_budget_home / loss_unit).to_integral_value(rounding=ROUND_DOWN))
    return max(1, units)


def get_day_start_nav(nav_now: Decimal) -> Decimal:
    day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
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

    # TP disabled by default; if enabled and tp_pips>0, include TP on fill.
    if TAKE_PROFIT_ENABLED and tp_dist is not None:
        order["order"]["takeProfitOnFill"] = {"distance": fmt_price_decimal(tp_dist, instrument)}

    return requests.post(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders",
        json=order,
        headers=HEADERS,
        timeout=10
    )


# ============================================================
# Exposure / correlation helpers
# ============================================================
def _usd_direction_for(instrument: str, net_units: int) -> Optional[str]:
    """
    Returns 'LONG_USD' or 'SHORT_USD' if instrument contains USD and net_units != 0, else None.
    Heuristic: for EUR_USD, net<0 => long USD (short EUR), net>0 => short USD.
              for USD_JPY, net>0 => long USD, net<0 => short USD.
    """
    if net_units == 0:
        return None
    base, quote = instrument.split("_")
    if base == "USD":
        return "LONG_USD" if net_units > 0 else "SHORT_USD"
    if quote == "USD":
        return "LONG_USD" if net_units < 0 else "SHORT_USD"
    return None


def count_usd_basket_positions() -> Tuple[int, int]:
    """
    Counts how many currently-open positions are effectively LONG_USD vs SHORT_USD.
    """
    try:
        r = oanda_get_open_positions()
        if not r.ok:
            return (0, 0)
        data = r.json() or {}
        long_usd = 0
        short_usd = 0
        for pos in data.get("positions", []) or []:
            inst = pos.get("instrument")
            if not inst or "_" not in inst:
                continue
            long_units = int(pos.get("long", {}).get("units", "0"))
            short_units = int(pos.get("short", {}).get("units", "0"))
            net = long_units + short_units
            d = _usd_direction_for(inst, net)
            if d == "LONG_USD":
                long_usd += 1
            elif d == "SHORT_USD":
                short_usd += 1
        return (long_usd, short_usd)
    except Exception:
        return (0, 0)


# ============================================================
# Broker truth sync (and close logging when broker becomes flat)
# ============================================================
def sync_trade_state_from_oanda(
    *,
    instrument: str,
    acct_ccy: Optional[str] = None,
    entry_time_ms_hint: Optional[int] = None,
    fallback_entry_price: Optional[Decimal] = None,
    fallback_units_abs: Optional[int] = None,
    alert_id_for_close_log: Optional[str] = None
) -> None:
    prev = get_trade_state(instrument)
    prev_is_open = bool(prev) and int(prev.get("is_open", 0) or 0) == 1

    preserved_entry_time = None
    if prev_is_open and prev.get("entry_time_ms"):
        try:
            preserved_entry_time = int(prev["entry_time_ms"])
        except Exception:
            preserved_entry_time = None

    try:
        net, side, avg_entry = get_net_and_avg_entry_from_oanda(instrument)
    except Exception:
        net, side, avg_entry = 0, None, None

    if net == 0:
        # If broker is flat but our state thought open, log a "close" event for observability.
        if prev_is_open and alert_id_for_close_log:
            db_record_execution(
                alert_id=alert_id_for_close_log,
                action="close",
                instrument=instrument,
                side=None,
                units=None,
                sl_pips=None,
                tp_pips=None,
                current_net=0,
                projected_net=None,
                spread_pips=None,
                oanda_http=None,
                status="BROKER_FLAT_DETECTED",
                oanda_response=None,
                meta=json.dumps({"note": "Position is flat at broker (likely TP/SL/manual). Logged by sync."})
            )

        upsert_trade_state(
            instrument=instrument,
            is_open=False,
            side=None,
            units=None,
            entry_price=None,
            entry_time_ms=None,
            last_mark_price=None,
            unrealized_pl_home=None,
            realized_pl_home=(Decimal(str(prev["realized_pl_home"])) if prev and prev.get("realized_pl_home") else None),
        )
        return

    entry_price = avg_entry or fallback_entry_price
    if entry_price is None:
        try:
            _, _, mid = get_price_snapshot(instrument)
            entry_price = mid
        except Exception:
            entry_price = None

    last_mark = None
    try:
        bid, ask, _ = get_price_snapshot(instrument)
        if side == "buy":
            last_mark = bid
        elif side == "sell":
            last_mark = ask
        else:
            last_mark = (bid + ask) / Decimal(2)
    except Exception:
        last_mark = None

    units_abs = abs(net) if net is not None else (fallback_units_abs or None)
    entry_time_ms = preserved_entry_time or entry_time_ms_hint or int(time.time() * 1000)

    upl_home = None
    if acct_ccy and entry_price is not None and last_mark is not None and side in ("buy", "sell") and units_abs:
        try:
            upl_home = compute_unrealized_pl_home(instrument, side, int(units_abs), entry_price, last_mark, acct_ccy)
        except Exception:
            upl_home = None

    upsert_trade_state(
        instrument=instrument,
        is_open=True,
        side=side,
        units=int(units_abs) if units_abs is not None else None,
        entry_price=entry_price,
        entry_time_ms=entry_time_ms,
        last_mark_price=last_mark,
        unrealized_pl_home=upl_home,
        realized_pl_home=(Decimal(str(prev["realized_pl_home"])) if prev and prev.get("realized_pl_home") else None),
    )


# ============================================================
# Trailing exit logic (Chandelier stop) -> adjusts STOP LOSS at broker
# ============================================================
def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _compute_chandelier_stop(
    *,
    instrument: str,
    side: str,
    lookback: List[dict],
    atr_val: float,
    bid: Decimal,
    ask: Decimal
) -> Optional[Decimal]:
    """
    Computes a chandelier-style stop price based on last N bars:
      long: stop = highest_high - ATR*k
      short: stop = lowest_low + ATR*k
    Adds break-even tighten when pips >= BE_TRIGGER_PIPS.
    """
    if not lookback or len(lookback) < 5:
        return None
    if atr_val <= 0:
        return None

    n = max(5, min(TRAIL_LOOKBACK_BARS, len(lookback)))
    recent = lookback[-n:]

    highs = []
    lows = []
    for c in recent:
        h = _f(c.get("high"))
        l = _f(c.get("low"))
        if h > 0:
            highs.append(h)
        if l > 0:
            lows.append(l)

    if not highs or not lows:
        return None

    highest_high = Decimal(str(max(highs)))
    lowest_low = Decimal(str(min(lows)))
    atr = Decimal(str(atr_val))
    k = Decimal(str(TRAIL_ATR_MULT))

    if side == "buy":
        stop = highest_high - (atr * k)
        # Ensure stop is not above current bid (would immediately trigger)
        pip = pip_size_for(instrument)
        max_stop = bid - (pip * Decimal("1"))
        if stop > max_stop:
            stop = max_stop
    else:
        stop = lowest_low + (atr * k)
        pip = pip_size_for(instrument)
        min_stop = ask + (pip * Decimal("1"))
        if stop < min_stop:
            stop = min_stop

    return stop


def _tighten_with_breakeven(
    *,
    instrument: str,
    side: str,
    entry_price: Decimal,
    desired_stop: Decimal,
    mark_price: Decimal
) -> Decimal:
    """
    If trade is beyond BE_TRIGGER_PIPS, ensure stop is at least entry +/- BE_OFFSET_PIPS.
    """
    pip = pip_size_for(instrument)
    be_offset = pip * Decimal(str(BE_OFFSET_PIPS))
    be_trigger = float(BE_TRIGGER_PIPS)

    pips_now = pips_from_entry(instrument, side, entry_price, mark_price)

    if pips_now < be_trigger:
        return desired_stop

    if side == "buy":
        be_stop = entry_price + be_offset
        return max(desired_stop, be_stop)
    else:
        be_stop = entry_price - be_offset
        return min(desired_stop, be_stop)


def update_trailing_stop_if_needed(
    *,
    instrument: str,
    features: dict,
    acct_ccy: Optional[str],
    alert_id: str,
    spread_pips_val: float,
    exit_spread_limit: float
) -> None:
    if not TRAIL_ENABLED:
        return

    # Throttle
    now = int(time.time())
    last_ts = get_last_trail_ts(instrument)
    if last_ts and (now - last_ts) < TRAIL_UPDATE_SECONDS:
        return

    # Avoid changing stops during extreme spreads
    if spread_pips_val is not None and exit_spread_limit is not None and float(spread_pips_val) > float(exit_spread_limit):
        return

    ts = get_trade_state(instrument)
    if not ts or int(ts.get("is_open", 0) or 0) != 1:
        return

    side = str(ts.get("side") or "").lower()
    if side not in ("buy", "sell"):
        return

    entry_price = Decimal(str(ts.get("entry_price")))
    try:
        bid, ask, _ = get_price_snapshot(instrument)
    except Exception:
        return

    mark = bid if side == "buy" else ask

    lookback = features.get("lookback", []) or []
    atr_val = _f(features.get("atr", 0.0), 0.0)

    desired = _compute_chandelier_stop(
        instrument=instrument,
        side=side,
        lookback=lookback,
        atr_val=atr_val,
        bid=bid,
        ask=ask
    )
    if desired is None:
        return

    desired = _tighten_with_breakeven(
        instrument=instrument,
        side=side,
        entry_price=entry_price,
        desired_stop=desired,
        mark_price=mark
    )

    # Fetch open trades for this instrument and tighten each trade's SL (FIFO-safe)
    try:
        r = oanda_get_open_trades(instrument)
        if not r.ok:
            return
        data = r.json() or {}
        trades = data.get("trades", []) or []
    except Exception:
        return

    updated_any = False
    for t in trades:
        trade_id = str(t.get("id", "")).strip()
        if not trade_id:
            continue

        # Determine trade side
        try:
            cu = int(t.get("currentUnits", "0"))
        except Exception:
            cu = 0
        trade_side = "buy" if cu > 0 else ("sell" if cu < 0 else None)
        if trade_side != side:
            continue

        # Current SL price if exists
        old_sl = None
        slo = t.get("stopLossOrder") or {}
        if isinstance(slo, dict) and "price" in slo:
            try:
                old_sl = Decimal(str(slo["price"]))
            except Exception:
                old_sl = None

        # Tighten only (never loosen)
        tighten_ok = False
        if old_sl is None:
            tighten_ok = True
        else:
            if side == "buy" and desired > old_sl:
                tighten_ok = True
            if side == "sell" and desired < old_sl:
                tighten_ok = True

        if not tighten_ok:
            continue

        payload = {"stopLoss": {"price": fmt_price_decimal(desired, instrument)}}
        try:
            rr = oanda_set_trade_orders(trade_id, payload)
            if rr.ok:
                updated_any = True
        except Exception:
            pass

    if updated_any:
        set_last_trail_ts(instrument, now)
        db_record_execution(
            alert_id=alert_id,
            action="observe",
            instrument=instrument,
            status="TRAIL_UPDATED",
            spread_pips=spread_pips_val,
            meta=json.dumps({
                "trail": "chandelier",
                "trail_atr_mult": TRAIL_ATR_MULT,
                "trail_lookback_bars": TRAIL_LOOKBACK_BARS,
                "be_trigger_pips": BE_TRIGGER_PIPS,
                "be_offset_pips": BE_OFFSET_PIPS
            })
        )


# ============================================================
# Validation
# ============================================================
def validate_payload(data: dict):
    alert_id = str(data.get("alert_id", "")).strip()
    action = str(data.get("action", "observe")).lower().strip()
    pair = str(data.get("pair", "")).upper().strip()

    if not alert_id:
        return None, "Missing alert_id"
    if action not in {"observe", "open", "close", "close_all"}:
        return None, "Invalid action"

    if action == "close_all":
        return {"alert_id": alert_id, "action": action}, None

    if action in {"observe", "close", "open"}:
        if "_" not in pair:
            return None, "Invalid pair format"
        if ALLOWED_PAIRS and pair not in ALLOWED_PAIRS:
            return None, f"Pair not allowed: {pair}"

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

    side = str(data.get("side", "")).lower().strip()
    if side not in {"buy", "sell"}:
        return None, "Invalid side"
    return {
        "alert_id": alert_id,
        "action": "open",
        "pair": pair,
        "side": side,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "features": features
    }, None


def check_cooldown(instrument: str, cooldown_minutes: int) -> bool:
    if cooldown_minutes <= 0:
        return True
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ts FROM executions
        WHERE instrument = ? AND action = 'open' AND status = 'OK'
        ORDER BY ts DESC LIMIT 1
    """, (instrument,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return True
    last_ts = row["ts"]
    now = int(time.time())
    elapsed_minutes = (now - last_ts) / 60.0
    return elapsed_minutes >= cooldown_minutes


def check_min_spacing(instrument: str, spacing_seconds: int) -> bool:
    if spacing_seconds <= 0:
        return True
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ts FROM executions
        WHERE instrument = ? AND action = 'open' AND status = 'OK'
        ORDER BY ts DESC LIMIT 1
    """, (instrument,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return True
    last_ts = row["ts"]
    return (int(time.time()) - last_ts) >= spacing_seconds


# ============================================================
# Core execution helper
# ============================================================
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
    meta: dict,
    max_spread_pips: float
) -> Tuple[Dict[str, Any], int]:

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # --- ENTRY SESSION FILTER ---
    if SESSION_FILTER_ENABLED and not in_session_window(now_utc):
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_SESSION",
            meta=json.dumps({
                **(meta or {}),
                "session_filter_enabled": True,
                "session_start_utc": SESSION_START_UTC,
                "session_end_utc": SESSION_END_UTC
            })
        )
        return {"status": "RISK_BLOCKED_SESSION", "reason": "OPEN blocked outside session window."}, 200

    # --- SAFETY BLOCK: NO OPENS DURING ROLLOVER WINDOW (UTC) ---
    if ROLLOVER_BLOCK_OPENS and in_rollover_window(now_utc):
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_ROLLOVER",
            meta=json.dumps({
                **(meta or {}),
                "rollover_block_opens": True,
                "rollover_start_utc": ROLLOVER_START_UTC,
                "rollover_end_utc": ROLLOVER_END_UTC
            })
        )
        return {"status": "RISK_BLOCKED_ROLLOVER", "reason": "OPEN blocked during rollover window."}, 200

    # --- SAFETY BLOCK: NO HEDGE/FLIP ENFORCEMENT ---
    if current_net > 0 and side == "sell":
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=None, tp_pips=None,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="SAFETY_BLOCKED_FLIP",
            meta=json.dumps(meta)
        )
        return {"status": "SAFETY_BLOCKED_FLIP", "reason": "Cannot OPEN SELL while Long. Use CLOSE first."}, 200

    if current_net < 0 and side == "buy":
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=None, tp_pips=None,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="SAFETY_BLOCKED_FLIP",
            meta=json.dumps(meta)
        )
        return {"status": "SAFETY_BLOCKED_FLIP", "reason": "Cannot OPEN BUY while Short. Use CLOSE first."}, 200

    # 1) Spread Gate
    current_max_spread = float(max_spread_pips) if max_spread_pips is not None else float(MAX_SPREAD_PIPS)

    if spread_pips_val is None:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=None,
            status="RISK_BLOCKED_PRICE_FETCH_FAIL",
            meta=json.dumps({**meta, "max_spread_pips": current_max_spread})
        )
        return {"status": "RISK_BLOCKED_PRICE_FETCH_FAIL"}, 503

    if spread_pips_val > current_max_spread:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_SPREAD",
            meta=json.dumps({**meta, "max_spread_pips": current_max_spread})
        )
        return {"status": "RISK_BLOCKED_SPREAD", "spread_pips": spread_pips_val, "cap": current_max_spread}, 403

    # 2) Daily Loss Limit
    if DAILY_LOSS_LIMIT_USD > 0:
        day_start_nav = get_day_start_nav(nav)
        if day_start_nav - nav >= DAILY_LOSS_LIMIT_USD:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_DAILY_LOSS_LIMIT",
                meta=json.dumps(meta)
            )
            return {"status": "RISK_BLOCKED_DAILY_LOSS_LIMIT"}, 403

    # 2b) USD basket / correlation cap (prevents stacking the same USD bet)
    long_usd, short_usd = count_usd_basket_positions()
    proposed_signed_units = -1 if side == "sell" else 1
    proposed_net = current_net + proposed_signed_units  # sign only for direction check

    proposed_usd_dir = _usd_direction_for(pair, proposed_net)
    if proposed_usd_dir == "LONG_USD" and (long_usd + (0 if _usd_direction_for(pair, current_net) == "LONG_USD" else 1)) > USD_BASKET_MAX_POS:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_USD_BASKET",
            meta=json.dumps({"usd_dir": "LONG_USD", "long_usd": long_usd, "short_usd": short_usd, "cap": USD_BASKET_MAX_POS})
        )
        return {"status": "RISK_BLOCKED_USD_BASKET", "reason": "Too many LONG_USD positions."}, 403

    if proposed_usd_dir == "SHORT_USD" and (short_usd + (0 if _usd_direction_for(pair, current_net) == "SHORT_USD" else 1)) > USD_BASKET_MAX_POS:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_USD_BASKET",
            meta=json.dumps({"usd_dir": "SHORT_USD", "long_usd": long_usd, "short_usd": short_usd, "cap": USD_BASKET_MAX_POS})
        )
        return {"status": "RISK_BLOCKED_USD_BASKET", "reason": "Too many SHORT_USD positions."}, 403

    # 3) Position sizing (Risk % + GPT scale + Min/Max constraints)
    try:
        units = compute_units_from_risk(instrument=pair, sl_pips=sl_pips, nav=nav, account_ccy=acct_ccy)
    except Exception as e:
        log(f"RISK SIZING ERROR | {pair} | {e}")
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=(0 if not TAKE_PROFIT_ENABLED else tp_pips),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_CALC_ERROR", oanda_http=None, oanda_response=str(e),
            meta=json.dumps(meta)
        )
        return {"status": "RISK_CALC_ERROR", "error": str(e)}, 403

    units_mult = max(0.1, min(float(units_mult), GPT_MAX_UNITS_MULT))
    units = int(units * units_mult)

    units = max(MIN_TRADE_UNITS, units)
    units = min(units, MAX_UNITS)

    # TP disabled if TAKE_PROFIT_ENABLED is false
    if not TAKE_PROFIT_ENABLED:
        tp_pips = 0
    else:
        if tp_pips is not None and tp_pips > 0:
            tp_pips = int(max(10, min(tp_pips, MAX_TP_PIPS)))

    # 3b) FIFO safeguard
    if FIFO_ENFORCE_UNIQUE_UNITS:
        headroom = MAX_NET_UNITS - abs(int(current_net))
        if headroom <= 0:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_MAX_NET_UNITS",
                meta=json.dumps({**(meta or {}), "reason": "no_headroom_for_fifo"})
            )
            return {"status": "RISK_BLOCKED_MAX_NET_UNITS", "reason": "No headroom."}, 403

        existing = get_open_trade_unit_sizes(pair)
        if existing is None:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_FIFO_SIZES_UNKNOWN",
                meta=json.dumps({**(meta or {}), "reason": "open_trades_fetch_failed"})
            )
            return {"status": "RISK_BLOCKED_FIFO_SIZES_UNKNOWN"}, 503

        max_units_fifo = min(MAX_UNITS, max(MIN_TRADE_UNITS, headroom))
        original_units = units

        units = fifo_make_units_unique(
            pair,
            units,
            min_units=MIN_TRADE_UNITS,
            max_units=max_units_fifo,
            bump_limit=FIFO_BUMP_LIMIT
        )

        if units in existing:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=units, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_FIFO_UNIQUE_UNAVAILABLE",
                meta=json.dumps({**(meta or {}), "fifo_units_original": original_units, "fifo_units_final": units})
            )
            return {"status": "RISK_BLOCKED_FIFO_UNIQUE_UNAVAILABLE"}, 403

        if units != original_units:
            meta = {**(meta or {}), "fifo_units_original": original_units, "fifo_units_final": units}

    # 4) Absolute Max Risk Gate
    try:
        loss_unit = loss_per_unit_home(instrument=pair, sl_pips=sl_pips, account_ccy=acct_ccy)
        expected_loss = loss_unit * Decimal(units)
        max_loss = min(nav, MAX_RISK_USD)

        if expected_loss > max_loss:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=units, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_MAX_RISK_USD",
                meta=json.dumps(meta)
            )
            return {"status": "RISK_BLOCKED_MAX_RISK_USD", "expected_loss": str(expected_loss), "cap": str(max_loss)}, 403
    except Exception as e:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_CALC_ERROR", oanda_http=None, oanda_response=str(e),
            meta=json.dumps(meta)
        )
        return {"status": "RISK_CALC_ERROR", "error": str(e)}, 403

    # 5) Net exposure gate (Per Pair)
    signed_units = -units if side == "sell" else units
    projected_net = current_net + signed_units

    if abs(projected_net) > MAX_NET_UNITS:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_MAX_NET_UNITS",
            meta=json.dumps(meta)
        )
        return {"status": "RISK_BLOCKED_MAX_NET_UNITS", "current_net": current_net, "projected_net": projected_net, "cap": MAX_NET_UNITS}, 403

    # 5b) Global Exposure Gate (All Pairs)
    try:
        current_total_exposure = get_total_exposure_units()
        if current_total_exposure + units > GLOBAL_MAX_UNITS:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=units, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_GLOBAL_MAX_UNITS",
                meta=json.dumps({**(meta or {}), "global_exposure": current_total_exposure, "cap": GLOBAL_MAX_UNITS})
            )
            return {"status": "RISK_BLOCKED_GLOBAL_MAX_UNITS", "current_total": current_total_exposure, "cap": GLOBAL_MAX_UNITS}, 403
    except Exception as e:
        log(f"GLOBAL EXPOSURE CHECK FAIL | {e}")
        return {"status": "RISK_CHECK_FAIL", "error": str(e)}, 503

    # 6) Execute Order
    try:
        r = place_market_order(instrument=pair, signed_units=signed_units, sl_pips=sl_pips, tp_pips=tp_pips)
        body_text = r.text
        resp_json = {}
        try:
            if r.headers.get("Content-Type", "").startswith("application/json"):
                resp_json = r.json()
        except Exception:
            resp_json = {}

        filled = isinstance(resp_json, dict) and ("orderFillTransaction" in resp_json)
        cancelled = isinstance(resp_json, dict) and ("orderCancelTransaction" in resp_json)
        rejected = isinstance(resp_json, dict) and ("orderRejectTransaction" in resp_json)

        cancel_reason = None
        reject_reason = None

        if cancelled:
            octx = resp_json.get("orderCancelTransaction", {})
            if isinstance(octx, dict):
                cancel_reason = octx.get("reason") or octx.get("cancelReason")

        if rejected:
            ortx = resp_json.get("orderRejectTransaction", {})
            if isinstance(ortx, dict):
                reject_reason = ortx.get("rejectReason") or ortx.get("reason")

        if bool(r.ok) and filled and not cancelled and not rejected:
            status_txt = "OK"
        else:
            if rejected:
                status_txt = "OANDA_REJECTED"
            elif cancelled:
                status_txt = "OANDA_CANCELLED"
            else:
                status_txt = "OANDA_ERROR"

        extra = ""
        if status_txt == "OANDA_CANCELLED" and cancel_reason:
            extra = f" | cancel_reason={cancel_reason}"
        elif status_txt == "OANDA_REJECTED" and reject_reason:
            extra = f" | reject_reason={reject_reason}"

        log(
            f"OPEN | alert_id={alert_id} | {pair} {side} {signed_units} | "
            f"net={current_net}->{projected_net} | SL={sl_pips} TP={tp_pips} | "
            f"spread={spread_pips_val} | HTTP {r.status_code} | {status_txt}{extra}"
        )

        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            oanda_http=r.status_code, status=status_txt,
            oanda_response=body_text,
            meta=json.dumps({
                **meta,
                "fill_detected": filled,
                "cancel_detected": cancelled,
                "reject_detected": rejected,
                "oanda_cancel_reason": cancel_reason,
                "oanda_reject_reason": reject_reason,
                "take_profit_enabled": bool(TAKE_PROFIT_ENABLED),
                "trail_enabled": bool(TRAIL_ENABLED),
            })
        )

        # Sync to broker truth
        if bool(r.ok):
            sync_trade_state_from_oanda(
                instrument=pair,
                acct_ccy=acct_ccy,
                entry_time_ms_hint=int(time.time() * 1000),
            )

        return {
            "status": status_txt,
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
            "oanda_cancel_reason": cancel_reason,
            "oanda_reject_reason": reject_reason,
            "response": (resp_json if resp_json else {"raw": body_text})
        }, (200 if status_txt == "OK" else 502)

    except requests.RequestException as e:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="NETWORK_ERROR", oanda_http=None, oanda_response=repr(e),
            meta=json.dumps(meta)
        )
        return {"status": "NETWORK_ERROR", "error": str(e)}, 502


# ============================================================
# Routes
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mode": MODE,
        "pairs": sorted(list(ALLOWED_PAIRS)),
        "session_filter_enabled": SESSION_FILTER_ENABLED,
        "session_start_utc": SESSION_START_UTC,
        "session_end_utc": SESSION_END_UTC,
        "trail_enabled": TRAIL_ENABLED,
        "take_profit_enabled": TAKE_PROFIT_ENABLED
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    key = request.args.get("key", "")
    if key != WEBHOOK_KEY:
        log("AUTH BLOCKED")
        return {"status": "UNAUTHORIZED"}, 401

    if not TRADING_ENABLED:
        return {"status": "DISABLED"}, 200

    data = request.get_json(silent=True) or {}
    payload, err = validate_payload(data)
    if err:
        log(f"BAD REQUEST | {err}")
        return {"status": "BAD_REQUEST", "error": err}, 400

    alert_id = payload["alert_id"]
    if seen_before(alert_id):
        log(f"DUPLICATE IGNORED | alert_id={alert_id}")
        db_record_execution(
            alert_id=alert_id,
            action=payload["action"],
            instrument=payload.get("pair", "N/A"),
            status="DUPLICATE_IGNORED",
            meta=None
        )
        return {"status": "DUPLICATE_IGNORED"}, 200

    action = payload["action"]

    if action == "close_all":
        try:
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
                db_record_execution(
                    alert_id=alert_id, action="close", instrument=inst,
                    oanda_http=rc.status_code, status=("OK" if rc.ok else "OANDA_ERROR"),
                    oanda_response=str(results[-1]["response"]), meta=None
                )
                sync_trade_state_from_oanda(instrument=inst, alert_id_for_close_log=alert_id)
            log(f"CLOSE_ALL | alert_id={alert_id} | closed={len(results)}")
            return {"status": "OK", "results": results}, 200
        except Exception as e:
            db_record_execution(alert_id=alert_id, action="close_all", instrument="ALL", status="ERROR", oanda_response=repr(e), meta=None)
            return {"status": "ERROR", "error": str(e)}, 502

    pair = payload["pair"]

    if action == "close":
        try:
            r = close_position(pair)
            log(f"CLOSE | alert_id={alert_id} | {pair} | HTTP {r.status_code}")
            sync_trade_state_from_oanda(instrument=pair, alert_id_for_close_log=alert_id)
            status = "OK" if r.ok else "OANDA_ERROR"
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair,
                oanda_http=r.status_code, status=status, oanda_response=r.text, meta=None
            )
            return {"status": status, "http_status": r.status_code}, (200 if r.ok else 502)
        except requests.RequestException as e:
            db_record_execution(alert_id=alert_id, action="close", instrument=pair, status="NETWORK_ERROR", oanda_response=repr(e), meta=None)
            return {"status": "NETWORK_ERROR"}, 502

    # ============================================================
    # OBSERVE
    # ============================================================
    features = payload.get("features", {}) or {}
    hint_sl = payload.get("sl_pips", SL_PIPS_DEFAULT)
    hint_tp = payload.get("tp_pips", TP_PIPS_DEFAULT)

    # Stale alert guard
    if MAX_ALERT_AGE_SECONDS > 0:
        tv_time_ms = features.get("time_ms")
        if tv_time_ms is not None:
            try:
                age_sec = (int(time.time() * 1000) - int(tv_time_ms)) / 1000.0
                if age_sec > MAX_ALERT_AGE_SECONDS:
                    db_record_execution(
                        alert_id=alert_id, action="observe", instrument=pair,
                        current_net=None, spread_pips=None,
                        status="STALE_ALERT_IGNORED",
                        meta=json.dumps({"age_sec": age_sec, "max_age": MAX_ALERT_AGE_SECONDS})
                    )
                    return {"status": "STALE_ALERT_IGNORED", "age_sec": age_sec}, 200
            except Exception:
                log(f"STALE_GUARD_PARSE_FAIL | alert_id={alert_id}")

    # Fetch state (net + avg entry)
    try:
        current_net, _, _ = get_net_and_avg_entry_from_oanda(pair)
    except Exception as e:
        log(f"STATE_FETCH_FAIL | {repr(e)}")
        return {"status": "STATE_FETCH_FAIL", "error": str(e)}, 503

    # If broker is flat but our stored state says open, sync + log close.
    prev_ts = get_trade_state(pair)
    prev_is_open = bool(prev_ts) and int(prev_ts.get("is_open", 0) or 0) == 1
    if current_net == 0 and prev_is_open:
        sync_trade_state_from_oanda(instrument=pair, alert_id_for_close_log=alert_id)

    # Lazy load NAV/currency (only needed if trade open)
    nav = None
    acct_ccy = None
    if current_net != 0:
        try:
            nav, acct_ccy = get_account_nav_and_currency()
        except Exception:
            pass

    # Compute analysis flags
    dist_to_ema = _f(features.get("dist_to_ema_200", 0))
    adx = _f(features.get("adx", 0))

    # Overextension logic: tighter than your previous build to avoid late chasing
    # JPY pairs: 70 pips leash; others: 40 pips leash
    overextended_threshold = 70 if "JPY" in pair else 50
    is_overextended = False
    if abs(dist_to_ema) > overextended_threshold:
        is_overextended = True
    elif abs(dist_to_ema) > (overextended_threshold * 0.8) and adx < 20:
        is_overextended = True

    lookback = features.get("lookback", [])
    trend_bias = "mixed"
    # === MODIFIED LOOKBACK LOGIC (10 BARS) ===
    if lookback and len(lookback) >= 10:
        recent = lookback[-10:]  # Look at last 10 candles
        up_count = sum(1 for c in recent if _f(c.get("close")) > _f(c.get("open")))
        if up_count >= 7:        # 70% Green = Bullish
            trend_bias = "bullish"
        elif up_count <= 3:      # 70% Red (<=3 Green) = Bearish
            trend_bias = "bearish"
    # Fallback for shorter history if needed (optional)
    elif lookback and len(lookback) >= 5:
        recent = lookback[-5:]
        up_count = sum(1 for c in recent if _f(c.get("close")) > _f(c.get("open")))
        if up_count >= 4:
            trend_bias = "bullish"
        elif up_count <= 1:
            trend_bias = "bearish"

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    in_roll = in_rollover_window(now_utc)
    in_sess = in_session_window(now_utc)

    analysis = {
        "is_overextended": is_overextended,
        "trend_bias": trend_bias,
        "dist_to_ema_200": dist_to_ema,
        "in_rollover_window": bool(in_roll),
        "in_session_window": bool(in_sess)
    }

    # Pricing snapshot
    spread_pips_val = None
    bid = ask = mid = None
    try:
        bid, ask, mid = get_price_snapshot(pair)
        spread_pips_val = float((ask - bid) / pip_size_for(pair))
    except Exception:
        spread_pips_val = None

    # Dynamic per-pair spread limits
    pair_spread_limits = {
        "EUR_USD": 2.5,
        "GBP_USD": 4.0,
        "AUD_USD": 2.5,
        "USD_JPY": 3.0,
        "EUR_GBP": 2.5,
    }
    dynamic_spread_limit = float(pair_spread_limits.get(pair, MAX_SPREAD_PIPS))

    # Derived EXIT limit
    exit_spread_limit = float(dynamic_spread_limit) * float(EXIT_SPREAD_MULT) + float(EXIT_SPREAD_MIN_ADD)
    if exit_spread_limit < dynamic_spread_limit:
        exit_spread_limit = dynamic_spread_limit

    analysis["max_spread_pips_entry"] = dynamic_spread_limit
    analysis["max_spread_pips_exit"] = exit_spread_limit
    analysis["spread_exit_caution"] = bool(current_net != 0 and spread_pips_val is not None and spread_pips_val > exit_spread_limit)

    if spread_pips_val is None:
        db_record_execution(
            alert_id=alert_id, action="observe", instrument=pair,
            current_net=current_net, spread_pips=None,
            status="RULE_BLOCKED_SPREAD_UNKNOWN", meta=None
        )
        return {"status": "RULE_BLOCKED_SPREAD_UNKNOWN"}, 200

    # ENTRY BLOCKS (only when flat)
    if current_net == 0:
        # Rollover block
        if ROLLOVER_BLOCK_OPENS and in_roll:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_ROLLOVER_OPEN",
                meta=json.dumps({
                    "rollover_start_utc": ROLLOVER_START_UTC,
                    "rollover_end_utc": ROLLOVER_END_UTC,
                    "spread_pips": spread_pips_val
                })
            )
            return {"status": "RULE_BLOCKED_ROLLOVER_OPEN"}, 200

        # Session filter block
        if SESSION_FILTER_ENABLED and not in_sess:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_SESSION_OPEN",
                meta=json.dumps({"session_start_utc": SESSION_START_UTC, "session_end_utc": SESSION_END_UTC})
            )
            return {"status": "RULE_BLOCKED_SESSION_OPEN"}, 200

        # Spread gate
        if spread_pips_val > dynamic_spread_limit:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_SPREAD",
                meta=json.dumps({"limit": dynamic_spread_limit})
            )
            return {"status": "RULE_BLOCKED_SPREAD"}, 200

        # Chop gate
        ema_sep = _f(features.get("ema_sep_pips", 0))
        required_sep = MIN_EMA_SEP_PIPS * (1.2 if "JPY" in pair else 1.0)
        if ema_sep < required_sep:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_CHOP",
                meta=json.dumps({"ema_sep": ema_sep, "required": required_sep})
            )
            return {"status": "RULE_BLOCKED_CHOP", "ema_sep": ema_sep}, 200

        # Overextended block
        if is_overextended:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_OVEREXTENDED",
                meta=json.dumps(analysis)
            )
            return {"status": "RULE_BLOCKED_OVEREXTENDED"}, 200

    # If position is open: sync state and compute UPL pips (also logs broker-flat closes)
    upl_pips = 0.0
    if current_net != 0:
        try:
            sync_trade_state_from_oanda(instrument=pair, acct_ccy=acct_ccy, alert_id_for_close_log=alert_id)
        except Exception:
            pass

        ts_data = get_trade_state(pair)
        if ts_data.get("entry_price") and ts_data.get("side") and bid is not None and ask is not None:
            entry = Decimal(str(ts_data["entry_price"]))
            side_ts = str(ts_data["side"]).lower()
            mark = bid if side_ts == "buy" else ask
            upl_pips = pips_from_entry(pair, side_ts, entry, mark)

        # Trailing stop update (Chandelier)
        try:
            update_trailing_stop_if_needed(
                instrument=pair,
                features=features,
                acct_ccy=acct_ccy,
                alert_id=alert_id,
                spread_pips_val=float(spread_pips_val),
                exit_spread_limit=float(exit_spread_limit)
            )
        except Exception:
            pass

    # Degradation exit (profit-protect + stall) — optional
    if DEGRADE_EXIT_ENABLED and current_net != 0:
        ts2 = get_trade_state(pair)
        if ts2 and int(ts2.get("is_open", 0) or 0) == 1 and ts2.get("entry_price") and ts2.get("entry_time_ms"):
            try:
                entry = Decimal(str(ts2["entry_price"]))
                entry_time_ms = int(ts2["entry_time_ms"])
                now_ms = int(time.time() * 1000)
                held_min = (now_ms - entry_time_ms) / 60000.0

                side_ts = str(ts2.get("side") or ("buy" if current_net > 0 else "sell")).lower()
                if side_ts not in ("buy", "sell"):
                    side_ts = "buy" if current_net > 0 else "sell"

                bid2, ask2, _ = get_price_snapshot(pair)
                mark2 = bid2 if side_ts == "buy" else ask2

                ema_sep_now = _f(features.get("ema_sep_pips", 0))
                adx_now = _f(features.get("adx", 0))
                trend_bias_now = analysis.get("trend_bias", "mixed")
                degrade_sep = DEGRADE_EMA_SEP_PIPS * (1.2 if "JPY" in pair else 1.0)

                is_degraded = (
                    adx_now <= DEGRADE_ADX_MAX and
                    ema_sep_now <= degrade_sep and
                    trend_bias_now == "mixed"
                )

                pips_now = pips_from_entry(pair, side_ts, entry, mark2)

                stall_exit = (
                    held_min >= DEGRADE_MIN_MINUTES and
                    abs(pips_now) <= DEGRADE_STALL_PIPS and
                    is_degraded
                )

                # Pair-specific profit-protect pips (USD_JPY uses slightly higher default)
                pp_pips = DEGRADE_PROFIT_PROTECT_PIPS_DEFAULT
                if pair == "USD_JPY":
                    pp_pips = DEGRADE_PROFIT_PROTECT_PIPS_USDJPY

                profit_protect_exit = (
                    held_min >= DEGRADE_PROFIT_PROTECT_MIN_MINUTES and
                    pips_now >= pp_pips and
                    is_degraded
                )

                if stall_exit or profit_protect_exit:
                    reason = (
                        f"RULE_DEGRADATION_EXIT: held={held_min:.1f}m, "
                        f"pips={pips_now:.1f}, adx={adx_now:.1f}, "
                        f"ema_sep={ema_sep_now:.2f} (req<={degrade_sep:.2f}), "
                        f"trend_bias={trend_bias_now}, "
                        f"mode={'STALL' if stall_exit else 'PROFIT_PROTECT'}"
                    )[:1200]

                    # Spread protection (avoid closing into horrible spreads unless emergency)
                    spread_block = float(spread_pips_val) > float(exit_spread_limit)
                    if spread_block:
                        db_record_execution(
                            alert_id=alert_id,
                            action="observe",
                            instrument=pair,
                            current_net=current_net,
                            spread_pips=spread_pips_val,
                            status="RULE_BLOCKED_EXIT_SPREAD_DEGRADE",
                            meta=json.dumps({
                                "blocked_by": "EXIT_SPREAD",
                                "rule": "DEGRADATION_EXIT",
                                "reason": reason,
                                "spread_pips": float(spread_pips_val),
                                "exit_spread_limit": float(exit_spread_limit)
                            }),
                        )
                        return {"status": "RULE_BLOCKED_EXIT_SPREAD_DEGRADE", "reason": "Degradation exit blocked due to wide spread."}, 200

                    r = close_position(pair)
                    sync_trade_state_from_oanda(instrument=pair, acct_ccy=acct_ccy, alert_id_for_close_log=alert_id)
                    status = "OK" if r.ok else "OANDA_ERROR"
                    db_record_execution(
                        alert_id=alert_id,
                        action="close",
                        instrument=pair,
                        status=status,
                        oanda_http=r.status_code,
                        oanda_response=r.text,
                        spread_pips=spread_pips_val,
                        meta=json.dumps({"rule": "DEGRADATION_EXIT", "reason": reason})
                    )
                    return {"status": status, "action": "close", "reason": reason}, (200 if r.ok else 502)

            except Exception as e:
                log(f"DEGRADE_EXIT_FAIL | {pair} | {repr(e)}")

    # Build GPT context and decide
    gpt_ctx = {
        "pair": pair,
        "features": features,
        "analysis": analysis,
        "hint": {"sl_pips": hint_sl, "tp_pips": hint_tp},
        "limits": {
            "min_sl_pips": MIN_SL_PIPS,
            "max_sl_pips": MAX_SL_PIPS,
            "max_tp_pips": MAX_TP_PIPS,
            "max_units": MAX_UNITS,
            "max_net_units": MAX_NET_UNITS,
            "max_spread_pips_entry": dynamic_spread_limit,
            "max_spread_pips_exit": exit_spread_limit,
            "gpt_max_units_mult": GPT_MAX_UNITS_MULT,
            "rollover_block_opens": bool(ROLLOVER_BLOCK_OPENS),
            "rollover_start_utc": ROLLOVER_START_UTC,
            "rollover_end_utc": ROLLOVER_END_UTC,
            "session_filter_enabled": bool(SESSION_FILTER_ENABLED),
            "session_start_utc": SESSION_START_UTC,
            "session_end_utc": SESSION_END_UTC,
            "take_profit_enabled": bool(TAKE_PROFIT_ENABLED),
            "trail_enabled": bool(TRAIL_ENABLED),
        },
        "state": {
            "spread_pips": spread_pips_val,
            "current_net": current_net,
            "upl_pips": round(float(upl_pips), 1),
            "in_rollover_window": bool(in_roll),
            "in_session_window": bool(in_sess),
        },
        "risk": {
            "risk_pct": str(RISK_PCT),
            "max_risk_usd": str(MAX_RISK_USD)
        }
    }

    gpt_decision = gpt_decide_trade(gpt_ctx)

    if gpt_decision["action"] == "HOLD" or (gpt_decision["action"] == "OPEN" and gpt_decision["confidence"] < GPT_MIN_CONFIDENCE):
        db_record_execution(
            alert_id=alert_id, action="observe", instrument=pair,
            side=None, units=None, sl_pips=hint_sl, tp_pips=(0 if not TAKE_PROFIT_ENABLED else hint_tp),
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status=f"GPT_{gpt_decision['action']}", oanda_http=None, oanda_response=None,
            meta=json.dumps(gpt_decision)
        )
        return {"status": f"GPT_{gpt_decision['action']}", "decision": gpt_decision}, 200

    if gpt_decision["action"] == "CLOSE":
        # Close onlyI: allow emergency invalidation closes; block non-emergency if spread too wide.
        try:
            spread_block = float(spread_pips_val) > float(exit_spread_limit)
            dist = float(analysis.get("dist_to_ema_200", 0.0) or 0.0)
            macro_flip = ((current_net > 0 and dist < 0) or (current_net < 0 and dist > 0))

            reason_txt = str(gpt_decision.get("reason", "")).lower()
            keyword_emergency = any(k in reason_txt for k in [
                "ema 200", "ema200", "200 ema",
                "market structure", "structure break", "break of structure", "bos",
                "invalidation", "trend reversed", "reversal"
            ])
            allow_emergency_close = bool(macro_flip or keyword_emergency)

            if spread_block and not allow_emergency_close:
                db_record_execution(
                    alert_id=alert_id,
                    action="observe",
                    instrument=pair,
                    current_net=current_net,
                    spread_pips=spread_pips_val,
                    status="RULE_BLOCKED_EXIT_SPREAD",
                    meta=json.dumps({
                        "blocked_by": "EXIT_SPREAD",
                        "spread_pips": float(spread_pips_val),
                        "exit_spread_limit": float(exit_spread_limit),
                        "gpt_decision": gpt_decision,
                        "note": "Close blocked due to wide spreads; not an emergency invalidation."
                    })
                )
                return {"status": "RULE_BLOCKED_EXIT_SPREAD", "reason": "Exit blocked due to wide spread (non-emergency)."}, 200

            r = close_position(pair)
            sync_trade_state_from_oanda(instrument=pair, acct_ccy=acct_ccy, alert_id_for_close_log=alert_id)
            status = "OK" if r.ok else "OANDA_ERROR"
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair,
                side=None, units=None, sl_pips=None, tp_pips=None,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                oanda_http=r.status_code, status=status, oanda_response=r.text,
                meta=json.dumps(gpt_decision)
            )
            return {"status": status, "action": "close"}, (200 if r.ok else 502)

        except Exception as e:
            db_record_execution(
                alert_id=alert_id,
                action="close",
                instrument=pair,
                status="ERROR",
                oanda_response=str(e),
                meta=json.dumps(gpt_decision)
            )
            return {"status": "ERROR", "error": str(e)}, 502

    if gpt_decision["action"] == "OPEN":
        cooldown_min = int(os.getenv("PYRAMID_COOLDOWN_MINUTES", "60"))
        if current_net != 0 and not check_cooldown(pair, cooldown_min):
            db_record_execution(
                alert_id=alert_id,
                action="observe",
                instrument=pair,
                current_net=current_net,
                spread_pips=spread_pips_val,
                status="COOLDOWN_ACTIVE",
                meta=json.dumps({"cooldown_minutes": cooldown_min, "gpt_proposed": gpt_decision})
            )
            return {"status": "COOLDOWN_ACTIVE"}, 200

        min_spacing_s = int(os.getenv("MIN_ALERT_SPACING_SECONDS", "300"))
        if not check_min_spacing(pair, min_spacing_s):
            db_record_execution(
                alert_id=alert_id,
                action="observe",
                instrument=pair,
                current_net=current_net,
                spread_pips=spread_pips_val,
                status="RULE_BLOCKED_MIN_SPACING",
                meta=json.dumps({"min_spacing_seconds": min_spacing_s})
            )
            return {"status": "RULE_BLOCKED_MIN_SPACING"}, 200

        side = gpt_decision["side"]
        if side not in ("buy", "sell"):
            db_record_execution(alert_id=alert_id, action="observe", instrument=pair, status="GPT_INVALID_SIDE", meta=json.dumps(gpt_decision))
            return {"status": "GPT_INVALID_SIDE"}, 200

        # Dynamic SL (ATR-based). TP disabled by default (trailing manages exits).
        atr_val = _f(features.get("atr", 0))
        if atr_val > 0:
            pip_unit = float(pip_size_for(pair))
            atr_in_pips = atr_val / pip_unit
            sl_multiplier = float(os.getenv("SL_ATR_MULTIPLIER", str(SL_ATR_MULTIPLIER)))
            sl_pips = max(10, int(atr_in_pips * sl_multiplier))
        else:
            sl_pips = gpt_decision["sl_pips"] if gpt_decision["sl_pips"] is not None else hint_sl

        sl_pips = int(max(MIN_SL_PIPS, min(sl_pips, MAX_SL_PIPS)))
        tp_pips = 0 if not TAKE_PROFIT_ENABLED else (gpt_decision["tp_pips"] or hint_tp)

        meta = {"gpt": gpt_decision, "features": features}

        if nav is None or acct_ccy is None:
            try:
                nav, acct_ccy = get_account_nav_and_currency()
            except Exception as e:
                log(f"NAV FETCH FAIL | {e}")
                db_record_execution(
                    alert_id=alert_id, action="open", instrument=pair, side=side,
                    status="NAV_FETCH_FAIL", meta=json.dumps(meta)
                )
                return {"status": "NAV_FETCH_FAIL", "error": str(e)}, 503

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
            meta=meta,
            max_spread_pips=dynamic_spread_limit
        )
        return resp, status

    return {"status": "BAD_REQUEST"}, 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
