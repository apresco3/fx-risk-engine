from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import datetime
import os
import time
import sqlite3
import json
import random  # Added for FIFO randomization
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Optional, Tuple, Dict, Any

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
# Ensure you use a valid model name (e.g., gpt-4o or gpt-3.5-turbo)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o") 
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

    # --- [FIXED] Updated Strategic Prompt with Exhaustion Filter ---
    instructions = (
        "You are an FX execution gatekeeper. Output ONLY the decide_trade tool call.\n\n"
        "Context:\n"
        "- Analysis flags (is_overextended, trend_bias) are provided.\n"
        "- 'dist_to_ema_200' tells you the macro trend. Positive = Bullish, Negative = Bearish.\n\n"
        
        "Strategy Rules:\n"
        "1. MACRO FILTER (Crucial): \n"
        "   - If dist_to_ema_200 > 0: ONLY OPEN BUY (or HOLD).\n"
        "   - If dist_to_ema_200 < 0: ONLY OPEN SELL (or HOLD).\n"
        "   - Do not fight the 200 EMA.\n\n"

        "2. ENTRY (Pullback/Momentum):\n"
        "   - OPEN if features.adx > 20 AND features.adx < 40.\n"
        "   - BLOCK ENTRY if features.adx >= 40 (Trend Likely Exhausted/Hyperextended).\n"
        "   - trend_bias must align with dist_to_ema_200.\n"
        "   - Avoid opening if analysis.is_overextended is TRUE (Wait for pullback).\n\n"

        "3. EXIT (Let Winners Run):\n"
        "   - DO NOT RECOMMEND CLOSING just because of small profits.\n"
        "   - ONLY CLOSE if: Trend has fully reversed (e.g., price crosses EMA 200) OR Market Structure break detected.\n"
        "   - Otherwise, return ACTION: HOLD and let the TP/SL manage the trade.\n\n"

        "Output Logic:\n"
        "- If existing position matches Macro Trend: ACTION: HOLD.\n"
        "- If no position AND signals align: ACTION: OPEN.\n"
        "- If existing position fights Macro Trend: ACTION: CLOSE."
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

MAX_UNITS = int(os.getenv("MAX_UNITS", "1000"))
MAX_NET_UNITS = int(os.getenv("MAX_NET_UNITS", "2000"))
GLOBAL_MAX_UNITS = int(os.getenv("GLOBAL_MAX_UNITS", "20000"))

# Updated for M15 candles (15m * 60s + buffer)
MAX_ALERT_AGE_SECONDS = int(os.getenv("MAX_ALERT_AGE_SECONDS", "1200"))

SL_PIPS_DEFAULT = int(os.getenv("SL_PIPS", "30"))
TP_PIPS_DEFAULT = int(os.getenv("TP_PIPS", "60"))
MAX_SL_PIPS = int(os.getenv("MAX_SL_PIPS", "150"))
MAX_TP_PIPS = int(os.getenv("MAX_TP_PIPS", "300"))

# Updated ATR multipliers for M15 volatility
TP_ATR_MULTIPLIER = float(os.getenv("TP_ATR_MULTIPLIER", "3.0"))
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", "2.5"))
MIN_SL_PIPS = int(os.getenv("MIN_SL_PIPS", "15"))

MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "2.5"))
MAX_SPREAD_PIPS_XAU = float(os.getenv("MAX_SPREAD_PIPS_XAU", "60.0"))
MIN_EMA_SEP_PIPS = float(os.getenv("MIN_EMA_SEP_PIPS", "1.5"))

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

def in_rollover_window(dt_utc: datetime.datetime) -> bool:
    try:
        sh, sm = _parse_hhmm(ROLLOVER_START_UTC)
        eh, em = _parse_hhmm(ROLLOVER_END_UTC)
    except Exception:
        return False

    start = dt_utc.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = dt_utc.replace(hour=eh, minute=em, second=0, microsecond=0)

    if end < start:
        return (dt_utc >= start) or (dt_utc <= end)
    return start <= dt_utc <= end

# ============================================================
# Trade Degradation Exit (stall + profit-protect)
# ============================================================
DEGRADE_EXIT_ENABLED = os.getenv("DEGRADE_EXIT_ENABLED", "true").lower() == "true"
DEGRADE_MIN_MINUTES = int(os.getenv("DEGRADE_MIN_MINUTES", "30"))
DEGRADE_STALL_PIPS = float(os.getenv("DEGRADE_STALL_PIPS", "2.0"))
DEGRADE_ADX_MAX = float(os.getenv("DEGRADE_ADX_MAX", "18.0"))
DEGRADE_EMA_SEP_PIPS = float(os.getenv("DEGRADE_EMA_SEP_PIPS", "3.0"))
DEGRADE_PROFIT_PROTECT_MIN_MINUTES = int(os.getenv("DEGRADE_PROFIT_PROTECT_MIN_MINUTES", "10"))
DEGRADE_PROFIT_PROTECT_PIPS = float(os.getenv("DEGRADE_PROFIT_PROTECT_PIPS", "8.0"))

RISK_PCT = Decimal(os.getenv("RISK_PCT", "0.0025"))
MAX_RISK_USD = Decimal(os.getenv("MAX_RISK_USD", "100"))
DAILY_LOSS_LIMIT_USD = Decimal(os.getenv("DAILY_LOSS_LIMIT_USD", "0"))
ALLOW_NONQUOTE_HOME_SIZING = os.getenv("ALLOW_NONQUOTE_HOME_SIZING", "true").lower() == "true"

DB_PATH = os.getenv("DB_PATH", "bot.db")
ALERT_TTL_SECONDS = int(os.getenv("ALERT_TTL_SECONDS", "60"))
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
    # Enable Write-Ahead Logging for concurrency safety
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
    cur.execute("PRAGMA table_info(executions);")
    cols = {row["name"] for row in cur.fetchall()}
    if "meta" not in cols:
        cur.execute("ALTER TABLE executions ADD COLUMN meta TEXT;")
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
    if "XAU" in instrument:
        return Decimal("0.1") # Check your broker's spec for Gold
    if instrument.endswith("_JPY"):
        return Decimal("0.01")
    return Decimal("0.0001")

def fmt_price_decimal(x: Decimal, instrument: str) -> str:
    if "XAU" in instrument:
        places = Decimal("0.01")
    elif instrument.endswith("_JPY"):
        places = Decimal("0.001")
    else:
        places = Decimal("0.00001")
    return str(x.quantize(places, rounding=ROUND_DOWN))

def oanda_get_open_positions():
    return requests.get(f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions", headers=HEADERS, timeout=10)

def oanda_get_open_trades(instrument: str):
    # Used for FIFO unique-unit sizing
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/trades"
    params = {"state": "OPEN", "instrument": instrument}
    return requests.get(url, headers=HEADERS, params=params, timeout=10)

def get_open_trade_unit_sizes(instrument: str) -> set:
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
    
    # If the exact size doesn't exist, we are good
    if existing is not None and desired_units not in existing:
        return desired_units

    # --- [FIXED] Randomization Strategy for FIFO ---
    # Instead of purely linear bumping, try random offsets to avoid collision patterns
    # Try up to 'bump_limit' attempts to find a unique slot
    u = desired_units
    for _ in range(max(0, int(bump_limit))):
        offset = random.randint(1, 15) # Random jump between 1 and 15
        
        # Try adding
        u_up = u + offset
        if u_up <= max_units and (existing is None or u_up not in existing):
            return u_up
            
        # Try subtracting
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

    if "bids" in p and p["bids"]:
        bid_s = p["bids"][0]["price"]
    else:
        bid_s = p.get("closeoutBid")

    if "asks" in p and p["asks"]:
        ask_s = p["asks"][0]["price"]
    else:
        ask_s = p.get("closeoutAsk")

    bid = Decimal(str(bid_s))
    ask = Decimal(str(ask_s))
    mid = (bid + ask) / Decimal(2)
    return bid, ask, mid

def get_spread_pips(instrument: str) -> float:
    bid, ask, _ = get_price_snapshot(instrument)
    spread = ask - bid
    pip = pip_size_for(instrument)
    return float(spread / pip)

def get_mid_price(instrument: str) -> Decimal:
    _, _, mid = get_price_snapshot(instrument)
    return mid

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
    """
    Sums the absolute value of units across ALL open positions on the account.
    Used to prevent over-leveraging across multiple correlated pairs (USD Basket risk).
    """
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

def sync_trade_state_from_oanda(
    *,
    instrument: str,
    acct_ccy: Optional[str] = None,
    entry_time_ms_hint: Optional[int] = None,
    fallback_entry_price: Optional[Decimal] = None,
    fallback_units_abs: Optional[int] = None
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
        upsert_trade_state(
            instrument=instrument,
            is_open=False,
            side=None,
            units=None,
            entry_price=None,
            entry_time_ms=None,
            last_mark_price=None,
            unrealized_pl_home=None,
            realized_pl_home=None
        )
        return

    entry_price = avg_entry or fallback_entry_price
    if entry_price is None:
        try:
            entry_price = get_mid_price(instrument)
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
        realized_pl_home=(Decimal(str(prev["realized_pl_home"])) if prev and prev.get("realized_pl_home") else None)
    )

def get_net_units_for_instrument(instrument: str) -> int:
    net, _, _ = get_net_and_avg_entry_from_oanda(instrument)
    return int(net)

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

# ============================================================
# PnL helpers
# ============================================================
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
    mid2 = (bid2 + ask2) / Decimal(2)   # 1 home = mid2 quote
    return Decimal(1) / mid2            # 1 quote = 1/mid2 home

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

# ============================================================
# Risk sizing
# ============================================================
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

def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

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

    # --- SAFETY BLOCK: NO OPENS DURING ROLLOVER WINDOW (UTC) ---
    if ROLLOVER_BLOCK_OPENS and in_rollover_window(datetime.datetime.now(datetime.timezone.utc)):
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_ROLLOVER", oanda_http=None, oanda_response=None,
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
            status="SAFETY_BLOCKED_FLIP", oanda_http=None, oanda_response=None,
            meta=json.dumps(meta)
        )
        return {"status": "SAFETY_BLOCKED_FLIP", "reason": "Cannot OPEN SELL while Long. Use CLOSE first."}, 200

    if current_net < 0 and side == "buy":
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=None, tp_pips=None,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="SAFETY_BLOCKED_FLIP", oanda_http=None, oanda_response=None,
            meta=json.dumps(meta)
        )
        return {"status": "SAFETY_BLOCKED_FLIP", "reason": "Cannot OPEN BUY while Short. Use CLOSE first."}, 200

    # 1. Spread Gate 
    current_max_spread = float(max_spread_pips) if max_spread_pips is not None else float(MAX_SPREAD_PIPS)

    if spread_pips_val is None:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=None,
            status="RISK_BLOCKED_PRICE_FETCH_FAIL", oanda_http=None, oanda_response=None,
            meta=json.dumps({**meta, "max_spread_pips": current_max_spread})
        )
        return {"status": "RISK_BLOCKED_PRICE_FETCH_FAIL"}, 503

    if spread_pips_val > current_max_spread:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_SPREAD", oanda_http=None, oanda_response=None,
            meta=json.dumps({**meta, "max_spread_pips": current_max_spread})
        )
        return {"status": "RISK_BLOCKED_SPREAD", "spread_pips": spread_pips_val, "cap": current_max_spread}, 403

    # 2. Daily Loss Limit
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
            return {"status": "RISK_BLOCKED_DAILY_LOSS_LIMIT"}, 403

    # 3. Position sizing (Risk % + GPT scale + Min/Max constraints)
    try:
        units = compute_units_from_risk(instrument=pair, sl_pips=sl_pips, nav=nav, account_ccy=acct_ccy)
    except Exception as e:
        log(f"RISK SIZING ERROR | {pair} | {e}")
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=None, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_CALC_ERROR", oanda_http=None, oanda_response=str(e),
            meta=json.dumps(meta)
        )
        return {"status": "RISK_CALC_ERROR", "error": str(e)}, 403

    units_mult = max(0.1, min(float(units_mult), GPT_MAX_UNITS_MULT))
    units = int(units * units_mult)

    min_units = int(os.getenv("MIN_TRADE_UNITS", "10"))
    units = max(min_units, units)
    units = min(units, MAX_UNITS)

    # Enforce TP floor only if TP enabled
    if tp_pips is not None and tp_pips > 0:
        tp_pips = int(max(10, min(tp_pips, MAX_TP_PIPS)))

    # 3b. FIFO safeguard
    if FIFO_ENFORCE_UNIQUE_UNITS:
        headroom = MAX_NET_UNITS - abs(int(current_net))
        if headroom <= 0:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_MAX_NET_UNITS", oanda_http=None, oanda_response=None,
                meta=json.dumps({**(meta or {}), "reason": "no_headroom_for_fifo"})
            )
            return {"status": "RISK_BLOCKED_MAX_NET_UNITS", "reason": "No headroom."}, 403

        existing = get_open_trade_unit_sizes(pair)
        if existing is None:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=None, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_FIFO_SIZES_UNKNOWN", oanda_http=None, oanda_response=None,
                meta=json.dumps({**(meta or {}), "reason": "open_trades_fetch_failed"})
            )
            return {"status": "RISK_BLOCKED_FIFO_SIZES_UNKNOWN"}, 503

        max_units_fifo = min(MAX_UNITS, max(min_units, headroom))
        original_units = units

        units = fifo_make_units_unique(
            pair,
            units,
            min_units=min_units,
            max_units=max_units_fifo,
            bump_limit=FIFO_BUMP_LIMIT
        )

        # If still not unique, do NOT send an order that will likely cancel
        if units in existing:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=units, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_FIFO_UNIQUE_UNAVAILABLE", oanda_http=None, oanda_response=None,
                meta=json.dumps({**(meta or {}), "fifo_units_original": original_units, "fifo_units_final": units})
            )
            return {"status": "RISK_BLOCKED_FIFO_UNIQUE_UNAVAILABLE"}, 403

        if units != original_units:
            meta = {**(meta or {}), "fifo_units_original": original_units, "fifo_units_final": units}

    # 4. Absolute Max Risk Gate
    try:
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
    except Exception as e:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
            status="RISK_CALC_ERROR", oanda_http=None, oanda_response=str(e),
            meta=json.dumps(meta)
        )
        return {"status": "RISK_CALC_ERROR", "error": str(e)}, 403

    # 5. Net exposure gate (Per Pair)
    signed_units = -units if side == "sell" else units
    projected_net = current_net + signed_units

    if abs(projected_net) > MAX_NET_UNITS:
        db_record_execution(
            alert_id=alert_id, action="open", instrument=pair, side=side,
            units=units, sl_pips=sl_pips, tp_pips=tp_pips,
            current_net=current_net, projected_net=projected_net, spread_pips=spread_pips_val,
            status="RISK_BLOCKED_MAX_NET_UNITS", oanda_http=None, oanda_response=None,
            meta=json.dumps(meta)
        )
        return {"status": "RISK_BLOCKED_MAX_NET_UNITS", "current_net": current_net, "projected_net": projected_net, "cap": MAX_NET_UNITS}, 403

    # 5b. Global Exposure Gate (All Pairs)
    try:
        current_total_exposure = get_total_exposure_units()
        if current_total_exposure + units > GLOBAL_MAX_UNITS:
            db_record_execution(
                alert_id=alert_id, action="open", instrument=pair, side=side,
                units=units, sl_pips=sl_pips, tp_pips=tp_pips,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status="RISK_BLOCKED_GLOBAL_MAX_UNITS", oanda_http=None, oanda_response=None,
                meta=json.dumps({**(meta or {}), "global_exposure": current_total_exposure, "cap": GLOBAL_MAX_UNITS})
            )
            return {"status": "RISK_BLOCKED_GLOBAL_MAX_UNITS", "current_total": current_total_exposure, "cap": GLOBAL_MAX_UNITS}, 403
    except Exception as e:
        log(f"GLOBAL EXPOSURE CHECK FAIL | {e}")
        return {"status": "RISK_CHECK_FAIL", "error": str(e)}, 503

    # 6. Execute Order
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
            })
        )

        # Update Trade State only if we have a fill or if we were already in a trade (sync anyway)
        fallback_fill_price = None
        try:
            oft = resp_json.get("orderFillTransaction", {}) if isinstance(resp_json, dict) else {}
            if isinstance(oft, dict) and "price" in oft:
                fallback_fill_price = Decimal(str(oft["price"]))
        except Exception:
            fallback_fill_price = None

        if fallback_fill_price is None:
            try:
                fallback_fill_price = get_mid_price(pair)
            except Exception:
                fallback_fill_price = None

        prev = get_trade_state(pair)
        prev_is_open = bool(prev) and int(prev.get("is_open", 0) or 0) == 1
        entry_time_ms_hint = int(prev.get("entry_time_ms")) if (prev_is_open and prev.get("entry_time_ms")) else int(time.time() * 1000)

        if bool(r.ok):
            # Sync to broker truth (handles fill/cancel and pyramids)
            sync_trade_state_from_oanda(
                instrument=pair,
                acct_ccy=acct_ccy,
                entry_time_ms_hint=entry_time_ms_hint,
                fallback_entry_price=fallback_fill_price,
                fallback_units_abs=abs(projected_net)
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
    return jsonify({"ok": True, "mode": MODE})

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
            alert_id=alert_id, action=payload["action"], instrument=payload.get("pair", "N/A"),
            status="DUPLICATE_IGNORED", meta=None
        )
        return {"status": "DUPLICATE_IGNORED"}, 200

    action = payload["action"]

    if action == "close_all":
        try:
            results = close_all_positions()
            log(f"CLOSE_ALL | alert_id={alert_id} | closed={len(results)}")
            for item in results:
                db_record_execution(
                    alert_id=alert_id, action="close", instrument=item["instrument"],
                    oanda_http=item["http_status"], status=("OK" if item["ok"] else "OANDA_ERROR"),
                    oanda_response=str(item["response"]), meta=None
                )
                sync_trade_state_from_oanda(instrument=item["instrument"])
            return {"status": "OK", "results": results}, 200
        except Exception as e:
            db_record_execution(alert_id=alert_id, action="close_all", instrument="ALL", status="ERROR", oanda_response=repr(e), meta=None)
            return {"status": "ERROR", "error": str(e)}, 502

    pair = payload["pair"]

    if action == "close":
        try:
            r = close_position(pair)
            log(f"CLOSE | alert_id={alert_id} | {pair} | HTTP {r.status_code}")
            sync_trade_state_from_oanda(instrument=pair)

            status = "OK" if r.ok else "OANDA_ERROR"
            db_record_execution(
                alert_id=alert_id, action="close", instrument=pair,
                oanda_http=r.status_code, status=status, oanda_response=r.text, meta=None
            )
            return {"status": status, "http_status": r.status_code}, (200 if r.ok else 502)
        except requests.RequestException as e:
            db_record_execution(alert_id=alert_id, action="close", instrument=pair, status="NETWORK_ERROR", oanda_response=repr(e), meta=None)
            return {"status": "NETWORK_ERROR"}, 502

    if action == "observe":
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

        # Lazy load NAV/currency
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
        is_overextended = False
        
        # --- [FIXED] JPY "Leash" Logic ---
        # Allow wider extension for JPY pairs due to volatility
        overextended_threshold = 100 if "JPY" in pair else 50
        
        if abs(dist_to_ema) > overextended_threshold:
            is_overextended = True
        elif abs(dist_to_ema) > (overextended_threshold * 0.8) and adx < 20:
            is_overextended = True

        lookback = features.get("lookback", [])
        trend_bias = "mixed"
        if lookback and len(lookback) >= 5:
            recent = lookback[-5:]
            up_count = sum(1 for c in recent if _f(c.get("close")) > _f(c.get("open")))
            if up_count >= 4:
                trend_bias = "bullish"
            elif up_count <= 1:
                trend_bias = "bearish"

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        in_roll = in_rollover_window(now_utc)

        analysis = {
            "is_overextended": is_overextended,
            "trend_bias": trend_bias,
            "dist_to_ema_200": dist_to_ema,
            "in_rollover_window": bool(in_roll)
        }

        # --- Pricing snapshot ---
        spread_pips_val = None
        bid = ask = mid = None

        try:
            bid, ask, mid = get_price_snapshot(pair)
            spread_pips_val = float((ask - bid) / pip_size_for(pair))
        except Exception:
            spread_pips_val = None

        # Compute UPL pips
        upl_pips = 0.0
        if current_net != 0:
            try:
                sync_trade_state_from_oanda(instrument=pair, acct_ccy=acct_ccy)
            except Exception:
                pass

            ts_data = get_trade_state(pair)
            if ts_data.get("entry_price") and ts_data.get("side") and bid is not None and ask is not None:
                entry = Decimal(str(ts_data["entry_price"]))
                side_ts = str(ts_data["side"]).lower()
                mark = bid if side_ts == "buy" else ask
                upl_pips = pips_from_entry(pair, side_ts, entry, mark)

        if spread_pips_val is None:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=None,
                status="RULE_BLOCKED_SPREAD_UNKNOWN", meta=None
            )
            return {"status": "RULE_BLOCKED_SPREAD_UNKNOWN"}, 200

        # Dynamic per-pair spread limits
        pair_spread_limits = {
            "XAU_USD": MAX_SPREAD_PIPS_XAU,
            "GBP_JPY": 6.0,
            "GBP_USD": 4.0,
            "USD_JPY": 3.0,
            "EUR_USD": 2.5,
            "AUD_USD": 2.5
        }
        dynamic_spread_limit = float(pair_spread_limits.get(pair, MAX_SPREAD_PIPS))

        # Derived EXIT limit
        exit_spread_limit = float(dynamic_spread_limit) * float(EXIT_SPREAD_MULT) + float(EXIT_SPREAD_MIN_ADD)
        if exit_spread_limit < dynamic_spread_limit:
            exit_spread_limit = dynamic_spread_limit

        analysis["max_spread_pips_entry"] = dynamic_spread_limit
        analysis["max_spread_pips_exit"] = exit_spread_limit
        analysis["spread_exit_caution"] = bool(current_net != 0 and spread_pips_val > exit_spread_limit)

        # Rollover block
        if current_net == 0 and ROLLOVER_BLOCK_OPENS and in_roll:
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

        # Spread gate
        if spread_pips_val > dynamic_spread_limit and current_net == 0:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_SPREAD", meta=json.dumps({"limit": dynamic_spread_limit})
            )
            return {"status": "RULE_BLOCKED_SPREAD"}, 200

        # Chop gate
        ema_sep = _f(features.get("ema_sep_pips", 0))
        required_sep = MIN_EMA_SEP_PIPS * (1.2 if "JPY" in pair else 1.0)
        if current_net == 0 and ema_sep < required_sep:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_CHOP",
                meta=json.dumps({"ema_sep": ema_sep, "required": required_sep})
            )
            return {"status": "RULE_BLOCKED_CHOP", "ema_sep": ema_sep}, 200

        # Sunday liquidity filter
        if "XAU" in pair and now_utc.weekday() == 6 and now_utc.hour < 22:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_SUNDAY_OPEN",
                meta=json.dumps({"hour_utc": now_utc.hour})
            )
            return {"status": "RULE_BLOCKED_SUNDAY_OPEN"}, 200

        # Overextended block
        if current_net == 0 and is_overextended:
            db_record_execution(
                alert_id=alert_id, action="observe", instrument=pair,
                current_net=current_net, spread_pips=spread_pips_val,
                status="RULE_BLOCKED_OVEREXTENDED", meta=json.dumps(analysis)
            )
            return {"status": "RULE_BLOCKED_OVEREXTENDED"}, 200

        # Degradation exit
        if DEGRADE_EXIT_ENABLED and current_net != 0:
            ts2 = get_trade_state(pair)
            if ts2 and int(ts2.get("is_open", 0)) == 1 and ts2.get("entry_price") and ts2.get("entry_time_ms"):
                try:
                    entry = Decimal(str(ts2["entry_price"]))
                    entry_time_ms = int(ts2["entry_time_ms"])
                    now_ms = int(time.time() * 1000)
                    held_min = (now_ms - entry_time_ms) / 60000.0

                    side_ts = str(ts2.get("side") or ("buy" if current_net > 0 else "sell")).lower()
                    if side_ts not in ("buy", "sell"):
                        side_ts = "buy" if current_net > 0 else "sell"

                    bid, ask, _ = get_price_snapshot(pair)
                    mark = bid if side_ts == "buy" else ask

                    ema_sep_now = _f(features.get("ema_sep_pips", 0))
                    adx_now = _f(features.get("adx", 0))
                    trend_bias_now = analysis.get("trend_bias", "mixed")
                    degrade_sep = DEGRADE_EMA_SEP_PIPS * (1.2 if "JPY" in pair else 1.0)

                    is_degraded = (
                        adx_now <= DEGRADE_ADX_MAX and
                        ema_sep_now <= degrade_sep and
                        trend_bias_now == "mixed"
                    )

                    pips_now = pips_from_entry(pair, side_ts, entry, mark)

                    stall_exit = (
                        held_min >= DEGRADE_MIN_MINUTES and
                        abs(pips_now) <= DEGRADE_STALL_PIPS and
                        is_degraded
                    )

                    profit_protect_exit = (
                        held_min >= DEGRADE_PROFIT_PROTECT_MIN_MINUTES and
                        pips_now >= DEGRADE_PROFIT_PROTECT_PIPS and
                        is_degraded
                    )

                    if stall_exit or profit_protect_exit:
                        reason = (
                            f"RULE_DEGRADATION_EXIT: held={held_min:.1f}m, "
                            f"pips={pips_now:.1f}, adx={adx_now:.1f}, "
                            f"ema_sep={ema_sep_now:.2f} (req<={degrade_sep:.2f}), "
                            f"trend_bias={trend_bias_now}, "
                            f"mode={'STALL' if stall_exit else 'PROFIT_PROTECT'}"
                        )

                        # --- Exit-spread protection (STRICT for degradation exits) ---
                        spread_block = (
                            spread_pips_val is not None and
                            exit_spread_limit is not None and
                            float(spread_pips_val) > float(exit_spread_limit)
                        )

                        if spread_block:
                            block_meta = {
                                "blocked_by": "EXIT_SPREAD",
                                "rule": "DEGRADATION_EXIT",
                                "mode": ("STALL" if stall_exit else "PROFIT_PROTECT"),
                                "reason": reason,
                                "spread_pips": float(spread_pips_val),
                                "exit_spread_limit": float(exit_spread_limit),
                                "held_min": held_min,
                                "pips_now": pips_now,
                                "adx": adx_now,
                                "ema_sep_pips": ema_sep_now,
                                "trend_bias": trend_bias_now,
                                "in_rollover_window": bool(in_roll),
                            }
                            db_record_execution(
                                alert_id=alert_id,
                                action="observe",
                                instrument=pair,
                                current_net=current_net,
                                spread_pips=spread_pips_val,
                                status="RULE_BLOCKED_EXIT_SPREAD_DEGRADE",
                                meta=json.dumps(block_meta),
                            )
                            return {
                                "status": "RULE_BLOCKED_EXIT_SPREAD_DEGRADE",
                                "reason": "Degradation exit blocked due to wide spread.",
                                "spread_pips": spread_pips_val,
                                "exit_spread_limit": exit_spread_limit,
                            }, 200

                        # --- Allowed to close ---
                        r = close_position(pair)
                        sync_trade_state_from_oanda(instrument=pair, acct_ccy=acct_ccy)

                        status = "OK" if r.ok else "OANDA_ERROR"
                        db_record_execution(
                            alert_id=alert_id,
                            action="close",
                            instrument=pair,
                            side=None,
                            units=None,
                            sl_pips=None,
                            tp_pips=None,
                            current_net=current_net,
                            projected_net=None,
                            spread_pips=spread_pips_val,
                            oanda_http=r.status_code,
                            status=status,
                            oanda_response=r.text,
                            meta=json.dumps({
                                "action": "CLOSE",
                                "confidence": 1.0,
                                "reason": reason,
                                "rule": "DEGRADATION_EXIT",
                                "mode": ("STALL" if stall_exit else "PROFIT_PROTECT"),
                                "held_min": held_min,
                                "pips_now": pips_now,
                                "adx": adx_now,
                                "ema_sep_pips": ema_sep_now,
                                "trend_bias": trend_bias_now,
                                "in_rollover_window": bool(in_roll),
                                "max_spread_pips_exit": exit_spread_limit,
                            })
                        )
                        return {"status": status, "action": "close", "reason": reason}, (200 if r.ok else 502)


                except Exception as e:
                    log(f"DEGRADE_EXIT_FAIL | {pair} | {repr(e)}")

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
            },
            "state": {
                "spread_pips": spread_pips_val,
                "current_net": current_net,
                "upl_pips": round(float(upl_pips), 1),
                "in_rollover_window": bool(in_roll),
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
                side=None, units=None, sl_pips=hint_sl, tp_pips=hint_tp,
                current_net=current_net, projected_net=None, spread_pips=spread_pips_val,
                status=f"GPT_{gpt_decision['action']}", oanda_http=None, oanda_response=None,
                meta=json.dumps(gpt_decision)
            )
            return {"status": f"GPT_{gpt_decision['action']}", "decision": gpt_decision}, 200

        if gpt_decision["action"] == "CLOSE":
            try:
                # --- Exit-spread protection ---
                spread_block = (
                    spread_pips_val is not None and
                    exit_spread_limit is not None and
                    float(spread_pips_val) > float(exit_spread_limit)
                )

                dist_to_ema = float(analysis.get("dist_to_ema_200", 0.0) or 0.0)
                macro_flip = (
                    (current_net > 0 and dist_to_ema < 0) or
                    (current_net < 0 and dist_to_ema > 0)
                )

                reason_txt = str(gpt_decision.get("reason", "")).lower()
                keyword_emergency = any(k in reason_txt for k in [
                    "ema 200", "ema200", "200 ema",
                    "market structure", "structure break", "break of structure", "bos",
                    "invalidation", "trend reversed", "reversal"
                ])

                allow_emergency_close = bool(macro_flip or keyword_emergency)

                if spread_block and not allow_emergency_close:
                    # Record that we intentionally refused the close
                    block_meta = {
                        "blocked_by": "EXIT_SPREAD",
                        "spread_pips": float(spread_pips_val),
                        "exit_spread_limit": float(exit_spread_limit),
                        "dynamic_spread_limit": float(analysis.get("max_spread_pips_entry", 0.0) or 0.0),
                        "gpt_decision": gpt_decision,
                        "analysis": analysis,
                        "note": "Close blocked due to wide spreads; not an emergency invalidation."
                    }
                    db_record_execution(
                        alert_id=alert_id,
                        action="observe",
                        instrument=pair,
                        current_net=current_net,
                        spread_pips=spread_pips_val,
                        status="RULE_BLOCKED_EXIT_SPREAD",
                        meta=json.dumps(block_meta)
                    )
                    return {
                        "status": "RULE_BLOCKED_EXIT_SPREAD",
                        "spread_pips": spread_pips_val,
                        "exit_spread_limit": exit_spread_limit,
                        "reason": "Exit blocked due to wide spread (non-emergency)."
                    }, 200

                # --- Allowed to close ---
                r = close_position(pair)
                sync_trade_state_from_oanda(instrument=pair, acct_ccy=acct_ccy)

                status = "OK" if r.ok else "OANDA_ERROR"
                db_record_execution(
                    alert_id=alert_id, action="close", instrument=pair, side=None,
                    units=None, sl_pips=None, tp_pips=None, current_net=current_net,
                    projected_net=None, spread_pips=spread_pips_val,
                    oanda_http=r.status_code, status=status, oanda_response=r.text,
                    meta=json.dumps({
                        **gpt_decision,
                        "exit_spread_limit": float(exit_spread_limit) if exit_spread_limit is not None else None,
                        "macro_flip": bool(macro_flip),
                        "keyword_emergency": bool(keyword_emergency),
                    })
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
            # --- [FIXED] Default Cooldown Increased to 60m ---
            cooldown_min = int(os.getenv("PYRAMID_COOLDOWN_MINUTES", "60"))
            if current_net != 0 and not check_cooldown(pair, cooldown_min):
                log(f"COOLDOWN_ACTIVE | {pair}")

                gpt_reason = str(gpt_decision.get("reason", "")).strip()
                gpt_conf = gpt_decision.get("confidence", None)

                cooldown_reason = (
                    f"COOLDOWN_ACTIVE ({cooldown_min}m): suppressed OPEN. "
                    f"GPT would have OPEN {gpt_decision.get('side')} "
                    f"(units_mult={gpt_decision.get('units_mult')}, conf={gpt_conf}). "
                    f"GPT reason: {gpt_reason}"
                )[:1200]

                cooldown_meta = {
                    "reason": cooldown_reason,
                    "confidence": float(gpt_conf) if gpt_conf is not None else None,
                    "blocked_by": "COOLDOWN_ACTIVE",
                    "cooldown_minutes": cooldown_min,
                    "gpt_proposed": gpt_decision,
                }

                db_record_execution(
                    alert_id=alert_id,
                    action="observe",
                    instrument=pair,
                    current_net=current_net,
                    spread_pips=spread_pips_val,
                    status="COOLDOWN_ACTIVE",
                    meta=json.dumps(cooldown_meta),
                )
                return {"status": "COOLDOWN_ACTIVE"}, 200

            # --- [FIXED] Default Spacing Increased to 300s ---
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

            # Dynamic SL/TP
            default_sl = gpt_decision["sl_pips"] if gpt_decision["sl_pips"] is not None else hint_sl
            default_tp = gpt_decision["tp_pips"] if gpt_decision["tp_pips"] is not None else hint_tp
            tp_disabled = (default_tp == 0)

            atr_val = _f(features.get("atr", 0))
            if atr_val > 0:
                pip_unit = float(pip_size_for(pair))
                atr_in_pips = atr_val / pip_unit
                # --- [FIXED] Using M15 calibrated multipliers ---
                tp_multiplier = float(os.getenv("TP_ATR_MULTIPLIER", "3.0"))
                sl_multiplier = float(os.getenv("SL_ATR_MULTIPLIER", "2.5"))

                sl_pips = max(10, int(atr_in_pips * sl_multiplier))
                if tp_disabled:
                    tp_pips = 0
                else:
                    tp_pips = max(10, int(atr_in_pips * tp_multiplier))

                log(f"DYNAMIC_CALC | {pair} | ATR={atr_val} ({atr_in_pips:.1f} pips) | TP={tp_pips} (x{tp_multiplier}) | SL={sl_pips} (x{sl_multiplier})")
            else:
                tp_pips = default_tp
                sl_pips = default_sl

            sl_pips = int(max(MIN_SL_PIPS, min(sl_pips, MAX_SL_PIPS)))
            tp_pips = int(max(0, min(tp_pips, MAX_TP_PIPS)))

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