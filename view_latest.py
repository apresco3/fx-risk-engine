import sqlite3
import json
import os
import textwrap
import datetime
import time
from dotenv import load_dotenv
from decimal import Decimal

load_dotenv()

# --- OANDA IMPORTS ---
try:
    import oandapyV20
    import oandapyV20.endpoints.trades as trades
    from oandapyV20 import API
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("WARNING: 'oandapyV20' not installed. Live audit will be skipped.")
    print("Run: pip install oandapyV20")

# --- CONFIGURATION ---
DB_PATH = "bot.db"
# Ensure these match your bot's settings.py or .env
ACCESS_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {}


def _fmt_num(x, fmt=":.2f"):
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


def _format_degradation_exit(meta: dict) -> str:
    mode = str(meta.get("mode", "UNKNOWN"))
    held = meta.get("held_min", None)
    pips = meta.get("pips_now", meta.get("pips", None))
    adx = meta.get("adx", None)
    sep = meta.get("ema_sep_pips", None)
    tb = meta.get("trend_bias", None)

    parts = [f"Degradation Exit ({mode})"]
    if held is not None:
        parts.append(f"held {_fmt_num(held, ':.1f')}m")
    if pips is not None:
        try:
            parts.append(f"pips {float(pips):+.1f}")
        except Exception:
            parts.append(f"pips {pips}")
    if adx is not None:
        parts.append(f"ADX {_fmt_num(adx, ':.1f')}")
    if sep is not None:
        parts.append(f"EMA sep {_fmt_num(sep, ':.2f')} pips")
    if tb is not None:
        parts.append(f"trend_bias={tb}")

    return ", ".join(parts) + "."


def format_reason(data, meta):
    gpt_reason = None
    confidence = None

    if isinstance(meta, dict) and meta.get("rule") == "DEGRADATION_EXIT":
        gpt_reason = _format_degradation_exit(meta)
        confidence = meta.get("confidence", 1.0)
        return gpt_reason, confidence

    if isinstance(meta, dict) and "gpt" in meta and isinstance(meta["gpt"], dict):
        gpt_reason = meta["gpt"].get("reason")
        confidence = meta["gpt"].get("confidence")
    elif isinstance(meta, dict) and ("reason" in meta or "confidence" in meta):
        gpt_reason = meta.get("reason")
        confidence = meta.get("confidence")

    if not gpt_reason:
        status = str(data.get("status", "")).upper()
        spread = data.get("spread_pips", None)

        if status == "RULE_BLOCKED_SPREAD":
            limit = "N/A"
            if isinstance(meta, dict):
                limit = meta.get("limit", meta.get("max_spread_pips", "N/A"))
            if spread is not None:
                gpt_reason = f"Spread too wide: {spread} pips > limit {limit}"
            else:
                gpt_reason = f"Spread too wide (limit {limit})"

        elif status == "RULE_BLOCKED_SPREAD_UNKNOWN":
            gpt_reason = "Spread unknown: pricing fetch failed."

        elif status == "RULE_BLOCKED_CHOP":
            if isinstance(meta, dict):
                sep = meta.get("ema_sep")
                req = meta.get("required")
                if isinstance(sep, (int, float)):
                    sep = f"{sep:.2f}"
                if isinstance(req, (int, float)):
                    req = f"{req:.2f}"
                gpt_reason = f"Chop Gate: EMA separation {sep} pips < required {req} pips."
            else:
                gpt_reason = "Blocked by Chop Gate (low momentum)."

        elif status == "RULE_BLOCKED_SUNDAY_OPEN":
            if isinstance(meta, dict):
                h = meta.get("hour_utc", "?")
                gpt_reason = f"Sunday Liquidity Filter: Blocked (Hour {h} UTC)."
            else:
                gpt_reason = "Sunday Liquidity Filter: Blocked due to time/day."

        elif status == "RULE_BLOCKED_OVEREXTENDED":
            gpt_reason = "Blocked: overextended entry (mean-reversion risk)."

        elif status == "COOLDOWN_ACTIVE":
            gpt_reason = "Cooldown active: suppressed re-entry/pyramiding."

        elif status == "STALE_ALERT_IGNORED":
            if isinstance(meta, dict) and "age_sec" in meta and "max_age" in meta:
                gpt_reason = f"Stale alert ignored: age {meta['age_sec']:.1f}s > max {meta['max_age']}s"
            else:
                gpt_reason = "Stale alert ignored."

        elif status == "DUPLICATE_IGNORED":
            gpt_reason = "Duplicate alert_id ignored."

        elif status.startswith("RISK_BLOCKED_"):
            gpt_reason = status

        elif status == "RISK_CALC_ERROR":
            gpt_reason = "Risk sizing calculation error."

        elif status == "OANDA_ERROR":
            gpt_reason = "OANDA returned error (see executions.oanda_response)."

        elif status == "NETWORK_ERROR":
            gpt_reason = "Network error when calling OANDA."

        elif status == "SAFETY_BLOCKED_FLIP":
            reason = "Safety Block: Cannot flip Net Long/Short instantly."
            if isinstance(meta, dict) and "reason" in meta:
                reason = meta["reason"]
            gpt_reason = reason

        else:
            gpt_reason = "No reason found in meta."

    if confidence is None:
        confidence = "N/A"

    return gpt_reason, confidence


def _fmt_ts_both(ts: int) -> tuple[str, str]:
    dt_local = datetime.datetime.fromtimestamp(ts)
    dt_utc = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt_local.strftime("%Y-%m-%d %H:%M:%S"), dt_utc.strftime("%Y-%m-%d %H:%M:%S")


def pip_size_for(instrument: str) -> Decimal:
    if "XAU" in instrument:
        return Decimal("0.1")
    if instrument.endswith("_JPY"):
        return Decimal("0.01")
    return Decimal("0.0001")


def pips_from_entry(instrument: str, side: str, entry: Decimal, mark: Decimal) -> float:
    pip = pip_size_for(instrument)
    if side.lower() == "buy":
        return float((mark - entry) / pip)
    return float((entry - mark) / pip)


def print_execution(title, data, meta):
    gpt_reason, confidence = format_reason(data, meta)

    ts = int(data.get("ts", 0) or 0)
    local_str, utc_str = _fmt_ts_both(ts)

    print("\n" + "=" * 60)
    print(f" {title}: {str(data.get('action', '')).upper()} on {data['instrument']} ({data.get('status', 'N/A')})")
    print("=" * 60)
    print(f"TIME (LOCAL): {local_str}")
    print(f"TIME (UTC):   {utc_str} UTC")
    print(f"CONFIDENCE:   {confidence}")
    print(f"SPREAD:       {data.get('spread_pips', 'N/A')}")
    print("\nREASONING:")
    print("-" * 20)
    print(textwrap.fill(str(gpt_reason), width=80))
    print("-" * 20)


def get_oanda_trades():
    """Fetches individual tickets directly from OANDA API."""
    if not OANDA_AVAILABLE:
        return None
    if not ACCESS_TOKEN or not ACCOUNT_ID:
        print("\n[!] Cannot fetch Live Tickets: OANDA_ACCESS_TOKEN or OANDA_ACCOUNT_ID missing from environment.")
        return None

    try:
        client = API(access_token=ACCESS_TOKEN)
        r = trades.TradesList(accountID=ACCOUNT_ID, params={"state": "OPEN"})
        client.request(r)
        return r.response.get("trades", [])
    except Exception as e:
        print(f"\n[!] Error contacting OANDA API: {e}")
        return None


def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Fetch Latest Execution Logs
    cutoff_ts = int(time.time() - 86400)
    query_logs = """
        SELECT e.* FROM executions e
        JOIN (
            SELECT instrument, MAX(ts) as max_ts
            FROM executions
            WHERE ts > ?
            GROUP BY instrument
        ) latest ON e.instrument = latest.instrument AND e.ts = latest.max_ts
        ORDER BY e.instrument ASC
    """
    cur.execute(query_logs, (cutoff_ts,))
    recent_rows = cur.fetchall()

    # 2. Fetch Portfolio from Database (Aggregated State)
    query_portfolio = "SELECT * FROM trade_state WHERE is_open = 1 ORDER BY instrument ASC"
    cur.execute(query_portfolio)
    active_trades = cur.fetchall()

    conn.close()

    # --- PRINT SYSTEM STATUS ---
    print("\n" + "#" * 60)
    print(" SYSTEM STATUS (Last Report per Pair)")
    print("#" * 60)

    if not recent_rows:
        print("No activity in last 24 hours.")
    else:
        for row in recent_rows:
            data = dict(row)
            meta = _safe_json_loads(data["meta"]) if data.get("meta") else {}
            print_execution(f"LATEST LOG: {data['instrument']}", data, meta)

    # --- PRINT DATABASE PORTFOLIO ---
    print("\n\n" + "#" * 60)
    print(" 🤖 ACTIVE PORTFOLIO (Bot Internal State)")
    print("#" * 60)
    print("(Aggregated Net Positions tracked by bot logic)")

    acct_ccy = os.getenv("ACCOUNT_CURRENCY", "USD")

    if not active_trades:
        print("\n[FLAT] No open trades in Bot DB.")
    else:
        for row in active_trades:
            t = dict(row)

            entry_ms = t.get("entry_time_ms")
            if entry_ms:
                entry_time = datetime.datetime.fromtimestamp(entry_ms / 1000.0)
                duration = datetime.datetime.now() - entry_time
                duration_str = str(duration).split(".")[0]
            else:
                duration_str = "N/A"

            entry = None
            mark = None
            pips = None
            try:
                if t.get("entry_price") is not None:
                    entry = Decimal(str(t["entry_price"]))
                if t.get("last_mark_price") is not None:
                    mark = Decimal(str(t["last_mark_price"]))
                if entry is not None and mark is not None and t.get("side"):
                    pips = pips_from_entry(t["instrument"], str(t["side"]), entry, mark)
            except Exception:
                pips = None

            print(f"\n>>> NET POSITION: {t['instrument']}")
            print(f"    SIDE:        {str(t['side']).upper()}")
            print(f"    NET UNITS:   {t['units']}")
            print(f"    AVG ENTRY:   {t['entry_price']}")
            print(f"    MARK PRICE:  {t.get('last_mark_price', 'N/A')}")
            if pips is not None:
                print(f"    PIPS (Est):  {pips:+.1f}")
            print(f"    DURATION:    {duration_str}")
            print(f"    UPL (Est):   {t.get('unrealized_pl_home', '0.00')} {acct_ccy}")
            print("-" * 40)

    # --- PRINT LIVE OANDA TICKETS ---
    if OANDA_AVAILABLE:
        print("\n\n" + "#" * 60)
        print(" 🏦 LIVE OANDA TICKETS (Broker Source of Truth)")
        print("#" * 60)
        print("(Individual tickets currently open on OANDA account)")
        
        live_trades = get_oanda_trades()
        
        if live_trades is None:
            print("[INFO] Live fetch skipped (config missing or network error).")
        elif not live_trades:
            print("\n[FLAT] No open tickets found on OANDA.")
        else:
            print(f"\n{'TICKET':<10} {'INSTRUMENT':<12} {'UNITS':<10} {'ENTRY':<10} {'UPL':<10}")
            print("-" * 55)
            for t in live_trades:
                tid = t.get('id', 'N/A')
                instr = t.get('instrument', 'N/A')
                units = t.get('currentUnits', '0')
                price = t.get('price', '0')
                upl = t.get('unrealizedPL', '0.00')
                print(f"{tid:<10} {instr:<12} {units:<10} {price:<10} {upl:<10}")


if __name__ == "__main__":
    main()