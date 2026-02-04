import sqlite3
import json
import os
import textwrap
import datetime
import time
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Tuple, Dict, Any, List

import requests

load_dotenv()

# Optional: oandapyV20 for "LIVE OANDA TICKETS" (trades list)
try:
    import oandapyV20
    import oandapyV20.endpoints.trades as trades
    from oandapyV20 import API
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

DB_PATH = os.getenv("DB_PATH", "bot.db")

ACCESS_TOKEN = os.getenv("OANDA_TOKEN", "")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
MODE = os.getenv("OANDA_MODE", "PRACTICE").upper()
ACCOUNT_CCY = os.getenv("ACCOUNT_CURRENCY", "USD").upper()

if not ACCESS_TOKEN or not ACCOUNT_ID:
    raise RuntimeError("Missing OANDA_TOKEN or OANDA_ACCOUNT_ID in env.")

BASE_URL = (
    "https://api-fxpractice.oanda.com/v3"
    if MODE == "PRACTICE"
    else "https://api-fxtrade.oanda.com/v3"
)

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json",
}


# --------------------------
# Utility
# --------------------------
def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {}

def _oanda_cancel_or_reject_reason(meta: dict, oanda_response: str) -> str:
    """
    Prefer meta["oanda_cancel_reason"] / meta["oanda_reject_reason"] (new rows),
    otherwise parse raw oanda_response JSON (old rows).
    Returns "" if not found.
    """
    if isinstance(meta, dict):
        cr = meta.get("oanda_cancel_reason")
        if cr:
            return f"OANDA cancel: {cr}"
        rr = meta.get("oanda_reject_reason")
        if rr:
            return f"OANDA reject: {rr}"

    try:
        j = json.loads(oanda_response or "{}")
        if isinstance(j, dict):
            if "orderCancelTransaction" in j and isinstance(j["orderCancelTransaction"], dict):
                r = j["orderCancelTransaction"].get("reason") or j["orderCancelTransaction"].get("cancelReason")
                if r:
                    return f"OANDA cancel: {r}"
            if "orderRejectTransaction" in j and isinstance(j["orderRejectTransaction"], dict):
                r = j["orderRejectTransaction"].get("rejectReason") or j["orderRejectTransaction"].get("reason")
                if r:
                    return f"OANDA reject: {r}"
    except Exception:
        pass

    return ""

def _fmt_num(x, fmt=":.2f"):
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)

def _fmt_ts_both(ts: int):
    dt_l = datetime.datetime.fromtimestamp(ts)
    dt_u = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt_l.strftime("%Y-%m-%d %H:%M:%S"), dt_u.strftime("%Y-%m-%d %H:%M:%S")

def pip_size_for(inst: str) -> Decimal:
    if "XAU" in inst:
        return Decimal("0.1")
    return Decimal("0.01") if inst.endswith("_JPY") else Decimal("0.0001")

def pips_from_entry(inst: str, side: str, entry: Decimal, mark: Decimal) -> float:
    pip = pip_size_for(inst)
    if str(side).lower() == "buy":
        return float((mark - entry) / pip)
    return float((entry - mark) / pip)


# --------------------------
# DB helpers
# --------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_upsert_trade_state(
    *,
    instrument: str,
    is_open: bool,
    side: Optional[str],
    units: Optional[int],
    entry_price: Optional[Decimal],
    entry_time_ms: Optional[int],
    last_mark_price: Optional[Decimal],
    unrealized_pl_home: Optional[Decimal],
    realized_pl_home: Optional[Decimal],
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
        int(time.time()),
    ))
    conn.commit()
    conn.close()

def db_get_trade_state(instrument: str) -> dict:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trade_state WHERE instrument = ?", (instrument,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}

def db_mark_closed_not_in(instruments_open: set):
    """
    For any rows currently marked open in DB but not in instruments_open, mark them closed.
    """
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT instrument FROM trade_state WHERE is_open = 1")
    rows = cur.fetchall()
    for r in rows:
        inst = r["instrument"]
        if inst not in instruments_open:
            db_upsert_trade_state(
                instrument=inst,
                is_open=False,
                side=None,
                units=None,
                entry_price=None,
                entry_time_ms=None,
                last_mark_price=None,
                unrealized_pl_home=None,
                realized_pl_home=None,
            )
    conn.close()


# --------------------------
# OANDA REST helpers (refresh trade_state)
# --------------------------
def oanda_get_open_positions() -> dict:
    r = requests.get(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions",
        headers=HEADERS,
        timeout=10
    )
    if not r.ok:
        raise RuntimeError(f"openPositions HTTP {r.status_code}: {r.text}")
    return r.json()

def oanda_get_pricing(instruments_csv: str) -> dict:
    r = requests.get(
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing",
        headers=HEADERS,
        params={"instruments": instruments_csv},
        timeout=10
    )
    if not r.ok:
        raise RuntimeError(f"pricing HTTP {r.status_code}: {r.text}")
    return r.json()

def _extract_bid_ask(p: dict) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    try:
        if "bids" in p and p["bids"]:
            bid_s = p["bids"][0]["price"]
        else:
            bid_s = p.get("closeoutBid")
        if "asks" in p and p["asks"]:
            ask_s = p["asks"][0]["price"]
        else:
            ask_s = p.get("closeoutAsk")
        if bid_s is None or ask_s is None:
            return None, None
        return Decimal(str(bid_s)), Decimal(str(ask_s))
    except Exception:
        return None, None

def _get_conversion_mid(quote_ccy: str, home_ccy: str) -> Optional[Decimal]:
    """
    Returns mid rate converting 1 quote_ccy into home_ccy (quote -> home).
    Uses OANDA pricing for quote_home or its inverse.
    """
    if quote_ccy == home_ccy:
        return Decimal(1)

    pair1 = f"{quote_ccy}_{home_ccy}"
    pair2 = f"{home_ccy}_{quote_ccy}"

    # Try quote_home directly
    try:
        data = oanda_get_pricing(pair1)
        prices = data.get("prices", [])
        if prices:
            bid, ask = _extract_bid_ask(prices[0])
            if bid is not None and ask is not None:
                return (bid + ask) / Decimal(2)
    except Exception:
        pass

    # Try inverse and invert
    try:
        data = oanda_get_pricing(pair2)
        prices = data.get("prices", [])
        if prices:
            bid, ask = _extract_bid_ask(prices[0])
            if bid is not None and ask is not None:
                mid2 = (bid + ask) / Decimal(2)  # 1 home = mid2 quote
                if mid2 == 0:
                    return None
                return Decimal(1) / mid2          # 1 quote = 1/mid2 home
    except Exception:
        pass

    return None

def _compute_unrealized_pl_home(inst: str, side: str, units: int, entry: Decimal, mark: Decimal, account_ccy: str) -> Optional[Decimal]:
    """
    Estimate UPL in account currency:
      PL_quote = (mark-entry)*units (buy) or (entry-mark)*units (sell)
      then convert quote->home using mid.
    """
    try:
        base, quote = inst.split("_")
    except Exception:
        return None

    u = abs(int(units))
    if u == 0:
        return Decimal(0)

    if side == "buy":
        pl_quote = (mark - entry) * Decimal(u)
    else:
        pl_quote = (entry - mark) * Decimal(u)

    conv = _get_conversion_mid(quote.upper(), account_ccy.upper())
    if conv is None:
        return None
    return pl_quote * conv

def refresh_trade_state_from_oanda(account_ccy: str) -> None:
    """
    Refreshes trade_state table using OANDA openPositions + pricing.
    Preserves entry_time_ms for already-open rows.
    """
    try:
        op = oanda_get_open_positions()
    except Exception as e:
        print(f"\n[!] Could not refresh trade_state from OANDA: {e}")
        return

    positions = op.get("positions", []) or []
    if not positions:
        # Mark everything closed
        db_mark_closed_not_in(set())
        return

    # Build list of instruments and position summaries
    open_instruments: List[str] = []
    pos_map: Dict[str, Dict[str, Any]] = {}

    for p in positions:
        inst = p.get("instrument")
        if not inst:
            continue

        long_units = int(p.get("long", {}).get("units", "0"))
        short_units = int(p.get("short", {}).get("units", "0"))
        net = long_units + short_units

        if net == 0:
            continue

        if net > 0:
            side = "buy"
            avg_price = p.get("long", {}).get("averagePrice")
        else:
            side = "sell"
            avg_price = p.get("short", {}).get("averagePrice")

        entry = Decimal(str(avg_price)) if avg_price is not None else None
        open_instruments.append(inst)
        pos_map[inst] = {"net": net, "side": side, "entry": entry}

    instruments_open_set = set(open_instruments)

    # Mark any stale DB rows closed
    db_mark_closed_not_in(instruments_open_set)

    # Fetch pricing for all instruments in one go (best effort)
    pricing_map: Dict[str, Tuple[Optional[Decimal], Optional[Decimal]]] = {}
    try:
        data = oanda_get_pricing(",".join(open_instruments))
        for pr in data.get("prices", []) or []:
            inst = pr.get("instrument")
            if not inst:
                continue
            bid, ask = _extract_bid_ask(pr)
            pricing_map[inst] = (bid, ask)
    except Exception as e:
        print(f"\n[!] Pricing refresh failed (will still update entry/units): {e}")

    now_ms = int(time.time() * 1000)

    # Upsert each open position
    for inst, info in pos_map.items():
        net = int(info["net"])
        side = str(info["side"])
        entry = info.get("entry")

        prev = db_get_trade_state(inst)
        prev_is_open = bool(prev) and int(prev.get("is_open", 0) or 0) == 1
        entry_time_ms = int(prev.get("entry_time_ms")) if (prev_is_open and prev.get("entry_time_ms")) else now_ms

        bid, ask = pricing_map.get(inst, (None, None))
        mark = None
        if bid is not None and ask is not None:
            mark = bid if side == "buy" else ask

        upl_home = None
        if entry is not None and mark is not None:
            upl_home = _compute_unrealized_pl_home(inst, side, abs(net), entry, mark, account_ccy)

        db_upsert_trade_state(
            instrument=inst,
            is_open=True,
            side=side,
            units=abs(net),
            entry_price=entry,
            entry_time_ms=entry_time_ms,
            last_mark_price=mark,
            unrealized_pl_home=upl_home,
            realized_pl_home=(Decimal(str(prev["realized_pl_home"])) if prev and prev.get("realized_pl_home") else None),
        )


# --------------------------
# Reason formatting (execution rows)
# --------------------------
def _format_degradation_exit(meta: dict) -> str:
    mode, held = str(meta.get("mode", "UNKNOWN")), meta.get("held_min")
    pips, adx = meta.get("pips_now", meta.get("pips")), meta.get("adx")
    sep, tb = meta.get("ema_sep_pips"), meta.get("trend_bias")

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

def _format_rollover_reason(meta: dict) -> str:
    start = meta.get("rollover_start_utc") or os.getenv("ROLLOVER_START_UTC", "21:55")
    end = meta.get("rollover_end_utc") or os.getenv("ROLLOVER_END_UTC", "22:45")
    return f"Rollover window: OPEN blocked ({start}–{end} UTC)."

def format_reason(data, meta):
    """
    Prefer broker cancel/reject reasons when status indicates it.
    Otherwise prefer GPT reason/confidence when present.
    Otherwise derive a human reason from status/spread/meta.
    """
    # ✅ Always prefer broker outcome reason for cancelled/rejected orders
    status0 = str(data.get("status", "")).upper()
    if status0 in {"OANDA_CANCELLED", "OANDA_REJECTED"}:
        oanda_r = _oanda_cancel_or_reject_reason(meta, data.get("oanda_response", ""))
        if oanda_r:
            return oanda_r, 0.0

    gpt_reason, confidence = None, None

    if isinstance(meta, dict) and meta.get("rule") == "DEGRADATION_EXIT":
        return _format_degradation_exit(meta), meta.get("confidence", 1.0)

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
            limit = meta.get("limit", meta.get("max_spread_pips", "N/A")) if isinstance(meta, dict) else "N/A"
            if spread is not None:
                gpt_reason = f"Spread too wide: {float(spread):.1f} pips > limit {limit}"
            else:
                gpt_reason = f"Spread too wide (limit {limit})"

        elif status == "RULE_BLOCKED_SPREAD_UNKNOWN":
            gpt_reason = "Spread unknown: pricing fetch failed."

        elif status == "RULE_BLOCKED_CHOP":
            sep = meta.get("ema_sep") if isinstance(meta, dict) else None
            req = meta.get("required") if isinstance(meta, dict) else None
            if sep is not None and req is not None:
                gpt_reason = f"Chop Gate: EMA separation {float(sep):.2f} pips < required {float(req):.2f} pips."
            else:
                gpt_reason = "Blocked by Chop Gate (low momentum)."

        elif status == "RULE_BLOCKED_SUNDAY_OPEN":
            h = meta.get("hour_utc", "?") if isinstance(meta, dict) else "?"
            gpt_reason = f"Sunday Liquidity Filter: Blocked (Hour {h} UTC)."

        elif status == "RULE_BLOCKED_OVEREXTENDED":
            gpt_reason = "Blocked: overextended entry (mean-reversion risk)."

        elif status == "RULE_BLOCKED_ROLLOVER_OPEN":
            gpt_reason = _format_rollover_reason(meta if isinstance(meta, dict) else {})

        elif status == "RISK_BLOCKED_ROLLOVER":
            gpt_reason = _format_rollover_reason(meta if isinstance(meta, dict) else {})

        elif status == "COOLDOWN_ACTIVE":
            gpt_reason = "Cooldown active: suppressed re-entry/pyramiding."

        elif status == "STALE_ALERT_IGNORED":
            age = meta.get("age_sec") if isinstance(meta, dict) else None
            mx = meta.get("max_age") if isinstance(meta, dict) else None
            if age is not None and mx is not None:
                gpt_reason = f"Stale alert ignored: age {float(age):.1f}s > max {float(mx):.1f}s"
            else:
                gpt_reason = "Stale alert ignored."

        elif status == "DUPLICATE_IGNORED":
            gpt_reason = "Duplicate alert_id ignored."

        elif status == "SAFETY_BLOCKED_FLIP":
            if isinstance(meta, dict) and meta.get("reason"):
                gpt_reason = str(meta.get("reason"))
            else:
                gpt_reason = "Safety Block: Cannot flip Net Long/Short instantly."

        elif status.startswith("RISK_BLOCKED_"):
            gpt_reason = status

        elif status == "RISK_CALC_ERROR":
            gpt_reason = "Risk sizing calculation error."

        elif status == "OANDA_ERROR":
            gpt_reason = "OANDA returned error."

        elif status == "NETWORK_ERROR":
            gpt_reason = "Network error when calling OANDA."

        else:
            gpt_reason = "No reason found in meta."

    return gpt_reason, (confidence if confidence is not None else "N/A")


def print_execution(title, data, meta):
    reason, conf = format_reason(data, meta)
    ts = int(data.get("ts", 0) or 0)
    l_str, u_str = _fmt_ts_both(ts)

    spread_val = data.get("spread_pips", None)
    spread_disp = "N/A"
    try:
        if spread_val is not None:
            spread_disp = f"{float(spread_val):.1f}"
    except Exception:
        spread_disp = str(spread_val)

    conf_disp = "N/A"
    try:
        if conf is not None and conf != "N/A":
            conf_disp = f"{float(conf):.2f}"
        else:
            conf_disp = "N/A"
    except Exception:
        conf_disp = str(conf)

    print(
        "\n" + "=" * 60 +
        f"\n {title}: {str(data.get('action', '')).upper()} on {data['instrument']} ({data.get('status', 'N/A')})\n" +
        "=" * 60
    )
    print(
        f"TIME (LOCAL): {l_str}\n"
        f"TIME (UTC):   {u_str} UTC\n"
        f"CONFIDENCE:   {conf_disp}\n"
        f"SPREAD:       {spread_disp}\n\n"
        f"REASONING:\n" + "-" * 20 +
        f"\n{textwrap.fill(str(reason), width=80)}\n" +
        "-" * 20
    )


# --------------------------
# Optional: OANDA trades list via oandapyV20
# --------------------------
def get_oanda_trades():
    if not OANDA_AVAILABLE or not ACCESS_TOKEN:
        return None
    try:
        client = API(access_token=ACCESS_TOKEN)
        r = trades.TradesList(accountID=ACCOUNT_ID, params={"state": "OPEN"})
        client.request(r)
        return r.response.get("trades", [])
    except Exception as e:
        print(f"\n[!] OANDA API Error: {e}")
        return None


# --------------------------
# Main
# --------------------------
def main():
    if not os.path.exists(DB_PATH):
        print(f"[!] DB not found: {DB_PATH}")
        return

    # Refresh internal trade_state so portfolio isn't stale
    refresh_trade_state_from_oanda(ACCOUNT_CCY)

    conn = db_conn()
    cur = conn.cursor()

    cutoff = int(time.time() - 86400)
    cur.execute(
        "SELECT e.* FROM executions e "
        "JOIN (SELECT instrument, MAX(ts) as max_ts FROM executions WHERE ts > ? GROUP BY instrument) latest "
        "ON e.instrument = latest.instrument AND e.ts = latest.max_ts "
        "WHERE e.instrument != 'XAU_USD' "
        "ORDER BY e.instrument ASC",
        (cutoff,)
    )
    recent = cur.fetchall()

    cur.execute("SELECT * FROM trade_state WHERE is_open = 1 ORDER BY instrument ASC")
    active = cur.fetchall()
    conn.close()

    print("\n" + "#" * 60 + "\n SYSTEM STATUS (Last Report per Pair)\n" + "#" * 60)
    if not recent:
        print("No activity.")
    else:
        for row in recent:
            d = dict(row)
            m = _safe_json_loads(d["meta"]) if d.get("meta") else {}
            print_execution(f"LATEST LOG: {d['instrument']}", d, m)

    print("\n\n" + "#" * 60 + "\n 🤖 ACTIVE PORTFOLIO (Bot Internal State)\n" + "#" * 60)

    if not active:
        print("\n[FLAT] No open trades.")
    else:
        for row in active:
            t = dict(row)
            if t["instrument"] == "XAU_USD":
                continue

            dur = "N/A"
            if t.get("entry_time_ms"):
                try:
                    dur = str(datetime.datetime.now() - datetime.datetime.fromtimestamp(t["entry_time_ms"] / 1000.0)).split(".")[0]
                except Exception:
                    dur = "N/A"

            pips = None
            try:
                if t.get("entry_price") and t.get("last_mark_price") and t.get("side"):
                    pips = pips_from_entry(
                        t["instrument"],
                        t["side"],
                        Decimal(str(t["entry_price"])),
                        Decimal(str(t["last_mark_price"]))
                    )
            except Exception:
                pips = None

            print(
                f"\n>>> NET POSITION: {t['instrument']}"
                f"\n    SIDE:        {str(t.get('side', '')).upper()}"
                f"\n    NET UNITS:   {t.get('units', 'N/A')}"
                f"\n    AVG ENTRY:   {t.get('entry_price', 'N/A')}"
                f"\n    MARK PRICE:  {t.get('last_mark_price', 'N/A')}"
            )
            if pips is not None:
                print(f"    PIPS (Est):  {pips:+.1f}")
            print(
                f"    DURATION:    {dur}"
                f"\n    UPL (Est):   {t.get('unrealized_pl_home', 'N/A')} {ACCOUNT_CCY}\n" +
                "-" * 40
            )

    if OANDA_AVAILABLE:
        print("\n\n" + "#" * 60 + "\n 🏦 LIVE OANDA TICKETS (Broker Source of Truth)\n" + "#" * 60)
        live = get_oanda_trades()
        if live:
            print(f"\n{'TICKET':<10} {'INSTRUMENT':<12} {'UNITS':<10} {'ENTRY':<10} {'UPL':<10}\n" + "-" * 55)
            for t in live:
                print(
                    f"{t.get('id'):<10} "
                    f"{t.get('instrument'):<12} "
                    f"{t.get('currentUnits'):<10} "
                    f"{t.get('price'):<10} "
                    f"{t.get('unrealizedPL'):<10}"
                )
        elif live == []:
            print("\n[FLAT] No open tickets.")


if __name__ == "__main__":
    main()
