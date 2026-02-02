import sqlite3
import json
import datetime

DB_PATH = "bot.db"

# True  = trading day in UTC (recommended for FX)
# False = trading day in your local timezone
USE_UTC_DAY = True


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {}


def _start_of_today_ts():
    """
    Returns UNIX timestamp for start of today (00:00)
    in either UTC or local time.
    """
    if USE_UTC_DAY:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        start_utc = datetime.datetime(
            now_utc.year, now_utc.month, now_utc.day,
            tzinfo=datetime.timezone.utc
        )
        return int(start_utc.timestamp())
    else:
        now_local = datetime.datetime.now()
        start_local = datetime.datetime(now_local.year, now_local.month, now_local.day)
        return int(start_local.timestamp())


def _fmt_num(x, fmt=":.2f"):
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


def get_pnl(action: str, status: str, oanda_response: str) -> str:
    """
    Extracts Realized P/L from the OANDA JSON response if available.
    Returns string representation (e.g. "-0.45" or "-")
    """
    if action.lower() != "close" or status != "OK" or not oanda_response:
        return "-"

    try:
        data = json.loads(oanda_response)
        # OANDA close responses usually contain fill transaction details
        fill = (
            data.get("orderFillTransaction") or 
            data.get("longOrderFillTransaction") or 
            data.get("shortOrderFillTransaction")
        )
        
        if fill:
            val = float(fill.get("pl", 0.0))
            # Format: explicit plus sign for profit, red/negative handling handled by caller context if needed
            return f"{val:+.2f}"
            
    except Exception:
        pass
    
    return "-"


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

    return ", ".join(parts)


def derive_reason(status: str, spread_pips, meta: dict) -> str:
    # --- NEW: Degradation exit rule ---
    if isinstance(meta, dict) and meta.get("rule") == "DEGRADATION_EXIT":
        return _format_degradation_exit(meta)

    s = (status or "").upper()

    if s == "RULE_BLOCKED_SPREAD":
        limit = "N/A"
        if isinstance(meta, dict):
            limit = meta.get("limit", meta.get("max_spread_pips", "N/A"))
        if spread_pips is not None:
            return f"Spread too wide: {spread_pips} pips > limit {limit}"
        return f"Spread too wide (limit {limit})"

    if s == "RULE_BLOCKED_SPREAD_UNKNOWN":
        return "Spread unknown: pricing fetch failed."

    # --- CHOP GATE ---
    if s == "RULE_BLOCKED_CHOP":
        if isinstance(meta, dict):
            sep = meta.get("ema_sep")
            req = meta.get("required")
            if isinstance(sep, (int, float)) and isinstance(req, (int, float)):
                return f"Chop Gate: EMA sep {sep:.2f} pips < required {req:.2f}"
        return "Blocked by Chop Gate (low momentum)."

    # --- SUNDAY FILTER ---
    if s == "RULE_BLOCKED_SUNDAY_OPEN":
        if isinstance(meta, dict):
            h = meta.get("hour_utc", "?")
            return f"Sunday Liquidity Filter: Blocked (Hour {h} UTC)"
        return "Sunday Liquidity Filter: Blocked due to time/day."

    if s == "RULE_BLOCKED_OVEREXTENDED":
        if isinstance(meta, dict):
            tb = meta.get("trend_bias")
            oe = meta.get("is_overextended")
            if tb is not None or oe is not None:
                return f"Overextended entry blocked (trend_bias={tb}, is_overextended={oe})"
        return "Overextended entry blocked (mean-reversion risk)."

    if s == "COOLDOWN_ACTIVE":
        return "Cooldown active: suppressed re-entry/pyramiding."

    if s == "STALE_ALERT_IGNORED":
        if isinstance(meta, dict) and "age_sec" in meta and "max_age" in meta:
            return f"Stale alert ignored: age {meta['age_sec']:.1f}s > max {meta['max_age']}s"
        return "Stale alert ignored."

    if s == "DUPLICATE_IGNORED":
        return "Duplicate alert_id ignored."

    if s == "NAV_FETCH_FAIL":
        return "NAV fetch failed before placing order."

    if s == "STATE_FETCH_FAIL":
        return "Failed to fetch open positions/state."

    if s == "SAFETY_BLOCKED_FLIP":
        if isinstance(meta, dict) and "reason" in meta:
            return str(meta["reason"])
        return "Safety Block: Cannot flip Net Long/Short instantly."

    if s.startswith("RISK_BLOCKED_"):
        return s

    if s == "RISK_CALC_ERROR":
        return "Risk sizing calculation error."

    if s == "OANDA_ERROR":
        return "OANDA returned error (see executions.oanda_response)."

    if s == "NETWORK_ERROR":
        return "Network error when calling OANDA."

    return "No reason found in meta."


def extract_gpt_fields(meta: dict):
    """
    Returns (confidence, reason) if present.
    """
    if not isinstance(meta, dict):
        return None, None

    if "gpt" in meta and isinstance(meta["gpt"], dict):
        g = meta["gpt"]
        return g.get("confidence"), g.get("reason")

    if "reason" in meta or "confidence" in meta:
        return meta.get("confidence"), meta.get("reason")

    return None, None


def show_today_decisions():
    start_ts = _start_of_today_ts()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- FIXED: Added 'FROM executions' which was missing in your paste ---
    cur.execute(
        """
        SELECT ts, instrument, action, status, spread_pips, meta, oanda_response
        FROM executions
        WHERE ts >= ?
        ORDER BY ts DESC
        """,
        (start_ts,),
    )

    day_label = "UTC" if USE_UTC_DAY else "LOCAL"
    print(f"\n########## AUDIT — TODAY ({day_label}) ##########\n")
    # Adjusted formatting to accommodate P/L column
    print(f"{'LOCAL':<8} | {'UTC':<8} | {'PAIR':<8} | {'ACTION':<7} | {'P/L':<9} | {'STATUS':<20} | {'SPRD':>6} | {'CONF':>5} | REASON")
    print("-" * 155)

    rows = cur.fetchall()
    if not rows:
        print("No executions today.")
        conn.close()
        return

    for row in rows:
        ts_int = int(row["ts"] or 0)

        dt_local = datetime.datetime.fromtimestamp(ts_int)
        dt_utc = datetime.datetime.fromtimestamp(ts_int, tz=datetime.timezone.utc)

        local_str = dt_local.strftime("%H:%M:%S")
        utc_str = dt_utc.strftime("%H:%M:%S")

        pair = row["instrument"] or "N/A"
        action = (row["action"] or "N/A").upper()
        status = row["status"] or "N/A"
        
        pnl = get_pnl(action, status, row["oanda_response"])

        spread = row["spread_pips"]
        spread_disp = f"{spread:.1f}" if isinstance(spread, (int, float)) else "N/A"

        meta = _safe_json_loads(row["meta"]) if row["meta"] else {}

        conf, reason = extract_gpt_fields(meta)

        if not reason:
            reason = derive_reason(status, spread, meta)
        else:
            if isinstance(meta, dict) and meta.get("rule") == "DEGRADATION_EXIT":
                reason = _format_degradation_exit(meta)

        conf_disp = f"{float(conf):.2f}" if conf is not None else "N/A"

        reason_s = str(reason)
        # Truncate to keep table clean
        reason_display = (reason_s[:85] + "..") if len(reason_s) > 85 else reason_s

        print(f"{local_str:<8} | {utc_str:<8} | {pair:<8} | {action:<7} | {pnl:<9} | {status:<25} | {spread_disp:>6} | {conf_disp:>5} | {reason_display}")

    conn.close()


if __name__ == "__main__":
    show_today_decisions()