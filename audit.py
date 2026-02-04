import sqlite3
import json
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "bot.db")

# True  = trading day in UTC (recommended for FX)
# False = trading day in your local timezone
USE_UTC_DAY = True

# =========================
# ROLLOVER WINDOW (UTC)
# =========================
ROLLOVER_START_UTC = os.getenv("ROLLOVER_START_UTC", "21:55")  # HH:MM
ROLLOVER_END_UTC   = os.getenv("ROLLOVER_END_UTC", "22:45")    # HH:MM
# =========================

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

def _start_of_today_ts():
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

def _parse_hhmm(hhmm: str) -> int:
    """
    Returns minutes from midnight for an HH:MM string.
    """
    try:
        parts = hhmm.strip().split(":")
        h = int(parts[0])
        m = int(parts[1])
        h = max(0, min(23, h))
        m = max(0, min(59, m))
        return h * 60 + m
    except Exception:
        return 22 * 60

def _is_within_rollover(dt_utc: datetime.datetime) -> bool:
    """
    dt_utc must be timezone-aware UTC datetime.
    Handles windows that might wrap midnight (unlikely here but robust).
    """
    tmin = dt_utc.hour * 60 + dt_utc.minute
    start = _parse_hhmm(ROLLOVER_START_UTC)
    end = _parse_hhmm(ROLLOVER_END_UTC)

    if start <= end:
        return start <= tmin <= end
    return (tmin >= start) or (tmin <= end)

def get_pnl(action: str, status: str, oanda_response: str) -> str:
    if action.lower() != "close" or status != "OK" or not oanda_response:
        return "-"
    try:
        data = json.loads(oanda_response)
        fill = (
            data.get("orderFillTransaction") or
            data.get("longOrderFillTransaction") or
            data.get("shortOrderFillTransaction")
        )
        if fill:
            val = float(fill.get("pl", 0.0))
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

def _format_rollover_reason(meta: dict) -> str:
    start = meta.get("rollover_start_utc") or os.getenv("ROLLOVER_START_UTC", ROLLOVER_START_UTC)
    end = meta.get("rollover_end_utc") or os.getenv("ROLLOVER_END_UTC", ROLLOVER_END_UTC)
    return f"Rollover window: OPEN blocked ({start}–{end} UTC)."

def derive_reason(status: str, spread_pips, meta: dict) -> str:
    if isinstance(meta, dict) and meta.get("rule") == "DEGRADATION_EXIT":
        return _format_degradation_exit(meta)

    s = (status or "").upper()

    if s == "RULE_BLOCKED_ROLLOVER_OPEN":
        return _format_rollover_reason(meta if isinstance(meta, dict) else {})

    if s == "RISK_BLOCKED_ROLLOVER":
        return _format_rollover_reason(meta if isinstance(meta, dict) else {})

    if s == "RULE_BLOCKED_SPREAD":
        limit = meta.get("limit", meta.get("max_spread_pips", "N/A")) if isinstance(meta, dict) else "N/A"
        if spread_pips is not None:
            try:
                return f"Spread too wide: {float(spread_pips):.1f} pips > limit {limit}"
            except Exception:
                return f"Spread too wide: {spread_pips} > limit {limit}"
        return f"Spread too wide (limit {limit})"

    if s == "RULE_BLOCKED_SPREAD_UNKNOWN":
        return "Spread unknown: pricing fetch failed."

    if s == "RULE_BLOCKED_CHOP":
        if isinstance(meta, dict):
            sep = meta.get("ema_sep")
            req = meta.get("required")
            if isinstance(sep, (int, float)) and isinstance(req, (int, float)):
                return f"Chop Gate: EMA sep {sep:.2f} pips < required {req:.2f}"
        return "Blocked by Chop Gate (low momentum)."

    if s == "RULE_BLOCKED_SUNDAY_OPEN":
        if isinstance(meta, dict):
            h = meta.get("hour_utc", "?")
            return f"Sunday Liquidity Filter: Blocked (Hour {h} UTC)"
        return "Sunday Liquidity Filter: Blocked due to time/day."

    if s == "RULE_BLOCKED_OVEREXTENDED":
        if isinstance(meta, dict):
            tb = meta.get("trend_bias")
            oe = meta.get("is_overextended")
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
        return "OANDA returned error."
    if s == "NETWORK_ERROR":
        return "Network error when calling OANDA."

    return "No reason found in meta."

def extract_gpt_fields(meta: dict):
    """
    Returns (confidence, reason) from either:
      - meta["gpt"] (nested)
      - meta["reason"]/meta["confidence"] (flat)
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

    cur.execute(
        "SELECT ts, instrument, action, status, spread_pips, meta, oanda_response "
        "FROM executions WHERE ts >= ? ORDER BY ts DESC",
        (start_ts,)
    )

    day_label = "UTC" if USE_UTC_DAY else "LOCAL"
    print(f"\n########## AUDIT — TODAY ({day_label}) ##########\n")

    print(
        f"Rollover window (UTC): {ROLLOVER_START_UTC}–{ROLLOVER_END_UTC}  "
        f"(flags rows with 'R' in the ROLL column)\n"
    )

    print(f"{'LOCAL':<8} | {'UTC':<8} | {'ROLL':<4} | {'PAIR':<8} | {'ACTION':<7} | {'P/L':<9} | {'STATUS':<30} | {'SPRD':>6} | {'CONF':>5} | REASON")
    print("-" * 185)

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

        roll_mark = "R" if _is_within_rollover(dt_utc) else ""

        pair = row["instrument"] or "N/A"
        action = (row["action"] or "N/A").upper()
        status = row["status"] or "N/A"
        pnl = get_pnl(action, status, row["oanda_response"])

        spread = row["spread_pips"]
        spread_disp = f"{spread:.1f}" if isinstance(spread, (int, float)) else "N/A"

        meta = _safe_json_loads(row["meta"]) if row["meta"] else {}

        conf, reason = extract_gpt_fields(meta)

        # ✅ Always prefer broker reason for CANCELLED/REJECTED (even if GPT reason exists)
        s = str(status or "").upper()
        if s in {"OANDA_CANCELLED", "OANDA_REJECTED"}:
            oanda_r = _oanda_cancel_or_reject_reason(meta, row["oanda_response"])
            if oanda_r:
                reason = oanda_r
                conf = 0.0  # optional: broker outcome, not GPT confidence

        # Degradation exit meta overrides everything
        if isinstance(meta, dict) and meta.get("rule") == "DEGRADATION_EXIT":
            reason = _format_degradation_exit(meta)
            conf = meta.get("confidence", conf)

        # If no reason, derive from status/meta
        if not reason:
            reason = derive_reason(status, spread, meta)

        conf_disp = "N/A"
        try:
            conf_disp = f"{float(conf):.2f}" if conf is not None else "N/A"
        except Exception:
            conf_disp = str(conf) if conf is not None else "N/A"

        reason_s = str(reason)
        reason_display = (reason_s[:110] + "..") if len(reason_s) > 110 else reason_s

        print(
            f"{local_str:<8} | {utc_str:<8} | {roll_mark:<4} | {pair:<8} | {action:<7} | {pnl:<9} | "
            f"{status:<30} | {spread_disp:>6} | {conf_disp:>5} | {reason_display}"
        )

    conn.close()

if __name__ == "__main__":
    show_today_decisions()
