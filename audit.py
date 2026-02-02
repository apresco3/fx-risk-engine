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


def derive_reason(status: str, spread_pips, meta: dict) -> str:
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
    if not isinstance(meta, dict):
        return None, None

    # OPEN meta often: {"gpt": {...}, "features": {...}}
    if "gpt" in meta and isinstance(meta["gpt"], dict):
        g = meta["gpt"]
        return g.get("confidence"), g.get("reason")

    # HOLD/CLOSE meta often flat
    if "reason" in meta or "confidence" in meta:
        return meta.get("confidence"), meta.get("reason")

    return None, None


def show_today_decisions():
    start_ts = _start_of_today_ts()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT ts, instrument, action, status, spread_pips, meta
        FROM executions
        WHERE ts >= ?
        ORDER BY ts DESC
        """,
        (start_ts,),
    )

    day_label = "UTC" if USE_UTC_DAY else "LOCAL"
    print(f"\n########## AUDIT — TODAY ({day_label}) ##########\n")
    print(f"{'LOCAL':<8} | {'UTC':<8} | {'PAIR':<8} | {'ACTION':<7} | {'STATUS':<24} | {'SPRD':>6} | {'CONF':>5} | REASON")
    print("-" * 150)

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

        spread = row["spread_pips"]
        spread_disp = f"{spread:.1f}" if isinstance(spread, (int, float)) else "N/A"

        meta = _safe_json_loads(row["meta"]) if row["meta"] else {}

        conf, reason = extract_gpt_fields(meta)
        if not reason:
            reason = derive_reason(status, spread, meta)

        conf_disp = f"{float(conf):.2f}" if conf is not None else "N/A"
        reason_s = str(reason)
        reason_display = (reason_s[:90] + "..") if len(reason_s) > 90 else reason_s

        print(f"{local_str:<8} | {utc_str:<8} | {pair:<8} | {action:<7} | {status:<24} | {spread_disp:>6} | {conf_disp:>5} | {reason_display}")

    conn.close()


if __name__ == "__main__":
    show_today_decisions()
