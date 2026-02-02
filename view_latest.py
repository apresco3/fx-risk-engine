import sqlite3
import json
import os
import textwrap

DB_PATH = "bot.db"

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {}

def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM executions ORDER BY ts DESC LIMIT 2")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No executions found.")
        return

    for i, row in enumerate(rows):
        data = dict(row)

        meta = _safe_json_loads(data["meta"]) if data.get("meta") else {}

        # Default outputs
        gpt_reason = "No reason text found."
        confidence = "N/A"

        # 1) OPEN trades often store meta={"gpt": {...}, "features": {...}}
        if isinstance(meta, dict) and "gpt" in meta and isinstance(meta["gpt"], dict):
            gpt_reason = meta["gpt"].get("reason", gpt_reason)
            confidence = meta["gpt"].get("confidence", confidence)

        # 2) HOLD/CLOSE often store meta flat: {"reason": "...", "confidence": 0.xx}
        elif isinstance(meta, dict) and "reason" in meta:
            gpt_reason = meta.get("reason", gpt_reason)
            confidence = meta.get("confidence", confidence)

        # 3) RULE BLOCKS: you often store meta like {"limit": 60.0} (no "reason")
        # Add a helpful fallback based on status + known keys.
        else:
            status = str(data.get("status", "")).upper()
            spread = data.get("spread_pips", None)
            if status == "RULE_BLOCKED_SPREAD":
                limit = meta.get("limit", "N/A") if isinstance(meta, dict) else "N/A"
                if spread is not None:
                    gpt_reason = f"Spread too wide: {spread} pips > limit {limit}"
                else:
                    gpt_reason = f"Spread too wide (limit {limit})"
            elif status == "RULE_BLOCKED_SPREAD_UNKNOWN":
                gpt_reason = "Spread unknown: pricing fetch failed."
            elif status == "RULE_BLOCKED_OVEREXTENDED":
                gpt_reason = "Blocked: overextended entry (mean-reversion risk)."

        print("\n" + "="*60)
        print(f" LATEST #{i+1}: {data['action'].upper()} on {data['instrument']} ({data['status']})")
        print("="*60)
        print(f"CONFIDENCE: {confidence}")
        print(f"SPREAD: {data.get('spread_pips', 'N/A')}")
        print("\nFULL REASONING:")
        print("-" * 20)
        print(textwrap.fill(str(gpt_reason), width=80))
        print("-" * 20)

if __name__ == "__main__":
    main()
