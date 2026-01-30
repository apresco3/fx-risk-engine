import sqlite3
import json
import datetime

# Usage: python audit.py

def show_recent_decisions():
    conn = sqlite3.connect("bot.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Added STATUS column to the header
    print(f"{'TIME':<20} | {'PAIR':<8} | {'ACTION':<7} | {'STATUS':<25} | {'CONF':<4} | REASON")
    print("-" * 115)
    
    cur.execute("SELECT ts, instrument, action, status, meta FROM executions ORDER BY ts DESC LIMIT 15")
    
    for row in cur.fetchall():
        ts_str = datetime.datetime.fromtimestamp(row["ts"]).strftime('%H:%M:%S')
        pair = row["instrument"]
        action = row["action"]
        status = row["status"]  # New field
        
        # Extract GPT data from meta safely
        meta_json = row["meta"]
        meta = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except:
                meta = {}
        
        # Handle cases where 'gpt' key might be nested or flat
        gpt = meta.get("gpt", meta) if "gpt" in meta else meta
        
        conf = gpt.get("confidence", 0.0)
        reason = gpt.get("reason", "No reason provided")
        
        # Truncate reason for display
        reason_display = (reason[:50] + '..') if reason and len(reason) > 50 else reason
        
        print(f"{ts_str:<20} | {pair:<8} | {action:<7} | {status:<25} | {conf:<4.2f} | {reason_display}")

    conn.close()

if __name__ == "__main__":
    show_recent_decisions()