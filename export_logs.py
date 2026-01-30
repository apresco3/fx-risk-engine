import sqlite3
import pandas as pd
import json
import datetime

# Connect to the database
conn = sqlite3.connect("bot.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Select all executions, newest first
cur.execute("SELECT * FROM executions ORDER BY ts DESC")
rows = cur.fetchall()

data = []

print(f"Found {len(rows)} execution records. Processing...")

for row in rows:
    # Convert timestamp to readable date
    ts = datetime.datetime.fromtimestamp(row["ts"]).strftime('%Y-%m-%d %H:%M:%S')
    
    # Parse the metadata JSON to get the full reason and confidence
    reason_text = ""
    confidence_val = ""
    
    try:
        if row["meta"]:
            meta_json = json.loads(row["meta"])
            # Sometimes reasoning is nested under "gpt", sometimes it's flat
            gpt_data = meta_json.get("gpt", meta_json)
            
            if isinstance(gpt_data, dict):
                reason_text = gpt_data.get("reason", "")
                confidence_val = gpt_data.get("confidence", "")
    except Exception:
        reason_text = "Error parsing JSON"

    # Build the row for the CSV
    data.append({
        "Time": ts,
        "Pair": row["instrument"],
        "Action": row["action"],
        "Status": row["status"],
        "Side": row["side"] if row["side"] else "",
        "Units": row["units"] if row["units"] else "",
        "Confidence": confidence_val,
        "Reason": reason_text,
        "SL_Pips": row["sl_pips"],
        "TP_Pips": row["tp_pips"],
        "Net_Pos_Before": row["current_net"]
    })

# Close DB connection
conn.close()

# Convert to DataFrame and Export to CSV
df = pd.DataFrame(data)
filename = "audit_log_full.csv"
df.to_csv(filename, index=False)

print(f"✅ Successfully exported to {filename}")