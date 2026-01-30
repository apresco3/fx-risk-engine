import sqlite3
import json
import textwrap

def view_latest_reason():
    conn = sqlite3.connect("bot.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get the most recent execution
    cur.execute("SELECT ts, instrument, action, status, meta FROM executions ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    
    if row:
        print("\n" + "="*60)
        print(f" LATEST: {row['action'].upper()} on {row['instrument']} ({row['status']})")
        print("="*60)
        
        if not row["meta"]:
            print("\n[!] No reasoning data available (likely a system error or duplicate).")
            conn.close()
            return

        # Parse the JSON meta
        try:
            meta = json.loads(row["meta"])
            gpt_data = meta.get("gpt", meta) 
            
            # Handle cases where gpt_data might be a string (rare) or dict
            if isinstance(gpt_data, str):
                print(f"\nRaw GPT Output: {gpt_data}")
            else:
                reason = gpt_data.get("reason", "No reason text found.")
                confidence = gpt_data.get("confidence", "N/A")
                
                print(f"\nCONFIDENCE: {confidence}")
                print("\nFULL REASONING:")
                print("-" * 20)
                print(textwrap.fill(str(reason), width=80))
                print("-" * 20)
            
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            print("Raw Meta:", row["meta"])
    else:
        print("No executions found in database.")

    conn.close()

if __name__ == "__main__":
    view_latest_reason()