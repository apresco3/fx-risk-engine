import sqlite3
import json
import textwrap

def view_latest_reason():
    conn = sqlite3.connect("bot.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get the most recent execution
    cur.execute("SELECT ts, instrument, action, meta FROM executions ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    
    if row:
        print("\n" + "="*60)
        print(f" LATEST ACTION: {row['action'].upper()} on {row['instrument']}")
        print("="*60)
        
        # Parse the JSON meta
        try:
            meta = json.loads(row["meta"])
            # The reasoning is usually nested inside a "gpt" key, 
            # but sometimes flat depending on the execution path.
            gpt_data = meta.get("gpt", meta) 
            
            reason = gpt_data.get("reason", "No reason found in meta.")
            confidence = gpt_data.get("confidence", "N/A")
            
            print(f"\nCONFIDENCE: {confidence}")
            print("\nFULL REASONING:")
            print("-" * 20)
            # Wrap text nicely so it's readable in terminal
            print(textwrap.fill(reason, width=80))
            print("-" * 20)
            
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            print("Raw Meta:", row["meta"])
    else:
        print("No executions found in database.")

    conn.close()

if __name__ == "__main__":
    view_latest_reason()