import sqlite3
import json

DB_PATH = "bot.db"

def show_recent_pnl():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get the last 5 CLOSE actions that were successful (Status=OK)
    cur.execute("""
        SELECT ts, instrument, oanda_response 
        FROM executions 
        WHERE action = 'close' AND status = 'OK'
        ORDER BY ts DESC 
        LIMIT 5
    """)
    
    rows = cur.fetchall()
    
    print(f"{'PAIR':<8} | {'TYPE':<15} | {'P/L (Home Ccy)':<15} | {'PRICE':<10}")
    print("-" * 60)

    for row in rows:
        pair = row["instrument"]
        raw_resp = row["oanda_response"]
        
        try:
            data = json.loads(raw_resp)
            
            # OANDA Close Response usually has these keys for the fill
            # It could be 'longOrderFillTransaction' (if we closed a Long)
            # or 'shortOrderFillTransaction' (if we closed a Short)
            fill_tx = data.get("longOrderFillTransaction") or data.get("shortOrderFillTransaction") or data.get("orderFillTransaction")
            
            if fill_tx:
                pl = fill_tx.get("pl", "0.0")
                price = fill_tx.get("price", "0.0")
                # Determine type based on which key existed or the units
                # If we closed a short, we bought back, so units will be positive in the fill, 
                # but let's just label it "CLOSE"
                print(f"{pair:<8} | {'CLOSE_FILL':<15} | {pl:<15} | {price:<10}")
            else:
                # Sometimes it's a 'marketOrderTransaction' if it wasn't a strict position close
                print(f"{pair:<8} | {'(No Fill Data)':<15} | {'N/A':<15} | {'-'}")
                
        except Exception as e:
            print(f"{pair:<8} | {'PARSE_ERROR':<15} | {str(e):<15}")

    conn.close()

if __name__ == "__main__":
    show_recent_pnl()