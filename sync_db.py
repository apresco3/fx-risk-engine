import sqlite3
import os
from dotenv import load_dotenv
from collections import defaultdict
import oandapyV20
import oandapyV20.endpoints.trades as trades
from oandapyV20 import API

load_dotenv()

# --- CONFIGURATION ---
DB_PATH = "bot.db"
ACCESS_TOKEN = os.getenv("OANDA_TOKEN")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

def get_oanda_positions():
    """
    Fetches all OPEN trades from Oanda and aggregates them by Instrument.
    Returns a dict: { 'USD_JPY': {'units': 3105, 'side': 'BUY'}, ... }
    """
    if not ACCESS_TOKEN or not ACCOUNT_ID:
        print("[!] Error: OANDA credentials missing from .env")
        return {}

    try:
        client = API(access_token=ACCESS_TOKEN)
        # Fetch all currently OPEN trades
        r = trades.TradesList(accountID=ACCOUNT_ID, params={"state": "OPEN"})
        client.request(r)
        data = r.response.get("trades", [])
    except Exception as e:
        print(f"[!] Critical Error connecting to Oanda: {e}")
        return {}

    # Aggregate individual tickets into a Net Position per instrument
    aggregated = defaultdict(lambda: {'units': 0, 'side': None})

    for t in data:
        instrument = t['instrument']
        units = int(t['currentUnits'])
        
        # Oanda V20: positive units = Long, negative units = Short (usually check initialUnits)
        # However, 'currentUnits' is typically unsigned in some views, but signed in others.
        # Let's rely on the explicit sign if present, or infer from initialUnits.
        initial = int(t.get('initialUnits', 0))
        
        # Determine direction of this specific ticket
        if initial > 0:
            ticket_side = "BUY"
            raw_units = abs(units)
        else:
            ticket_side = "SELL"
            raw_units = abs(units)

        # Update aggregation
        aggregated[instrument]['units'] += raw_units
        
        # Set side (Simplified: Assuming bot doesn't hedge/hold mixed modes)
        if aggregated[instrument]['side'] is None:
            aggregated[instrument]['side'] = ticket_side
        elif aggregated[instrument]['side'] != ticket_side:
             print(f"[!] WARNING: Mixed positions found for {instrument}. Sync may be imperfect.")

    return dict(aggregated)

def sync_database():
    """
    Compares Oanda reality vs Local DB and fixes discrepancies.
    """
    if not os.path.exists(DB_PATH):
        print(f"[!] Database not found at {DB_PATH}")
        return

    print("--- 🔄 STARTING DB SYNC ---")
    
    # 1. Get Reality (Oanda)
    real_positions = get_oanda_positions()
    print(f"[Broker] Found active positions on: {list(real_positions.keys())}")

    # 2. Get Fiction (Local DB)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT instrument, units, side, is_open FROM trade_state WHERE is_open = 1")
    db_rows = cur.fetchall()
    
    updates_made = False

    # 3. Match DB rows against Reality
    checked_instruments = set()

    for row in db_rows:
        inst = row['instrument']
        db_units = int(row['units'])
        checked_instruments.add(inst)

        if inst not in real_positions:
            # CASE 1: DB says Open, Oanda says Closed (e.g., TP hit)
            print(f"[FIX] {inst}: DB has {db_units} units, Oanda has 0. -> MARKING CLOSED.")
            cur.execute("UPDATE trade_state SET is_open = 0, units = 0 WHERE instrument = ?", (inst,))
            updates_made = True
        else:
            # CASE 2: DB says X, Oanda says Y (e.g., Partial Close)
            real_units = real_positions[inst]['units']
            if db_units != real_units:
                diff = db_units - real_units
                print(f"[FIX] {inst}: DB has {db_units}, Oanda has {real_units} (Diff: {diff}). -> UPDATING UNITS.")
                cur.execute("UPDATE trade_state SET units = ? WHERE instrument = ?", (real_units, inst))
                updates_made = True
            else:
                print(f"[OK]  {inst}: Synced at {real_units} units.")

    # 4. (Optional) Check for 'Ghost' trades (Oanda has it, DB doesn't)
    # This script focuses on fixing existing DB entries, but we can warn.
    for inst in real_positions:
        if inst not in checked_instruments:
            print(f"[!] WARNING: Oanda has {inst} open, but Bot DB has no record of it.")

    conn.commit()
    conn.close()
    
    print("--- ✅ SYNC COMPLETE ---")
    if updates_made:
        print("Database updated. Run 'python view_latest.py' to verify.")
    else:
        print("No changes were needed.")

if __name__ == "__main__":
    sync_database()