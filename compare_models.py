import sqlite3
import datetime
import time

DB_PATH = "bot.db"

# --- CONFIGURATION ---
# REPLACE THIS WITH THE EXACT TIME YOU SWITCH MODELS (YYYY-MM-DD HH:MM)
# Example: If you switch tomorrow at 9:00 PM
SWITCH_TIME_STR = "2026-02-02 21:00" 

def get_stats(cur, start_ts, end_ts, label):
    # Get all trades opened in this window
    cur.execute("""
        SELECT count(*), instrument 
        FROM executions 
        WHERE action='open' AND status='OK' AND ts >= ? AND ts < ?
        GROUP BY instrument
    """, (start_ts, end_ts))
    
    trades = cur.fetchall()
    total_trades = sum([t[0] for t in trades])
    
    # Get P/L (Realized from closed trades in this window)
    # Note: This is an estimate based on your 'audit' logic. 
    # For exact P/L, we'd need to query the closed trades and sum their profit.
    # For now, we count activity.
    
    print(f"\n--- {label} REPORT ---")
    print(f"Time Window: {datetime.datetime.fromtimestamp(start_ts)} to {datetime.datetime.fromtimestamp(end_ts)}")
    print(f"Total Trades: {total_trades}")
    for t in trades:
        print(f"   - {t[1]}: {t[0]} trades")
        
    return total_trades

def main():
    try:
        switch_dt = datetime.datetime.strptime(SWITCH_TIME_STR, "%Y-%m-%d %H:%M")
        switch_ts = switch_dt.timestamp()
    except ValueError:
        print("Error: Please update SWITCH_TIME_STR in the script with format YYYY-MM-DD HH:MM")
        return

    # Calculate 24h Before and 24h After
    start_ts = switch_ts - 86400 # 24 hours back
    end_ts = switch_ts + 86400   # 24 hours forward
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("="*60)
    print(f"⚔️  MODEL BATTLE: GPT-5.2 vs GPT-5-mini")
    print("="*60)

    # Phase 1 Stats
    t1 = get_stats(cur, start_ts, switch_ts, "PHASE 1 (GPT-5.2)")
    
    # Phase 2 Stats
    t2 = get_stats(cur, switch_ts, end_ts, "PHASE 2 (GPT-5-mini)")

    # Cost Estimate (Based on current pricing)
    # 5.2 = ~$0.004 per call | Mini = ~$0.0002 per call
    # Assuming 1 trade = ~50 analysis calls (watching candles)
    est_cost_1 = (750 * 0.004) # Roughly 750 calls/day
    est_cost_2 = (750 * 0.0002)
    
    print("\n" + "="*60)
    print("💰 ESTIMATED COST COMPARISON")
    print("="*60)
    print(f"GPT-5.2 Cost:   ~${est_cost_1:.2f} (Est)")
    print(f"GPT-5-mini Cost:~${est_cost_2:.2f} (Est)")
    print(f"Savings:        ${est_cost_1 - est_cost_2:.2f}")

    conn.close()

if __name__ == "__main__":
    main()