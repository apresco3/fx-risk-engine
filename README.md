# OANDA Webhook Trading Bot

A Flask service that receives TradingView-style alerts, applies risk and execution rules, and places FX trades on OANDA. It keeps local state in SQLite, logs every decision, and can optionally ask GPT for a final go/no-go and sizing multiplier.

## What It Does

- Listens on a '/webhook' endpoint for alerts ('observe', 'open', 'close', 'close_all').
- Enforces risk limits, max spread, session windows, rollover windows, and position caps.
- Sizes trades based on account NAV and 'RISK_PCT' (with hard caps).
- Tracks executions and trade state in 'bot.db' and appends to 'trades.log'.
- Optional GPT gatekeeper to reduce or veto entries.

## Quick Start

1. Create a virtual environment and install dependencies.
2. Create a '.env' with OANDA credentials and webhook key.
3. Run the server and send alerts.

'''bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py
'''

## Core Configuration (.env)

Minimum required:

- 'OANDA_TOKEN'
- 'OANDA_ACCOUNT_ID'
- 'OANDA_MODE' ('PRACTICE' or 'LIVE')
- 'WEBHOOK_KEY' (required query param for '/webhook')

Common optional:

- 'PORT' (default '8080')
- 'TRADING_ENABLED' ('true'/'false')
- 'PAIRS' (comma-separated allowlist)
- 'RISK_PCT', 'MAX_RISK_USD'
- 'MAX_UNITS', 'MAX_NET_UNITS', 'GLOBAL_MAX_UNITS'
- 'SL_PIPS', 'TP_PIPS', 'MAX_SL_PIPS', 'MAX_TP_PIPS'
- 'MAX_SPREAD_PIPS', 'MIN_EMA_SEP_PIPS', 'MACRO_BUFFER_PIPS'
- 'SESSION_FILTER_ENABLED', 'SESSION_START_UTC', 'SESSION_END_UTC'
- 'ROLLOVER_BLOCK_OPENS', 'ROLLOVER_START_UTC', 'ROLLOVER_END_UTC'

GPT gatekeeper (optional):

- 'GPT_ENABLED' ('true'/'false')
- 'OPENAI_API_KEY'
- 'GPT_MODEL' (default 'gpt-5-mini')
- 'GPT_MIN_CONFIDENCE'
- 'GPT_MAX_UNITS_MULT'

## Running

'''bash
python bot.py
'''

Health check:

'''bash
curl http://localhost:8080/health
'''

## Webhook API

Endpoint:

- 'POST /webhook?key=WEBHOOK_KEY'

Payload shape:

- 'alert_id' (string, required)
- 'action' (string, one of 'observe', 'open', 'close', 'close_all')
- 'pair' (string like 'EUR_USD' for 'observe', 'open', 'close')
- 'side' (string 'buy' or 'sell' for 'open')
- 'sl_pips', 'tp_pips' (integers, optional)
- 'features' (object, optional; used for observe/GPT decisions)

Example 'observe' payload:

'''json
{
  "alert_id": "EUR_USD:1700000000:obs",
  "action": "observe",
  "pair": "EUR_USD",
  "sl_pips": 25,
  "tp_pips": 50,
  "features": {
    "tf": "5",
    "time_ms": 1700000000000,
    "rsi": 52.3,
    "adx": 18.9,
    "atr_pips": 8.4,
    "ema_sep_pips": 2.1
  }
}
'''

Example 'open' payload:

'''json
{
  "alert_id": "EUR_USD:1700000000:open",
  "action": "open",
  "pair": "EUR_USD",
  "side": "buy",
  "sl_pips": 25,
  "tp_pips": 50,
  "features": {"tf": "5"}
}
'''

Example 'close' payload:

'''json
{
  "alert_id": "EUR_USD:1700000000:close",
  "action": "close",
  "pair": "EUR_USD"
}
'''

Example 'close_all' payload:

'''json
{
  "alert_id": "ALL:1700000000:close_all",
  "action": "close_all"
}
'''

Notes:

- Alerts are deduped by 'alert_id' for a short TTL.
- If 'TRADING_ENABLED=false', the endpoint returns 'DISABLED' and does not trade.

## TradingView Alerts

'tradingview.pine' emits 'observe' alerts every bar close with a packed 'features' payload and lookback candles. Use it as a baseline for alert formatting.

## Local Tools

- 'audit.py': prints a per-day execution summary with derived reasons.
- 'view_latest.py': inspects latest executions and optionally pulls live OANDA trade status.
- 'view_pnl.py': shows the last 5 realized closes from 'bot.db'.
- 'export_logs.py': exports all executions to 'audit_log_full.csv'.
- 'sync_db.py': reconciles 'trade_state' with live OANDA trades.
- 'debug_spread.py': prints live spreads using OANDA pricing.
- 'compare_models.py': quick before/after model switch report (edit 'SWITCH_TIME_STR').

## Data Files

- 'bot.db': SQLite database with 'alerts', 'executions', and 'trade_state'.
- 'trades.log': append-only event log.
- 'audit_log_full.csv': optional CSV export.

## Safety Notes

- This system can place live orders. Test in 'PRACTICE' mode first.
- Keep 'WEBHOOK_KEY' secret and rotate if exposed.
- Review and tune risk and session parameters before enabling 'TRADING_ENABLED'.

  https://nomuracampus.tal.net/vx/lang-en-GB/mobile-0/channel-1/appcentre-1/brand-4/user-773260/xf-1cc8fc8323e0/wid-6/spa-1/tmpwid-c566_a5728c35-a328-47e3-8e82-5286e2dfaf47/candidate/application/632229/opportunity
