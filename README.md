# Micro-Range Statistical Trading Platform (Python Backtester)

Quick-start scaffold to research the micro-range median-band strategy at a **portfolio level**.

## Features
- Universe filter: price between $1 and $3
- Rolling **30-day median** and ± band entry/exit thresholds
- Daily **basket selection** (up to N names/day) with equal-weight capital allocation
- Basic fills:
  - **Entry**: next day's open after a signal
  - **Take-profit exit**: if next day's high >= target (band upper), exit at target
  - **Time-based exit**: optional **max_hold_days**; exits at next close when reached
- Costs: commission (%) + slippage (%), applied to both sides
- Metrics: total return, CAGR, Sharpe (daily), max drawdown, trades, win rate
- Example synthetic dataset included for a quick sanity-check run

> This is a **clean baseline**. Swap the synthetic CSVs in `data/` with your real OHLCV (one CSV per ticker).

## Project layout
```
micro_range_backtester/
  ├── data/                 # Put your CSVs here (one file per ticker)
  │   └── DEMO_*.csv        # Synthetic examples (ticker, date, ohlcv)
  ├── output/               # Backtest reports/plots
  ├── src/
  │   ├── data.py           # CSV loader + format docs
  │   ├── strategy.py       # Signals + band targets
  │   ├── backtester.py     # Portfolio simulator
  │   └── metrics.py        # Performance metrics
  ├── main.py               # CLI runner with default params
  └── requirements.txt
```

## Expected CSV format
- One file per ticker (e.g., `ABCD.csv`), with columns:
  - `date` (YYYY-MM-DD), `open`, `high`, `low`, `close`, `volume`, `ticker`
- Sorted ascending by `date`.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --data_dir data --out_dir output --band_pct 3.0 --n_names 10
```

## Parameters (main.py)
- `--band_pct`: e.g., 3.0 means ±3% from 30D median
- `--lookback`: rolling median lookback (default 30)
- `--n_names`: max names to trade per day (default 10)
- `--init_capital`: starting cash (default 20000)
- `--commission_pct`: per-side commission (default 0.02%) -> 0.0002
- `--slippage_pct`: per-side slippage (default 0.05%) -> 0.0005
- `--max_hold_days`: if >0, exit at close once this holding period is reached
- `--min_price`/`--max_price`: price universe filter (default 1–3)

## Notes
- This uses **daily bars**. Intraday nuances (partial fills, gap logic, etc.) are simplified.
- Replace `data/DEMO_*.csv` with real data to start proper testing.
- For production, consider richer execution modeling and borrow/fee costs for low-priced names.
