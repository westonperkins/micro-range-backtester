import argparse
import os
import json
import matplotlib.pyplot as plt

from src.data import load_data_dir
from src.backtester import PortfolioBacktester, Costs, Params
from src.metrics import equity_metrics
from src.screener import rolling_window_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="output")

    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--band_pct", type=float, default=3.0, help="± band as percent (e.g., 3.0)")
    ap.add_argument("--fluct_pct", type=float, default=5.0, help="Max 30D window range as % of median (e.g., 5.0)")
    ap.add_argument("--min_hit_rate", type=float, default=0.15, help="Min 30D hit-rate touching ±band (0..1)")

    ap.add_argument("--n_names", type=int, default=10)
    ap.add_argument("--init_capital", type=float, default=20000.0)
    ap.add_argument("--commission_pct", type=float, default=0.02, help="per-side, percent")
    ap.add_argument("--slippage_pct", type=float, default=0.05, help="per-side, percent")
    ap.add_argument("--max_hold_days", type=int, default=1)
    ap.add_argument("--min_price", type=float, default=1.0)
    ap.add_argument("--max_price", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames = load_data_dir(args.data_dir)
    if not frames:
        print("No data found. Put CSVs in the data directory.")
        return

    costs = Costs(
        commission_pct=args.commission_pct / 100.0,
        slippage_pct=args.slippage_pct / 100.0,
    )
    params = Params(
        lookback=args.lookback,
        band_pct=args.band_pct / 100.0,
        n_names=args.n_names,
        min_price=args.min_price,
        max_price=args.max_price,
        max_hold_days=args.max_hold_days,
        fluct_pct=args.fluct_pct / 100.0,
        min_hit_rate=args.min_hit_rate,
    )

    bt = PortfolioBacktester(frames, init_capital=args.init_capital, costs=costs, params=params)
    equity_df, trades_df, preselect_df = bt.run()

    # Save core outputs
    equity_path = os.path.join(args.out_dir, "equity.csv")
    trades_path = os.path.join(args.out_dir, "trades.csv")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)

    m = equity_metrics(equity_df["equity"])
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2)

    # Plot equity curve
    plt.figure()
    equity_df["equity"].plot(title="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "equity_curve.png"))

    # Transparency reports
    if len(preselect_df) > 0:
        preselect_path = os.path.join(args.out_dir, "daily_candidates.csv")
        preselect_df.to_csv(preselect_path, index=False)

        # Full per-ticker rolling stats (for manual validation)
        import pandas as pd
        stats_concat = []
        for t, df in frames.items():
            st = rolling_window_stats(df, args.lookback, args.band_pct / 100.0, args.fluct_pct / 100.0)
            st = st.assign(ticker=t).reset_index().rename(columns={"index": "date"})
            stats_concat.append(st)
        stats_df = pd.concat(stats_concat, ignore_index=True)
        stats_path = os.path.join(args.out_dir, "screener_stats_all.csv")
        stats_df.to_csv(stats_path, index=False)

    print("Saved:")
    print(f"  {equity_path}")
    print(f"  {trades_path}")
    print(f"  {metrics_path}")
    print(f"  {os.path.join(args.out_dir, 'equity_curve.png')}")
    if len(preselect_df) > 0:
        print(f"  {preselect_path}")
        print(f"  {stats_path}")


if __name__ == "__main__":
    main()
