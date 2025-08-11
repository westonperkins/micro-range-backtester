# verify.py
import argparse
import os
import pandas as pd
import numpy as np


def recompute_stats(df, lookback, band_pct):
    close = df["close"]
    med = close.rolling(lookback, min_periods=lookback).median()
    upper = med * (1 + band_pct)
    lower = med * (1 - band_pct)
    roll_max = close.rolling(lookback, min_periods=lookback).max()
    roll_min = close.rolling(lookback, min_periods=lookback).min()
    window_range_pct = (roll_max - roll_min) / med

    dev = (close - med).abs()
    band_width = med * band_pct
    daily_hit = (dev >= band_width).astype(int)
    hit_rate = daily_hit.rolling(lookback, min_periods=lookback).mean().fillna(0.0)

    return pd.DataFrame(
        {
            "median": med,
            "upper": upper,
            "lower": lower,
            "window_range_pct": window_range_pct,
            "hit_rate": hit_rate,
        },
        index=df.index,
    )


def load_all_data(data_dir):
    frames = {}
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".csv"):
            p = os.path.join(data_dir, fn)
            df = pd.read_csv(p, parse_dates=["date"])
            df = df.sort_values("date").set_index("date")
            t = str(df["ticker"].iloc[0])
            frames[t] = df
    return frames


def pick_col(df, base):
    """Return df[base+'_rc'] if it exists, else df[base] if it exists, else a Series of NaN."""
    rc = f"{base}_rc"
    if rc in df.columns:
        return df[rc]
    if base in df.columns:
        return df[base]
    return pd.Series(index=df.index, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--band_pct", type=float, default=3.0)  # percent
    ap.add_argument("--fluct_pct", type=float, default=5.0)  # percent
    ap.add_argument("--min_hit_rate", type=float, default=0.15)
    ap.add_argument("--n_names", type=int, default=10)
    args = ap.parse_args()

    band = args.band_pct / 100.0
    fluct = args.fluct_pct / 100.0

    cand_path = os.path.join(args.out_dir, "daily_candidates.csv")
    stats_path = os.path.join(
        args.out_dir, "screener_stats_all.csv"
    )  # not strictly needed now
    trades_path = os.path.join(args.out_dir, "trades.csv")

    if not os.path.exists(cand_path):
        print("Run main.py first to produce daily_candidates.csv")
        return

    cand = pd.read_csv(cand_path, parse_dates=["date"])

    if os.path.exists(trades_path) and os.path.getsize(trades_path) > 0:
        trades = pd.read_csv(trades_path, parse_dates=["date"])
    else:
        trades = pd.DataFrame(
            columns=["date", "ticker", "side", "price", "qty", "reason", "pnl"]
        )

    frames = load_all_data(args.data_dir)

    checks = []
    for t, df in frames.items():
        # recompute stats from raw data
        rc = recompute_stats(df, args.lookback, band)
        rc = rc.assign(ticker=t).reset_index().rename(columns={"index": "date"})

        # merge candidates with recomputed stats (if columns overlap, rc gets *_rc)
        sub = cand[cand["ticker"] == t].merge(
            rc,
            on=["date", "ticker"],
            how="left",
            suffixes=("", "_rc"),
        )

        # add raw close for price check
        sub = sub.merge(
            df.reset_index()[["date", "close"]],
            on="date",
            how="left",
            suffixes=("", "_raw"),
        )

        # pull columns robustly
        wrp = pick_col(sub, "window_range_pct")
        lower = pick_col(sub, "lower")
        hit_rate = pick_col(sub, "hit_rate")

        # expected gates
        exp_price_ok = sub["close"].between(1.0, 3.0, inclusive="both")
        exp_vol_ok = (wrp <= fluct).fillna(False)
        exp_entry_signal = (sub["close"] <= lower).fillna(False)
        exp_hit_ok = (hit_rate >= args.min_hit_rate).fillna(False)

        # comparisons
        sub["price_ok_match"] = sub["price_ok"].astype(bool) == exp_price_ok
        sub["vol_ok_match"] = sub["vol_ok"].astype(bool) == exp_vol_ok
        sub["entry_signal_match"] = sub["entry_signal"].astype(bool) == exp_entry_signal

        sub["selected_gate_match"] = (~sub["selected"].astype(bool)) | (
            exp_price_ok & exp_vol_ok & exp_hit_ok & exp_entry_signal
        )

        checks.append(sub)

    if not checks:
        print("No tickers found in data/.")
        return

    chk = pd.concat(checks, ignore_index=True)

    # per-date selection sanity
    grp = chk.groupby("date")
    per_date = grp.apply(
        lambda g: pd.Series(
            {
                "selected_count": int(g["selected"].sum()),
                "eligible_count": int(
                    (
                        (g["price_ok"])
                        & (g["vol_ok"])
                        & (pick_col(g, "hit_rate") >= args.min_hit_rate)
                        & (g["entry_signal"])
                    ).sum()
                ),
            }
        ),
        include_groups=False,  # future-proof pandas behavior
    ).reset_index()

    problems = {
        "price_ok_mismatch": chk[~chk["price_ok_match"]],
        "vol_ok_mismatch": chk[~chk["vol_ok_match"]],
        "entry_signal_mismatch": chk[~chk["entry_signal_match"]],
        "selected_with_failed_gate": chk[~chk["selected_gate_match"]],
        "date_overflow_selected": per_date[per_date["selected_count"] > args.n_names],
    }

    print("\n=== Verification Summary ===")
    for name, dfp in problems.items():
        print(f"{name}: {len(dfp)} issues")

    out_issues = os.path.join(args.out_dir, "verification_issues")
    os.makedirs(out_issues, exist_ok=True)
    for name, dfp in problems.items():
        if len(dfp) > 0:
            dfp.to_csv(os.path.join(out_issues, f"{name}.csv"), index=False)

    print(f"\nDetails written (if any) to: {out_issues}")


if __name__ == "__main__":
    main()
