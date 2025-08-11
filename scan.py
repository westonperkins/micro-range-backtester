# scan.py
from __future__ import annotations

import io
import os
import sys
import time
import math
import random
import argparse
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# ------------------ CONFIG DEFAULTS ------------------
OUTPUT_DIR = "output"

# Data windows
LOOKBACK_DAYS_DEFAULT = 70        # enough to have >=30 trading days
WINDOW_DEFAULT = 30               # rolling window length (days)

# Universe sources (NASDAQ Trader)
NASDAQ_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Selection criteria (override via CLI as needed)
MIN_PRICE_DEFAULT = 1.0
MAX_PRICE_DEFAULT = 3.0
MAX_DEV_PCT_DEFAULT = 0.025       # <= 2.5% from 30d mean
WITHIN_FRAC_DEFAULT = 0.90        # >= 90% of last N closes within ±max_dev
IGNORE_OUTLIERS_DEFAULT = 0       # drop K worst days before scoring (0 = none)

# Batch + parallelism
BATCH_SIZE_DEFAULT = 150          # tickers per multi-ticker request
MAX_WORKERS_DEFAULT = 4           # concurrent batches (outer threads)

# Exclude non-common instruments by NAME and by SYMBOL
NAME_EXCLUDE_PATTERN = r"\b(?:ETF|Fund|Trust|Income|Preferred|Warrant|Right|Unit|Note|Bond|ETN|SPAC)\b"
SYMBOL_EXCLUDE_REGEX = r"(?:\$)|(?:\.(?:W|WS|U|R)(?:$|[\.]))"  # e.g., WRB$E, ABC.W/WS, ABC.U, ABC.R

# Optional: cache the latest downloaded NASDAQ files
CACHE_DIR = "data"
CACHE_NASDAQ = os.path.join(CACHE_DIR, "nasdaqlisted.txt")
CACHE_OTHER  = os.path.join(CACHE_DIR, "otherlisted.txt")
# -----------------------------------------------------


def _ensure_packages():
    try:
        import yfinance  # noqa: F401
    except ImportError:
        raise SystemExit("Install dependencies: pip install yfinance pandas numpy pyarrow requests")


def _download_text(url: str, cache_path: str) -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        txt = r.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(txt)
        return txt
    except Exception:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        raise


def _load_symbol_directory() -> pd.DataFrame:
    """Load and clean NASDAQ/NYSE/AMEX listings; drop ETFs, test issues, funds/warrants/units, weird symbols."""
    rd_kwargs = dict(sep="|", engine="python", dtype=str, na_filter=False)

    nasdaq_txt = _download_text(NASDAQ_LISTED_URL, CACHE_NASDAQ)
    other_txt  = _download_text(OTHER_LISTED_URL,  CACHE_OTHER)

    nasdaq = pd.read_csv(io.StringIO(nasdaq_txt), **rd_kwargs).rename(columns=str.strip)
    nasdaq = nasdaq[~nasdaq[nasdaq.columns[0]].str.startswith("File Creation Time", na=False)]
    nas_sym  = next((c for c in nasdaq.columns if c.strip().lower()=="symbol"), None)
    nas_name = next((c for c in nasdaq.columns if c.strip().lower().startswith("security name")), None)
    nas_etf  = next((c for c in nasdaq.columns if c.strip().lower()=="etf"), None)
    nas_test = next((c for c in nasdaq.columns if c.strip().lower().startswith("test issue")), None)
    if not all([nas_sym, nas_name, nas_etf, nas_test]):
        raise RuntimeError(f"Unexpected NASDAQ schema: {list(nasdaq.columns)}")
    nasdaq = nasdaq[[nas_sym, nas_name, nas_etf, nas_test]].copy()
    nasdaq.columns = ["symbol","name","etf","test_issue"]
    nasdaq["exchange"] = "NASDAQ"

    other = pd.read_csv(io.StringIO(other_txt), **rd_kwargs).rename(columns=str.strip)
    other = other[~other[other.columns[0]].str.startswith("File Creation Time", na=False)]
    oth_sym  = next((c for c in other.columns if c.strip().lower() in ("act symbol","symbol","cqs symbol")), None)
    oth_name = next((c for c in other.columns if c.strip().lower().startswith("security name")), None)
    oth_exch = next((c for c in other.columns if c.strip().lower()=="exchange"), None)
    oth_etf  = next((c for c in other.columns if c.strip().lower()=="etf"), None)
    oth_test = next((c for c in other.columns if c.strip().lower().startswith("test issue")), None)
    if not all([oth_sym, oth_name, oth_exch, oth_etf, oth_test]):
        raise RuntimeError(f"Unexpected OTHER schema: {list(other.columns)}")
    other = other[[oth_sym, oth_name, oth_exch, oth_etf, oth_test]].copy()
    other.columns = ["symbol","name","exchange","etf","test_issue"]

    df = pd.concat([nasdaq, other], ignore_index=True)
    df["etf"] = df["etf"].astype(str).str.upper()
    df["test_issue"] = df["test_issue"].astype(str).str.upper()

    # Base filters
    df = df[(df["etf"] != "Y") & (df["test_issue"] != "Y")]
    df = df[df["symbol"].astype(str).str.len() > 0]
    df = df[~df["symbol"].astype(str).str.startswith("$")]
    # Name & symbol pattern exclusions
    df = df[~df["name"].str.contains(NAME_EXCLUDE_PATTERN, case=False, na=False)]
    df = df[~df["symbol"].str.contains(SYMBOL_EXCLUDE_REGEX, regex=True, na=False)]

    df = df.drop_duplicates(subset=["symbol"])
    return df[["symbol","name","exchange"]].reset_index(drop=True)


def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _yf_download_batch(symbols: List[str], days: int, attempt: int = 1, max_attempts: int = 3) -> Dict[str, pd.DataFrame]:
    """Single multi-ticker request with retries. yfinance internal threads OFF; we control outer concurrency."""
    import yfinance as yf
    out: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return out
    try:
        df = yf.download(
            tickers=" ".join(symbols),
            period=f"{days}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,  # critical: avoid thread explosion
        )
    except Exception as e:
        if attempt < max_attempts:
            sleep_s = min(30, 2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
            return _yf_download_batch(symbols, days, attempt+1, max_attempts)
        else:
            print(f"[WARN] Batch request failed for {len(symbols)} tickers: {e}", file=sys.stderr)
            return out

    # Parse response
    if isinstance(df.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in df.columns.get_level_values(0):
                try:
                    sub = df[sym][["Open","High","Low","Close","Volume"]].dropna(how="all")
                    sub.index = pd.to_datetime(sub.index, utc=True)
                    if not sub.empty:
                        out[sym] = sub
                except Exception:
                    pass
    else:
        if not df.empty and len(symbols)==1:
            sub = df[["Open","High","Low","Close","Volume"]].dropna(how="all")
            sub.index = pd.to_datetime(sub.index, utc=True)
            out[symbols[0]] = sub

    time.sleep(0.15 + random.random()*0.2)  # small jitter to ease rate limits
    return out


def fetch_history_bulk(symbols: List[str], period_days: int, batch_size: int, max_workers: int) -> Dict[str, pd.DataFrame]:
    """Parallel, retrying downloader with split-batch fallback."""
    all_data: Dict[str, pd.DataFrame] = {}
    batches = list(_chunk(symbols, batch_size))
    random.shuffle(batches)

    def run_batches(batches_local: List[List[str]], attempt: int) -> Tuple[Dict[str, pd.DataFrame], List[List[str]]]:
        got: Dict[str, pd.DataFrame] = {}
        failed: List[List[str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_yf_download_batch, b, period_days): b for b in batches_local}
            total = len(futures); done = 0
            for fut in as_completed(futures):
                batch_syms = futures[fut]
                try:
                    res = fut.result()
                    if res:
                        got.update(res)
                    else:
                        failed.append(batch_syms)
                except Exception:
                    failed.append(batch_syms)
                done += 1
                if total >= 10 and done % max(1, total // 20) == 0:
                    print(f"  progress (attempt {attempt}): {done}/{total} batches")
        return got, failed

    # attempt 1
    got, failed = run_batches(batches, attempt=1)
    all_data.update(got)

    # attempt 2 (retry failed as-is)
    if failed:
        time.sleep(1.0)
        print(f"Retrying {len(failed)} failed batches...")
        got2, failed2 = run_batches(failed, attempt=2)
        all_data.update(got2)
        failed = failed2

    # attempt 3 (split failed)
    if failed:
        split = []
        for b in failed:
            mid = max(1, len(b)//2)
            split.extend([b[:mid], b[mid:]])
        time.sleep(1.5)
        print(f"Retrying {len(split)} split batches...")
        got3, failed3 = run_batches(split, attempt=3)
        all_data.update(got3)
        if failed3:
            print(f"[WARN] Still failed {sum(len(b) for b in failed3)} symbols after retries.", file=sys.stderr)

    return all_data


# ------------------ METRICS ------------------
def last_price(df: pd.DataFrame) -> float:
    if df.empty or "Close" not in df.columns:
        return np.nan
    return float(df["Close"].iloc[-1])


def mean_price(df: pd.DataFrame, n: int) -> float:
    close = df["Close"].dropna().tail(n)
    if len(close) < n:
        return np.nan
    return float(close.mean())


def frac_within_band(df: pd.DataFrame, n: int, max_dev: float, ignore_outliers: int = 0) -> float:
    """
    Fraction of last n closes whose absolute deviation from the mean is <= max_dev.
    Optionally drop the worst K days (largest deviations) before scoring.
    """
    close = df["Close"].dropna().tail(n)
    if len(close) < n:
        return np.nan
    m = float(close.mean())
    if not np.isfinite(m) or m <= 0:
        return np.nan
    dev = (close - m).abs() / m   # per-day deviation fraction
    if ignore_outliers > 0 and ignore_outliers < len(dev):
        dev = dev.nsmallest(len(dev) - ignore_outliers)  # drop K worst days
    return float((dev <= max_dev).mean())


# ------------------ SCAN ------------------
def scan_symbols(
    data: Dict[str, pd.DataFrame],
    price_min: float,
    price_max: float,
    max_dev_pct: float,
    window: int,
    within_frac: float,
    ignore_outliers: int,
) -> pd.DataFrame:
    rows = []
    for sym, df in data.items():
        if len(df.dropna(subset=["Close"])) < window:
            continue

        lp = last_price(df)
        m30 = mean_price(df, window)
        within = frac_within_band(df, window, max_dev_pct, ignore_outliers)

        passes_price = (price_min <= lp <= price_max) if not np.isnan(lp) else False
        passes_band  = (within >= within_frac) if np.isfinite(within) else False

        rows.append(
            {
                "symbol": sym,
                "last_price": lp,
                "mean_30d": m30,
                "within_frac_30d": within,
                "passes_price": passes_price,
                "passes_band": passes_band,
                "selected": passes_price and passes_band,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "symbol","last_price","mean_30d","within_frac_30d",
            "passes_price","passes_band","selected","timestamp_utc"
        ])

    df_out = pd.DataFrame(rows).sort_values(
        ["selected","passes_price","passes_band","symbol"],
        ascending=[False, False, False, True],
    )
    return df_out


# ------------------ IO ------------------
def write_snapshot(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    parquet_path = os.path.join(out_dir, f"candidates_{date_str}.parquet")
    csv_path     = os.path.join(out_dir, f"candidates_{date_str}.csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    # latest symlinks
    latest_csv = os.path.join(out_dir, "candidates_latest.csv")
    latest_parquet = os.path.join(out_dir, "candidates_latest.parquet")
    try:
        for link_path, target_path in [(latest_csv, csv_path), (latest_parquet, parquet_path)]:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(os.path.basename(target_path), link_path)
    except Exception:
        pass

    selected_count = int(df.get("selected", pd.Series(dtype=bool)).sum()) if not df.empty else 0
    print(f"Wrote {parquet_path} and {csv_path} ({selected_count} selected)")
    return parquet_path, csv_path


# ------------------ CLI / MAIN ------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan US market for $1–$3 names that stay mostly within ±X% of 30-day average.")
    ap.add_argument("--min_price", type=float, default=MIN_PRICE_DEFAULT, help=f"Minimum last price (default {MIN_PRICE_DEFAULT})")
    ap.add_argument("--max_price", type=float, default=MAX_PRICE_DEFAULT, help=f"Maximum last price (default {MAX_PRICE_DEFAULT})")
    ap.add_argument("--max_dev",   type=float, default=MAX_DEV_PCT_DEFAULT, help=f"Max deviation from 30d mean (default {MAX_DEV_PCT_DEFAULT} = 2.5%)")
    ap.add_argument("--within_frac", type=float, default=WITHIN_FRAC_DEFAULT, help=f"Required fraction within band (default {WITHIN_FRAC_DEFAULT} = 90%)")
    ap.add_argument("--ignore_outliers", type=int, default=IGNORE_OUTLIERS_DEFAULT, help=f"Ignore worst K days before scoring (default {IGNORE_OUTLIERS_DEFAULT})")
    ap.add_argument("--lookback",  type=int,   default=LOOKBACK_DAYS_DEFAULT, help=f"Download window in days (default {LOOKBACK_DAYS_DEFAULT})")
    ap.add_argument("--window",    type=int,   default=WINDOW_DEFAULT, help=f"Rolling window length (default {WINDOW_DEFAULT})")
    ap.add_argument("--universe_limit", type=int, default=0, help="Cap number of symbols for testing (0 = no cap)")
    ap.add_argument("--batch",     type=int,   default=BATCH_SIZE_DEFAULT, help=f"Tickers per batch (default {BATCH_SIZE_DEFAULT})")
    ap.add_argument("--workers",   type=int,   default=MAX_WORKERS_DEFAULT, help=f"Concurrent batches (default {MAX_WORKERS_DEFAULT})")
    return ap.parse_args()


def main():
    _ensure_packages()
    args = parse_args()

    print("Fetching symbol directory...")
    listings = _load_symbol_directory()
    if args.universe_limit and args.universe_limit > 0:
        listings = listings.head(args.universe_limit).copy()
    symbols = listings["symbol"].tolist()
    print(f"Symbols in universe after filters: {len(symbols)}")

    print("Downloading historical data in batches...")
    data = fetch_history_bulk(symbols, args.lookback, batch_size=args.batch, max_workers=args.workers)
    print(f"Downloaded histories: {len(data)}")

    # Prefilter by last price before detailed checks (use CLI thresholds!)
    prefiltered = {s: df for s, df in data.items() if args.min_price <= last_price(df) <= args.max_price}
    print(f"Prefiltered by last price ${args.min_price}-{args.max_price}: {len(prefiltered)}")

    results = scan_symbols(
        prefiltered,
        price_min=args.min_price,
        price_max=args.max_price,
        max_dev_pct=args.max_dev,
        window=args.window,
        within_frac=args.within_frac,
        ignore_outliers=args.ignore_outliers,
    )

    # Quick diagnostics
    if not results.empty:
        print(f"\nCounts — price:{int(results['passes_price'].sum())}  band:{int(results['passes_band'].sum())}  all:{int(results['selected'].sum())}")

    write_snapshot(results)

    if not results.empty:
        sel = results[results["selected"]]
        print("\n=== Selected (price & ±band around 30d mean) ===")
        cols = ["symbol","last_price","mean_30d","within_frac_30d"]
        print(sel[cols].head(100).to_string(index=False))
    else:
        print("No results.")

if __name__ == "__main__":
    main()
