import pandas as pd
import numpy as np

def rolling_window_stats(
    df: pd.DataFrame,
    lookback: int = 30,
    band_pct: float = 0.03,
    fluct_pct: float = 0.05,
):
    """
    Rolling screening stats:
      - median, lower, upper (±band_pct)
      - window_range_pct = (30D rolling max - rolling min) / median
      - vol_ok: window_range_pct <= fluct_pct
      - hit_rate: share of last lookback days where |close - median| >= band_pct*median
      - exp_days_to_hit ~ 1 / hit_rate (inf when hit_rate=0)

    Returns DataFrame aligned to df.index with columns:
      median, lower, upper, window_range_pct, vol_ok, hit_rate, exp_days_to_hit
    """
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

    exp_days_to_hit = 1.0 / hit_rate.replace(0, np.nan)
    exp_days_to_hit = exp_days_to_hit.fillna(np.inf)

    vol_ok = (window_range_pct <= fluct_pct).fillna(False)

    out = pd.DataFrame(
        {
            "median": med,
            "lower": lower,
            "upper": upper,
            "window_range_pct": window_range_pct,
            "vol_ok": vol_ok,
            "hit_rate": hit_rate,
            "exp_days_to_hit": exp_days_to_hit,
        },
        index=df.index,
    )
    return out
