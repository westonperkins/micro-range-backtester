import pandas as pd
import numpy as np

def rolling_median_bands(close: pd.Series, lookback: int = 30, band_pct: float = 0.03):
    """
    Returns DataFrame with columns: median, lower, upper
    band_pct expressed as decimal (0.03 = 3%)
    """
    med = close.rolling(lookback, min_periods=lookback).median()
    upper = med * (1 + band_pct)
    lower = med * (1 - band_pct)
    return pd.DataFrame({"median": med, "lower": lower, "upper": upper})

def daily_entry_exit_signals(df: pd.DataFrame, lookback=30, band_pct=0.03):
    """
    For a single ticker DataFrame with columns open/high/low/close.
    - Entry signal when close <= lower band and price within universe constraints (handled upstream).
    - Exit target is upper band level on the day of entry; we check **next day's high** to see if TP hits.
    Returns df with 'median','lower','upper','entry_signal' (bool).
    """
    bands = rolling_median_bands(df["close"], lookback, band_pct)
    out = df.join(bands, how="left")
    out["entry_signal"] = (out["close"] <= out["lower"]).fillna(False)
    return out
