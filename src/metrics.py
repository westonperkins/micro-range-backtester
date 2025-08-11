import numpy as np
import pandas as pd

def equity_metrics(equity_curve: pd.Series, trading_days_per_year: int = 252):
    """
    equity_curve: pd.Series of equity values indexed by date
    Returns dict of metrics.
    """
    df = equity_curve.dropna()
    rets = df.pct_change().fillna(0.0)
    total_return = df.iloc[-1] / df.iloc[0] - 1.0 if len(df) > 1 else 0.0

    # CAGR
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 0
    cagr = (df.iloc[-1] / df.iloc[0])**(1/years) - 1 if years > 0 and df.iloc[0] > 0 else 0.0

    # Sharpe (daily)
    if rets.std() > 0:
        sharpe = np.sqrt(trading_days_per_year) * rets.mean() / rets.std()
    else:
        sharpe = 0.0

    # Max drawdown
    cum_max = df.cummax()
    dd = df / cum_max - 1.0
    max_dd = dd.min()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_daily": float(sharpe),
        "max_drawdown": float(max_dd),
    }
