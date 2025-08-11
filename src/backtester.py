from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.strategy import daily_entry_exit_signals

@dataclass
class Costs:
    commission_pct: float = 0.0002   # 0.02% per side
    slippage_pct: float = 0.0005     # 0.05% per side

@dataclass
class Params:
    lookback: int = 30
    band_pct: float = 0.03
    n_names: int = 10
    min_price: float = 1.0
    max_price: float = 3.0
    max_hold_days: int = 1  # if >0, cap holding period

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    target_price: float
    qty: float
    days_held: int = 0

class PortfolioBacktester:
    def __init__(self, frames: Dict[str, pd.DataFrame], init_capital: float = 20000.0, costs: Costs = Costs(), params: Params = Params()):
        self.frames = frames
        self.init_capital = init_capital
        self.costs = costs
        self.params = params
        # Precompute per-ticker signals
        self.signal_frames = {}
        for t, df in frames.items():
            self.signal_frames[t] = daily_entry_exit_signals(df, params.lookback, params.band_pct)

    def _universe_mask(self, df: pd.DataFrame):
        return (df["close"] >= self.params.min_price) & (df["close"] <= self.params.max_price)

    def run(self) -> pd.DataFrame:
        # Create a calendar of all dates present across tickers
        all_dates = sorted(set().union(*[df.index for df in self.frames.values()]))
        equity = []
        cash = self.init_capital
        positions: List[Position] = []
        trades = []  # records of entries/exits

        for i in range(1, len(all_dates)-1):  # leave one day for next-day fills
            d = all_dates[i]
            next_d = all_dates[i+1]

            # 1) Update existing positions: check TP on next day's high; or time-based exit
            new_positions = []
            for pos in positions:
                df_next = self.frames[pos.ticker]
                if next_d not in df_next.index:
                    # market closed for this ticker; carry position
                    pos.days_held += 1
                    new_positions.append(pos)
                    continue
                hi = float(df_next.loc[next_d, "high"])
                op = float(df_next.loc[next_d, "open"])
                cl = float(df_next.loc[next_d, "close"])
                exit_price = None
                reason = None

                # Take-profit at target if reachable
                if hi >= pos.target_price:
                    exit_price = pos.target_price
                    reason = "TP"
                # Else time-based exit
                elif self.params.max_hold_days > 0 and pos.days_held + 1 >= self.params.max_hold_days:
                    exit_price = cl
                    reason = "TIME"

                if exit_price is not None:
                    # apply costs on exit
                    exit_price_net = exit_price * (1 - self.costs.commission_pct - self.costs.slippage_pct)
                    pnl = (exit_price_net - pos.entry_price) * pos.qty
                    cash += exit_price_net * pos.qty
                    trades.append({"date": next_d, "ticker": pos.ticker, "side": "EXIT", "price": exit_price_net, "qty": pos.qty, "reason": reason, "pnl": pnl})
                else:
                    pos.days_held += 1
                    new_positions.append(pos)

            positions = new_positions

            # 2) New entries for today -> fill next day's open
            # Build candidate list from today's signals
            candidates = []
            for t, df in self.signal_frames.items():
                if d not in df.index or d not in self.frames[t].index:
                    continue
                if not self._universe_mask(self.frames[t]).loc[d]:
                    continue
                if not bool(df.loc[d, "entry_signal"]):
                    continue
                # Today's target (from today's upper band); we will try to fill tomorrow
                target = float(df.loc[d, "upper"]) if not np.isnan(df.loc[d, "upper"]) else None
                if target is None:
                    continue
                candidates.append((t, target))

            # Choose up to N names (simple: first N; you could rank later)
            candidates = candidates[: self.params.n_names]

            # Equal-weight capital allocation for new entries
            if len(candidates) > 0:
                alloc = cash / len(candidates)
                for t, target in candidates:
                    df_next = self.frames[t]
                    if next_d not in df_next.index:
                        continue  # can't fill
                    op = float(df_next.loc[next_d, "open"])
                    entry_price_gross = op
                    entry_price_net = entry_price_gross * (1 + self.costs.commission_pct + self.costs.slippage_pct)
                    if entry_price_net <= 0:
                        continue
                    qty = alloc // entry_price_net  # whole shares
                    if qty <= 0:
                        continue
                    cost = entry_price_net * qty
                    if cost > cash:
                        continue
                    cash -= cost
                    positions.append(Position(ticker=t, entry_date=next_d, entry_price=entry_price_net, target_price=target, qty=qty))
                    trades.append({"date": next_d, "ticker": t, "side": "ENTRY", "price": entry_price_net, "qty": qty, "reason": "SIGNAL", "pnl": 0.0})

            # 3) Mark-to-market equity at today's close (positions marked at close)
            pos_value = 0.0
            for pos in positions:
                df_today = self.frames[pos.ticker]
                if d in df_today.index:
                    cl = float(df_today.loc[d, "close"])
                    pos_value += cl * pos.qty
            total_equity = cash + pos_value
            equity.append({"date": d, "equity": total_equity})

        equity_df = pd.DataFrame(equity).set_index("date")
        trades_df = pd.DataFrame(trades)
        return equity_df, trades_df
