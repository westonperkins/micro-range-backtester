from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

from src.strategy import daily_entry_exit_signals
from src.screener import rolling_window_stats


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
    max_hold_days: int = 1  # if >0, cap holding period in days
    # NEW for screening:
    fluct_pct: float = 0.05       # require 30D window range <= 5% of median
    min_hit_rate: float = 0.15    # require >= 15% of last 30 days touch ±band


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    target_price: float
    qty: float
    days_held: int = 0


class PortfolioBacktester:
    def __init__(
        self,
        frames: Dict[str, pd.DataFrame],
        init_capital: float = 20000.0,
        costs: Costs = Costs(),
        params: Params = Params(),
    ):
        self.frames = frames
        self.init_capital = init_capital
        self.costs = costs
        self.params = params

        self.signal_frames: Dict[str, pd.DataFrame] = {}
        self.screen_frames: Dict[str, pd.DataFrame] = {}

        for t, df in frames.items():
            self.signal_frames[t] = daily_entry_exit_signals(
                df, params.lookback, params.band_pct
            )
            self.screen_frames[t] = rolling_window_stats(
                df, params.lookback, params.band_pct, params.fluct_pct
            )

    def _universe_mask(self, df: pd.DataFrame):
        return (df["close"] >= self.params.min_price) & (df["close"] <= self.params.max_price)

    def run(self):
        # unify calendar
        all_dates = sorted(set().union(*[df.index for df in self.frames.values()]))

        equity_rows = []
        cash = self.init_capital
        positions: List[Position] = []
        trades = []
        self._daily_preselect: List[dict] = []

        # leave one day at end for next-day fills
        for i in range(1, len(all_dates) - 1):
            d = all_dates[i]
            next_d = all_dates[i + 1]

            # --- 1) manage open positions (take profit on next day, or time exit) ---
            new_positions: List[Position] = []
            for pos in positions:
                df_next = self.frames[pos.ticker]
                if next_d not in df_next.index:
                    pos.days_held += 1
                    new_positions.append(pos)
                    continue

                hi = float(df_next.loc[next_d, "high"])
                cl = float(df_next.loc[next_d, "close"])

                exit_price = None
                reason = None

                # target fill if reachable
                if hi >= pos.target_price:
                    exit_price = pos.target_price
                    reason = "TP"
                elif self.params.max_hold_days > 0 and pos.days_held + 1 >= self.params.max_hold_days:
                    exit_price = cl
                    reason = "TIME"

                if exit_price is not None:
                    exit_price_net = exit_price * (1 - self.costs.commission_pct - self.costs.slippage_pct)
                    pnl = (exit_price_net - pos.entry_price) * pos.qty
                    cash += exit_price_net * pos.qty
                    trades.append(
                        {
                            "date": next_d,
                            "ticker": pos.ticker,
                            "side": "EXIT",
                            "price": exit_price_net,
                            "qty": pos.qty,
                            "reason": reason,
                            "pnl": pnl,
                        }
                    )
                else:
                    pos.days_held += 1
                    new_positions.append(pos)
            positions = new_positions

            # --- 2) build candidates for new entries to fill on next day's open ---
            candidates = []
            preselect_rows = []

            for t, sig_df in self.signal_frames.items():
                if d not in sig_df.index or d not in self.frames[t].index:
                    continue

                price_ok = bool(self._universe_mask(self.frames[t]).loc[d])

                scr = self.screen_frames.get(t)
                vol_ok = False
                hit_rate = None
                window_range_pct = None
                if scr is not None and d in scr.index:
                    vol_ok = bool(scr.loc[d, "vol_ok"])
                    hit_rate = float(scr.loc[d, "hit_rate"])
                    wrp = scr.loc[d, "window_range_pct"]
                    window_range_pct = float(wrp) if pd.notna(wrp) else None

                entry_signal = bool(sig_df.loc[d, "entry_signal"])

                # record preselection snapshot
                preselect_rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "price_ok": price_ok,
                        "vol_ok": vol_ok,
                        "hit_rate": hit_rate,
                        "window_range_pct": window_range_pct,
                        "entry_signal": entry_signal,
                    }
                )

                # apply gates to become a candidate
                if not price_ok:
                    continue
                if not vol_ok:
                    continue
                if hit_rate is None or hit_rate < self.params.min_hit_rate:
                    continue
                if not entry_signal:
                    continue

                target = sig_df.loc[d, "upper"]
                if pd.isna(target):
                    continue

                candidates.append((t, float(target)))

            # choose up to N names (simple: first N for now)
            candidates = candidates[: self.params.n_names]

            # mark which preselect rows were selected
            chosen = {t: True for t, _ in candidates}
            for r in preselect_rows:
                r["selected"] = bool(chosen.get(r["ticker"], False))
            self._daily_preselect.extend(preselect_rows)

            # --- 3) place entries for next day open (equal-weight) ---
            if len(candidates) > 0:
                alloc = cash / len(candidates)
                for t, target in candidates:
                    df_next = self.frames[t]
                    if next_d not in df_next.index:
                        continue
                    op = float(df_next.loc[next_d, "open"])
                    entry_price_net = op * (1 + self.costs.commission_pct + self.costs.slippage_pct)
                    if entry_price_net <= 0:
                        continue
                    qty = alloc // entry_price_net  # whole shares
                    if qty <= 0:
                        continue
                    cost = entry_price_net * qty
                    if cost > cash:
                        continue

                    cash -= cost
                    positions.append(
                        Position(
                            ticker=t,
                            entry_date=next_d,
                            entry_price=entry_price_net,
                            target_price=target,
                            qty=qty,
                        )
                    )
                    trades.append(
                        {
                            "date": next_d,
                            "ticker": t,
                            "side": "ENTRY",
                            "price": entry_price_net,
                            "qty": qty,
                            "reason": "SIGNAL",
                            "pnl": 0.0,
                        }
                    )

            # --- 4) mark-to-market equity at today's close ---
            pos_value = 0.0
            for pos in positions:
                df_today = self.frames[pos.ticker]
                if d in df_today.index:
                    cl = float(df_today.loc[d, "close"])
                    pos_value += cl * pos.qty

            total_equity = cash + pos_value
            equity_rows.append({"date": d, "equity": total_equity})

        equity_df = pd.DataFrame(equity_rows).set_index("date")
        trades_df = pd.DataFrame(trades)
        preselect_df = pd.DataFrame(self._daily_preselect)
        return equity_df, trades_df, preselect_df
