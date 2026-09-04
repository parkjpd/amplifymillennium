"""Replica of the AmplifyME runner, so the strategy can be exercised before submitting.

Uses `amp.engine_shim.TradingEngine` and synthetic daily panels built to trend or revert.
This proves the function runs clean and behaves sensibly in both regimes — it says nothing
about what it will score on the real data.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from amp.engine_shim import TradingEngine

TICKERS = ["NEXGEN", "QUANTUM", "VELOCITY", "PINNACLE",
           "ATLAS", "HORIZON", "ZENITH", "MERIDIAN"]
CAPITAL = 20_000_000.0
COMMISSION = 0.001


def daily_panel(n=400, kind="momentum", seed=3):
    """Daily bars with a deliberate cross-sectional signature.

    Noise must be injected into RETURNS, never into the log price level. White noise on
    the level shares a term between adjacent return windows and manufactures roughly -0.5
    cross-sectional autocorrelation out of nothing — which makes a trending panel measure
    as strongly mean-reverting. Verified by `verify_panels()` below.
    """
    rng = np.random.default_rng(seed)
    k = len(TICKERS)
    market_ret = rng.normal(0.0002, 0.009, n)

    if kind == "momentum":
        # Persistent per-name drift in return space, integrated into the price.
        drift = np.cumsum(rng.normal(0.0, 0.006, (n, k)), axis=0) * 0.02
        rets = market_ret[:, None] + drift + rng.normal(0.0, 0.008, (n, k))
        log_px = np.cumsum(rets, axis=0)
    else:
        # Stationary AR(1) idiosyncratic LEVEL: a name that falls behind catches up.
        idio = np.zeros((n, k))
        state = np.zeros(k)
        for t in range(n):
            state = 0.93 * state + rng.normal(0.0, 0.020, k)
            idio[t] = state
        log_px = np.cumsum(market_ret)[:, None] + idio

    index = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(100.0 * np.exp(log_px), index=index, columns=TICKERS)


def verify_panels():
    """Assert each panel actually has the signature it claims before trusting any test."""
    from amp.features import reversion_map

    mom = reversion_map(daily_panel(kind="momentum"),
                        lookbacks=(5, 20, 60), horizons=(5, 10, 20)).stack().mean()
    rev = reversion_map(daily_panel(kind="reverting"),
                        lookbacks=(5, 20, 60), horizons=(5, 10, 20)).stack().mean()
    assert mom > 0.05, "momentum panel is not trending: %+.3f" % mom
    assert rev < -0.05, "reverting panel is not reverting: %+.3f" % rev
    return mom, rev


def run(strategy, prices, label=""):
    """Mirror of the challenge runner, plus an equity curve for the score components."""
    engine = TradingEngine(data=prices, starting_balance=CAPITAL,
                           commission_percentage=COMMISSION)
    curve = []
    for i in range(len(prices)):
        ts = prices.index[i]
        historical = prices.iloc[: i + 1]
        strategy(engine, historical, ts)
        engine._mark_all(ts)
        curve.append(CAPITAL + engine.profit_loss)
    engine.close_all_positions(prices.index[-1])

    equity = pd.Series(curve, index=prices.index)
    rets = equity.pct_change().dropna()
    active = rets[rets != 0]

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
    mdd = float((equity / equity.cummax() - 1.0).min())
    win = float((active > 0).mean()) if len(active) else float("nan")

    return {
        "strategy": label,
        "pnl": float(engine.profit_loss),
        "return_pct": float(engine.profit_loss / CAPITAL * 100),
        "sharpe": sharpe,
        "max_dd_pct": mdd * 100,
        "win_rate_pct": win * 100,
        "commission": float(engine.commission_costs),
        "trades": engine.get_total_trade_count(),
    }


# ------------------------------------------------------------------ baselines


def starter(engine, prices, timestamp):
    """The provided starter: buy the alphabetically-first ticker every bar."""
    tickers = list(prices.columns)
    if len(prices) < 5:
        return
    engine.execute_order(tickers[0], 100, "BUY", timestamp)


def naive_momentum(engine, prices, timestamp):
    """Rank by raw 20-bar return, rebalance daily, no vol scaling or hysteresis.

    The obvious first answer, and the one most of the leaderboard will submit.
    """
    if len(prices) < 25:
        return
    ret = prices.pct_change(20).iloc[-1]
    order = list(ret.sort_values(ascending=False).index)
    longs, shorts = order[:2], order[-2:]
    last = prices.iloc[-1]
    per_name = CAPITAL * 0.8 / 4.0

    current = {}
    for t, p in engine.current_positions.items():
        v = getattr(p, "position_volume", 0) or 0
        d = getattr(getattr(p, "direction", None), "value", None)
        current[t] = v if d == "LONG" else (-v if d == "SHORT" else 0)

    targets = {t: 0 for t in prices.columns}
    for t in longs:
        targets[t] = int(per_name / float(last[t]))
    for t in shorts:
        targets[t] = -int(per_name / float(last[t]))

    for t in prices.columns:
        held, tgt = current.get(t, 0), targets[t]
        if held != 0 and tgt != 0 and (held > 0) != (tgt > 0):
            engine.execute_order(t, abs(held), "SELL" if held > 0 else "BUY", timestamp)
            held = 0
        d = tgt - held
        if d > 0:
            engine.execute_order(t, int(d), "BUY", timestamp)
        elif d < 0:
            engine.execute_order(t, int(-d), "SELL", timestamp)


if __name__ == "__main__":
    from momentum_strategy import momentum_strategy

    rows = []
    for kind in ("momentum", "reverting"):
        prices = daily_panel(kind=kind)
        for fn, label in ((starter, "starter (provided)"),
                          (naive_momentum, "naive momentum"),
                          (momentum_strategy, "ours")):
            r = run(fn, prices, label)
            r["market"] = kind
            rows.append(r)

    table = pd.DataFrame(rows)[
        ["market", "strategy", "pnl", "return_pct", "sharpe",
         "max_dd_pct", "win_rate_pct", "commission", "trades"]
    ]
    print(table.round(2).to_string(index=False))


class StrictEngine(TradingEngine):
    """Shim that behaves like the real engine's balance check.

    The live engine calls check_sufficient_balance(trade_value + commission) on every
    order and raises if it comes up short. Margin is modelled by direction of change,
    not by BUY/SELL: anything that grows a position (either way) ties up cash, anything
    that shrinks one gives it back. Charging every SELL as margin is wrong and starves
    the book, because closing a long is a SELL too.
    """

    def execute_order(self, ticker, volume, action, timestamp):
        volume = abs(int(volume))
        # The real engine rejects a second order for the same ticker on the same bar.
        seen = getattr(self, "_seen", {})
        if seen.get(ticker) == timestamp:
            raise RuntimeError(
                "You cannot record a trade on the same timestamp as the previous trade "
                "for %s. Reset the engine and try again." % ticker
            )
        price = self.price(ticker, timestamp)
        trade_value = volume * price
        commission = trade_value * self.commission_percentage

        pos = self._positions.get(ticker)
        before_qty = pos.signed_volume if pos else 0
        signed = volume if action.upper() == "BUY" else -volume
        after_qty = before_qty + signed
        growing = abs(after_qty) > abs(before_qty)

        if growing and trade_value + commission > self._cash + 1e-9:
            raise RuntimeError(
                "Insufficient balance to execute the trade. Required: $%.2f, Available: $%.2f"
                % (trade_value + commission, self._cash)
            )

        cash_before = self._cash
        record = super().execute_order(ticker, volume, action, timestamp)
        seen[ticker] = timestamp
        self._seen = seen

        # Recompute the cash move under a margin model: growing costs, shrinking refunds.
        opened = min(volume, abs(after_qty) - abs(before_qty)) if growing else 0
        closed = volume - opened
        self._cash = cash_before - opened * price + closed * price - commission
        return record
