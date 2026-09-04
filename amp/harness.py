"""Backtest harness and scorecard.

One contract: a strategy turns a wide price frame into a frame of **target weights**.
The harness does everything else — execution lag, position sizing, the no-trade band,
order generation, and the scorecard. Strategies never call ``execute_order`` themselves,
which is what makes turnover controllable and every strategy comparable.

The harness keeps its own mark-to-market ledger alongside the engine. That is deliberate:
the docs never say whether ``engine.profit_loss`` is realised-only or marked to market,
and Sharpe and drawdown are meaningless on a realised-only curve. The independent ledger
is the primary equity curve; the engine's own numbers are reported beside it as a
cross-check, and ``reconciliation`` flags any divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- config


@dataclass
class BacktestConfig:
    """Everything about *how* we trade, independent of *what* we trade."""

    starting_balance: float = 1_000_000.0

    #: Proportional fee: shares * price * rate. This is the only fee the documented
    #: Quantify TradingEngine constructor accepts.
    commission_percentage: float = 0.001

    #: Flat fee per order, in currency units. NOT part of the documented Quantify API —
    #: present so the ledger can model a challenge that charges one. Verify with probe 7
    #: in notebooks/00_probe.py before setting it; if the real engine charges a flat fee
    #: we cannot configure, reconciliation() is what will expose the mismatch.
    commission_per_trade: float = 0.0

    #: Flat fee per share traded, in currency units. Same caveat as above.
    commission_per_share: float = 0.0

    #: Bars between the signal and the fill. 1 = decide on bar t, fill on bar t+1.
    #: 0 fills at the same close the signal was computed from — optimistic; use for
    #: an upper bound only.
    execution_lag: int = 1

    #: Only rebalance every N bars; on off-grid bars positions are left alone.
    #: The cheapest turnover control there is.
    rebalance_every: int = 1

    #: Fraction of equity deployed when weights are at full gross.
    gross_exposure: float = 1.0

    #: Hard cap on any single name's absolute weight, applied before normalisation.
    max_weight: float = 0.30

    #: Skip a name's trade when the order is smaller than this fraction of equity.
    #: The single most important knob for beating commission.
    no_trade_band: float = 0.0

    allow_short: bool = False
    lot_size: int = 1

    #: Fraction of equity deliberately left in cash. Commission is reserved for
    #: automatically on top of this, so a full-gross long book never overdrafts.
    cash_buffer: float = 0.0

    #: Close everything on the final bar so P&L is fully realised.
    close_at_end: bool = True


def _fee(cfg: BacktestConfig, volume: float, price: float) -> float:
    """Total cost of one order.

    The proportional term scales with notional, so it penalises *turnover*. The flat terms
    do not, so they penalise *order count* and *small orders* — which pulls strategy design
    in the opposite direction. Keep them separate rather than blending into one rate.
    """
    return (
        volume * price * cfg.commission_percentage
        + volume * cfg.commission_per_share
        + (cfg.commission_per_trade if volume > 0 else 0.0)
    )


# --------------------------------------------------------------------------- result


@dataclass
class BacktestResult:
    name: str
    config: BacktestConfig
    equity: pd.Series
    exposure: pd.DataFrame          # weights actually held, per bar
    positions: pd.DataFrame         # shares actually held, per bar
    orders: pd.DataFrame
    engine: Any
    prices: pd.DataFrame
    gross_equity: pd.Series         # equity curve with commission added back
    periods_per_year: float

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    def scorecard(self) -> Dict[str, float]:
        return scorecard(self)

    def summary(self) -> str:
        return format_scorecard(self.scorecard(), title=self.name)

    def per_ticker_pnl(self) -> pd.DataFrame:
        return per_ticker_pnl(self)

    def reconciliation(self) -> pd.Series:
        """Harness ledger vs. the engine's own accounting. Large gaps mean the engine
        defines P&L differently than assumed — investigate before trusting either."""
        eng = self.engine
        ledger_net = float(self.equity.iloc[-1] - self.config.starting_balance)
        ledger_comm = float(self.orders["commission"].sum()) if len(self.orders) else 0.0
        return pd.Series(
            {
                "ledger_net_pnl": ledger_net,
                "engine_net_pnl": _safe(eng, "profit_loss"),
                "ledger_gross_pnl": ledger_net + ledger_comm,
                "engine_gross_pnl": _safe(eng, "profit_loss_without_commission"),
                "ledger_commission": ledger_comm,
                "engine_commission": _safe(eng, "commission_costs"),
                "engine_trade_count": float(getattr(eng, "get_total_trade_count", lambda: np.nan)()),
                "harness_order_count": float(len(self.orders)),
            }
        )


def _safe(obj: Any, attr: str) -> float:
    try:
        return float(getattr(obj, attr))
    except Exception:
        return float("nan")


# -------------------------------------------------------------------------- runner


class Backtest:
    """Drive a strategy's target weights through a TradingEngine."""

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Any,
        config: Optional[BacktestConfig] = None,
        engine_factory: Optional[Callable[..., Any]] = None,
        tickers: Optional[Sequence[str]] = None,
    ) -> None:
        if tickers is not None:
            prices = prices[list(tickers)]
        self.prices = prices.sort_index()
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.engine_factory = engine_factory or _default_engine_factory

    # -- weight preparation -------------------------------------------------
    def _target_weights(self) -> "tuple[pd.DataFrame, pd.Series]":
        """Return (target weights, hold mask).

        A strategy row that is entirely NaN means **hold whatever we have** — leave the
        book untouched this bar. A row of zeros means **go flat**. The distinction is the
        difference between a buy-and-hold benchmark that trades twice and one that
        re-sizes on every bar as equity drifts, paying commission for nothing.
        Partial NaNs within a row are treated as zero.
        """
        cfg = self.config
        raw = self.strategy.weights(self.prices)
        raw = raw.reindex(index=self.prices.index, columns=self.prices.columns)

        hold = raw.isna().all(axis=1)
        w = raw.fillna(0.0)

        if not cfg.allow_short:
            w = w.clip(lower=0.0)

        w = w.clip(lower=-cfg.max_weight, upper=cfg.max_weight)

        # Normalise to unit gross so `gross_exposure` means what it says.
        gross = w.abs().sum(axis=1)
        w = w.div(gross.where(gross > 0, np.nan), axis=0).fillna(0.0)
        w *= cfg.gross_exposure

        # A weight computed from bar t can only be traded on bar t+lag.
        if cfg.execution_lag:
            w = w.shift(cfg.execution_lag).fillna(0.0)
            hold = hold.shift(cfg.execution_lag, fill_value=False)

        # Between rebalances, leave the book alone rather than chasing equity drift.
        if cfg.rebalance_every > 1:
            off_grid = np.ones(len(w), dtype=bool)
            off_grid[:: cfg.rebalance_every] = False
            hold = hold | pd.Series(off_grid, index=w.index)

        return w, hold

    # -- main loop ----------------------------------------------------------
    def run(self) -> BacktestResult:
        cfg = self.config
        prices = self.prices
        tickers = list(prices.columns)
        index = prices.index

        weights, hold_mask = self._target_weights()
        px = prices.to_numpy(dtype=float)
        wt = weights.to_numpy(dtype=float)
        hold = hold_mask.to_numpy(dtype=bool)

        engine = self.engine_factory(
            data=prices,
            starting_balance=cfg.starting_balance,
            commission_percentage=cfg.commission_percentage,
            commission_per_trade=cfg.commission_per_trade,
            commission_per_share=cfg.commission_per_share,
        )

        cash = float(cfg.starting_balance)
        shares = np.zeros(len(tickers), dtype=float)
        commission_paid = 0.0

        equity_curve = np.empty(len(index), dtype=float)
        pos_hist = np.zeros((len(index), len(tickers)), dtype=float)
        exp_hist = np.zeros((len(index), len(tickers)), dtype=float)
        orders: List[Dict[str, Any]] = []

        for i, timestamp in enumerate(index):
            row_px = px[i]
            valid = np.isfinite(row_px) & (row_px > 0)

            # Mark to market on the prices we can see; stale names keep their last mark.
            held_value = float(np.nansum(np.where(valid, shares * row_px, 0.0)))
            equity = cash + held_value

            target_w = wt[i]
            is_last = i == len(index) - 1
            forced_exit = is_last and cfg.close_at_end
            if forced_exit:
                target_w = np.zeros_like(target_w)

            if equity > 0 and (forced_exit or not hold[i]):
                # Reserve the commission so a fully invested book can still pay its fees.
                deployable = equity * (1.0 - cfg.cash_buffer) / (1.0 + cfg.commission_percentage)
                if cfg.commission_per_trade or cfg.commission_per_share:
                    deployable -= cfg.commission_per_trade * len(tickers)
                target_shares = np.zeros_like(shares)
                with np.errstate(invalid="ignore", divide="ignore"):
                    raw = np.where(valid, target_w * deployable / np.where(valid, row_px, 1.0), 0.0)
                if cfg.lot_size > 1:
                    target_shares = np.trunc(raw / cfg.lot_size) * cfg.lot_size
                else:
                    target_shares = np.trunc(raw)

                delta = target_shares - shares
                # No-trade band: ignore orders too small to be worth the commission.
                if cfg.no_trade_band > 0 and not (is_last and cfg.close_at_end):
                    notional = np.abs(delta) * np.where(valid, row_px, 0.0)
                    delta = np.where(notional < cfg.no_trade_band * equity, 0.0, delta)

                for j, qty in enumerate(delta):
                    qty = int(qty)
                    if qty == 0 or not valid[j]:
                        continue
                    price = float(row_px[j])
                    action = "BUY" if qty > 0 else "SELL"
                    fee = _fee(cfg, abs(qty), price)

                    engine.execute_order(tickers[j], abs(qty), action, timestamp)

                    cash -= qty * price
                    cash -= fee
                    commission_paid += fee
                    shares[j] += qty
                    orders.append(
                        {
                            "timestamp": timestamp,
                            "ticker": tickers[j],
                            "action": action,
                            "volume": abs(qty),
                            "price": price,
                            "notional": abs(qty) * price,
                            "commission": fee,
                            "position_after": shares[j],
                        }
                    )

                held_value = float(np.nansum(np.where(valid, shares * row_px, 0.0)))
                equity = cash + held_value

            equity_curve[i] = equity
            pos_hist[i] = shares
            exp_hist[i] = np.where(
                valid & (equity > 0), shares * np.where(valid, row_px, 0.0) / max(equity, 1e-12), 0.0
            )

        orders_df = pd.DataFrame(
            orders,
            columns=[
                "timestamp", "ticker", "action", "volume",
                "price", "notional", "commission", "position_after",
            ],
        )
        equity = pd.Series(equity_curve, index=index, name="equity")
        cum_comm = (
            orders_df.groupby("timestamp")["commission"].sum().reindex(index).fillna(0.0).cumsum()
            if len(orders_df)
            else pd.Series(0.0, index=index)
        )

        return BacktestResult(
            name=getattr(self.strategy, "name", type(self.strategy).__name__),
            config=cfg,
            equity=equity,
            exposure=pd.DataFrame(exp_hist, index=index, columns=tickers),
            positions=pd.DataFrame(pos_hist, index=index, columns=tickers),
            orders=orders_df,
            engine=engine,
            prices=prices,
            gross_equity=equity + cum_comm,
            periods_per_year=infer_periods_per_year(index),
        )


def _default_engine_factory(
    data: pd.DataFrame,
    starting_balance: float,
    commission_percentage: float,
    commission_per_trade: float = 0.0,
    commission_per_share: float = 0.0,
) -> Any:
    """Real ``quantify.TradingEngine`` in the notebook, local shim everywhere else.

    The real constructor takes no flat-fee arguments, so they are passed to the shim only.
    If the live engine turns out to charge one anyway, the harness ledger will disagree
    with it and ``BacktestResult.reconciliation()`` will show the gap.
    """
    try:
        from quantify.TradingEngine import TradingEngine  # type: ignore

        return TradingEngine(
            data=data,
            starting_balance=starting_balance,
            commission_percentage=commission_percentage,
        )
    except ImportError:
        from amp.engine_shim import TradingEngine  # type: ignore

        return TradingEngine(
            data=data,
            starting_balance=starting_balance,
            commission_percentage=commission_percentage,
            commission_per_trade=commission_per_trade,
            commission_per_share=commission_per_share,
        )


# ------------------------------------------------------------------------ metrics


def infer_periods_per_year(index: pd.Index) -> float:
    """Bars per year, so Sharpe annualises correctly on daily *or* intraday data."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252.0

    n_days = index.normalize().nunique()
    if n_days >= 3:
        bars_per_day = len(index) / n_days
        if bars_per_day > 1.5:
            return 252.0 * bars_per_day

    med = pd.Series(index).diff().median()
    if pd.isna(med) or med.total_seconds() <= 0:
        return 252.0
    if med >= pd.Timedelta(days=1):
        return 252.0 / (med / pd.Timedelta(days=1))
    return 252.0 * (6.5 * 3600.0 / med.total_seconds())


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def scorecard(result: BacktestResult) -> Dict[str, float]:
    cfg = result.config
    eq = result.equity
    rets = result.returns
    ppy = result.periods_per_year
    start = cfg.starting_balance

    commission = float(result.orders["commission"].sum()) if len(result.orders) else 0.0
    net_pnl = float(eq.iloc[-1] - start)
    gross_pnl = net_pnl + commission

    sd = float(rets.std())
    sharpe = float(rets.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")
    downside = rets[rets < 0]
    dsd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(rets.mean() / dsd * np.sqrt(ppy)) if dsd > 0 else float("nan")

    mdd = max_drawdown(eq)
    years = max(len(eq) / ppy, 1e-9)

    # Annualising a short sample produces absurd numbers - a 20x return over six weeks
    # "annualises" to billions of percent. Below a quarter of a year, refuse to quote it
    # and let total_return_pct carry the result instead.
    if years < 0.25 or eq.iloc[-1] <= 0:
        cagr = float("nan")
    else:
        cagr = float((eq.iloc[-1] / start) ** (1.0 / years) - 1.0)

    in_market = result.exposure.abs().sum(axis=1) > 1e-9
    live = rets[in_market.shift(1, fill_value=False)]
    hit_rate = float((live > 0).mean()) if len(live) else float("nan")

    traded = float(result.orders["notional"].sum()) if len(result.orders) else 0.0
    avg_equity = float(eq.mean())
    turnover = traded / avg_equity / years if avg_equity > 0 else float("nan")

    return {
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "commission": commission,
        "commission_drag_pct": float(commission / abs(gross_pnl) * 100.0) if gross_pnl else float("nan"),
        "total_return_pct": float(net_pnl / start * 100.0),
        "cagr_pct": cagr * 100.0 if np.isfinite(cagr) else float("nan"),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": mdd * 100.0,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        "hit_rate_pct": hit_rate * 100.0 if np.isfinite(hit_rate) else float("nan"),
        "annual_turnover_x": turnover,
        "n_orders": float(len(result.orders)),
        "avg_gross_exposure": float(result.exposure.abs().sum(axis=1).mean()),
        "bars": float(len(eq)),
        "sample_years": float(years),
    }


def per_ticker_pnl(result: BacktestResult) -> pd.DataFrame:
    """Attribute P&L to each name: position held * next-bar price change, minus fees."""
    px = result.prices
    pos = result.positions
    pnl = (pos * px.diff().shift(-1)).sum()
    fees = (
        result.orders.groupby("ticker")["commission"].sum()
        if len(result.orders)
        else pd.Series(0.0, index=px.columns)
    )
    fees = fees.reindex(px.columns).fillna(0.0)
    trades = (
        result.orders.groupby("ticker").size().reindex(px.columns).fillna(0).astype(int)
        if len(result.orders)
        else pd.Series(0, index=px.columns)
    )
    return pd.DataFrame(
        {"gross_pnl": pnl, "commission": fees, "net_pnl": pnl - fees, "orders": trades}
    ).sort_values("net_pnl", ascending=False)


_ORDER = [
    "net_pnl", "gross_pnl", "commission", "commission_drag_pct",
    "total_return_pct", "cagr_pct", "sharpe", "sortino",
    "max_drawdown_pct", "calmar", "hit_rate_pct",
    "annual_turnover_x", "n_orders", "avg_gross_exposure", "bars", "sample_years",
]


def format_scorecard(card: Dict[str, float], title: str = "backtest") -> str:
    lines = [title, "-" * max(len(title), 34)]
    for key in _ORDER:
        if key in card:
            lines.append("%-22s %14.2f" % (key, card[key]))
    return "\n".join(lines)


def compare(results: Sequence[BacktestResult], labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Scorecards for many runs side by side — the output of a parameter sweep."""
    labels = list(labels) if labels is not None else [r.name for r in results]
    return pd.DataFrame(
        [r.scorecard() for r in results], index=labels
    )[_ORDER].sort_values("sharpe", ascending=False)
