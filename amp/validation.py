"""Guards against fooling yourself with a parameter sweep.

Sweeping lookback x top_k x band over one dataset and reporting the best Sharpe is not a
result — it is the maximum of a noise distribution. Everything here exists to make the
selection step honest: choose parameters on one slice of time, report performance on a
slice the selection never saw.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from amp.harness import Backtest, BacktestConfig, BacktestResult, compare


def split(prices: pd.DataFrame, train_frac: float = 0.6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split. Never shuffle time series."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    cut = int(len(prices) * train_frac)
    return prices.iloc[:cut], prices.iloc[cut:]


def sweep(
    prices: pd.DataFrame,
    strategies: Sequence[Any],
    config: Optional[BacktestConfig] = None,
) -> Tuple[pd.DataFrame, List[BacktestResult]]:
    """Run every strategy on the same data and config; return the scorecard table."""
    cfg = config or BacktestConfig()
    results = [Backtest(prices, s, cfg).run() for s in strategies]
    return compare(results, [s.name for s in strategies]), results


def select_and_validate(
    prices: pd.DataFrame,
    strategies: Sequence[Any],
    config: Optional[BacktestConfig] = None,
    train_frac: float = 0.6,
    rank_by: str = "sharpe",
) -> Dict[str, Any]:
    """Pick the best strategy in-sample, then report it out-of-sample.

    The gap between ``train`` and ``test`` for the selected strategy is the number that
    matters. A big positive in-sample Sharpe collapsing to zero out-of-sample means the
    sweep found noise, and no amount of further tuning will fix that.

    ``decay`` is the fraction of the in-sample metric that survived. Below ~0.3, treat the
    strategy as unproven regardless of how good the training table looked.
    """
    cfg = config or BacktestConfig()
    train_px, test_px = split(prices, train_frac)

    train_table, _ = sweep(train_px, strategies, cfg)
    best_name = train_table[rank_by].idxmax()
    best = next(s for s in strategies if s.name == best_name)

    test_result = Backtest(test_px, best, cfg).run()
    full_result = Backtest(prices, best, cfg).run()

    train_metric = float(train_table.loc[best_name, rank_by])
    test_metric = float(test_result.scorecard()[rank_by])
    decay = test_metric / train_metric if train_metric not in (0.0, np.nan) else float("nan")

    return {
        "train_table": train_table,
        "best_name": best_name,
        "best_strategy": best,
        "train_metric": train_metric,
        "test_metric": test_metric,
        "decay": decay,
        "test_result": test_result,
        "full_result": full_result,
        "verdict": _verdict(train_metric, test_metric, decay),
    }


def _verdict(train_metric: float, test_metric: float, decay: float) -> str:
    if not np.isfinite(train_metric) or not np.isfinite(test_metric):
        return "INCONCLUSIVE - a metric was not finite"
    if test_metric <= 0:
        return "REJECT - the out-of-sample result is not positive; the sweep found noise"
    if decay < 0.3:
        return "WEAK - most of the in-sample edge did not survive; treat as unproven"
    if decay < 0.7:
        return "PLAUSIBLE - edge degraded but survived; size it conservatively"
    return "HOLDS - out-of-sample performance is close to in-sample"


_FEE_FIELDS = ("commission_percentage", "commission_per_trade", "commission_per_share")


def commission_sensitivity(
    prices: pd.DataFrame,
    strategy: Any,
    config: Optional[BacktestConfig] = None,
    rates: Sequence[float] = (0.0, 0.0005, 0.001, 0.002, 0.005),
    field: str = "commission_percentage",
) -> pd.DataFrame:
    """Where does this strategy stop making money?

    A strategy whose net P&L crosses zero just above the real commission rate is not an
    edge, it is a rounding error. This table tells you how much headroom you have.

    ``field`` selects which fee component to sweep. Sweeping ``commission_per_trade``
    answers a different question than ``commission_percentage``: a flat fee punishes small
    orders rather than large notional, so a strategy can be robust to one and fragile to
    the other.
    """
    if field not in _FEE_FIELDS:
        raise ValueError("field must be one of %s" % (_FEE_FIELDS,))
    base = config or BacktestConfig()
    rows = []
    for rate in rates:
        cfg = BacktestConfig(**{**base.__dict__, field: rate})
        card = Backtest(prices, strategy, cfg).run().scorecard()
        rows.append(
            {
                field: rate,
                "net_pnl": card["net_pnl"],
                "gross_pnl": card["gross_pnl"],
                "commission": card["commission"],
                "drag_pct": card["commission_drag_pct"],
                "sharpe": card["sharpe"],
                "n_orders": card["n_orders"],
            }
        )
    return pd.DataFrame(rows).set_index(field)


def breakeven_commission(
    prices: pd.DataFrame,
    strategy: Any,
    config: Optional[BacktestConfig] = None,
    hi: float = 0.02,
    tol: float = 1e-5,
    field: str = "commission_percentage",
) -> float:
    """Fee level at which net P&L hits zero. Higher is a more robust strategy.

    ``field`` selects the component; raise ``hi`` when sweeping a flat fee, since its
    units are currency per order rather than a rate.
    """
    if field not in _FEE_FIELDS:
        raise ValueError("field must be one of %s" % (_FEE_FIELDS,))
    base = config or BacktestConfig()

    def net(rate: float) -> float:
        cfg = BacktestConfig(**{**base.__dict__, field: rate})
        return Backtest(prices, strategy, cfg).run().scorecard()["net_pnl"]

    if net(0.0) <= 0:
        return 0.0
    lo = 0.0
    if net(hi) > 0:
        return hi  # profitable even at the ceiling
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if net(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
