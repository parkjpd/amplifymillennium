"""Strategy contract and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:  # typing.Protocol is 3.8+, but keep the import defensive
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore


class Strategy(Protocol):
    """A strategy is a pure function from prices to target weights.

    ``weights(prices)`` returns a frame aligned to ``prices`` where row ``t`` is the
    portfolio we want to hold based on information available **at or before** ``t``.
    The harness applies the execution lag, so do not shift inside a strategy.

    Weights are relative: the harness normalises them to unit gross and scales by
    ``BacktestConfig.gross_exposure``. Positive is long, negative is short, zero is flat.
    """

    name: str

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        ...


@dataclass
class BuyAndHold:
    """Equal-weight the universe. The benchmark every strategy has to beat.

    ``rebalance=False`` (the default) buys once and then holds: subsequent rows are NaN,
    which the harness reads as "leave the book alone". That is what buy-and-hold actually
    means. ``rebalance=True`` re-equalises every bar instead, which is a different and far
    more expensive strategy — useful mainly for showing how much commission pure drift
    chasing costs.
    """

    warmup: int = 0
    rebalance: bool = False
    name: str = "buy_and_hold"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        w = w.where(prices.notna(), 0.0)
        if self.warmup:
            w.iloc[: self.warmup] = 0.0
        if not self.rebalance:
            entry = self.warmup
            w.iloc[entry + 1 :] = np.nan  # NaN row == hold
        return w


@dataclass
class SingleName:
    """Hold one ticker outright. Useful for sanity-checking the harness against
    a price series you can eyeball."""

    ticker: str
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = "hold_%s" % self.ticker

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w[self.ticker] = 1.0
        return w
