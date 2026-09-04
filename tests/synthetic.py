"""Synthetic price generators with known statistical properties.

Used to prove the harness and strategies behave correctly before they ever touch real
data: a reversion strategy must make money on a mean-reverting panel and lose it on a
trending one. If it makes money on both, the backtest is lying.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def _index(n: int, freq: str = "5min") -> pd.DatetimeIndex:
    """Weekday 09:30-16:00 session index, so bars-per-year inference sees real sessions.

    A naive ``date_range`` runs around the clock and makes intraday data look ~3x denser
    than it is, which silently inflates annualised Sharpe.
    """
    stamps: list = []
    day = pd.Timestamp("2024-01-02")
    while len(stamps) < n:
        if day.weekday() < 5:
            session = pd.date_range(
                day + pd.Timedelta(hours=9, minutes=30),
                day + pd.Timedelta(hours=16),
                freq=freq,
                inclusive="left",
            )
            stamps.extend(session[: n - len(stamps)])
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def mean_reverting_panel(
    n: int = 1500,
    tickers: Sequence[str] = TICKERS,
    phi: float = 0.85,
    idio_vol: float = 0.010,
    market_vol: float = 0.002,
    seed: int = 7,
    freq: str = "5min",
) -> pd.DataFrame:
    """Common random-walk market factor plus a stationary AR(1) idiosyncratic level.

    Because the idiosyncratic component mean-reverts, a name that has fallen *relative to
    its peers* tends to bounce — textbook cross-sectional reversion.
    """
    rng = np.random.default_rng(seed)
    k = len(tickers)
    market = np.cumsum(rng.normal(0.0, market_vol, n))

    idio = np.zeros((n, k))
    for t in range(1, n):
        idio[t] = phi * idio[t - 1] + rng.normal(0.0, idio_vol, k)

    levels = 100.0 * np.exp(market[:, None] + idio)
    return pd.DataFrame(levels, index=_index(n, freq), columns=list(tickers))


def trending_panel(
    n: int = 1500,
    tickers: Sequence[str] = TICKERS,
    drift_vol: float = 0.002,
    noise_vol: float = 0.004,
    seed: int = 11,
    freq: str = "5min",
) -> pd.DataFrame:
    """Each name carries a slow-moving persistent drift, so relative winners keep winning.

    This is the adversarial case for a reversion strategy — it should LOSE here.
    """
    rng = np.random.default_rng(seed)
    k = len(tickers)
    drift = np.cumsum(rng.normal(0.0, drift_vol, (n, k)), axis=0) * 0.02
    noise = rng.normal(0.0, noise_vol, (n, k))
    rets = drift + noise
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=_index(n, freq), columns=list(tickers)
    )


def deterministic_panel(path: Sequence[float], tickers: Sequence[str] = ("AAA",)) -> pd.DataFrame:
    """Hand-written price path for exact-arithmetic assertions."""
    data = {t: list(path) for t in tickers}
    return pd.DataFrame(data, index=_index(len(path)))


# Re-exported so tests assert on the same implementation the notebooks use.
from amp.features import xs_autocorrelation  # noqa: E402,F401
