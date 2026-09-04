"""Shared feature library.

Every strategy imports from here so features are computed one way, once. All functions
take a wide price frame (index = timestamp, columns = tickers) and return something
aligned to it. Nothing here looks forward: row ``t`` uses only data at or before ``t``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- returns


def returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Simple return over ``periods`` bars."""
    return prices.pct_change(periods)


def log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return np.log(prices).diff(periods)


def forward_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Return over the NEXT ``periods`` bars. Targets only — never a feature."""
    return prices.pct_change(periods).shift(-periods)


# ------------------------------------------------------------------------ volatility


def realized_vol(prices: pd.DataFrame, window: int = 20, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Rolling standard deviation of 1-bar returns."""
    r = prices.pct_change()
    return r.rolling(window, min_periods=min_periods or window).std()


def atr_proxy(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Close-only stand-in for ATR: rolling mean absolute 1-bar move."""
    return prices.diff().abs().rolling(window, min_periods=window).mean()


# ----------------------------------------------------------------------------- trend


def moving_average(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.rolling(window, min_periods=window).mean()


def ma_ratio(prices: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """Short MA / long MA - 1. Positive = short-term strength."""
    return moving_average(prices, short) / moving_average(prices, long) - 1.0


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder-style RSI on close prices, in [0, 100]."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 means an unbroken rally -> RSI 100
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


# ------------------------------------------------------------------ normalisation


def rolling_zscore(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Time-series z-score: how unusual is this value for THIS ticker lately."""
    mu = frame.rolling(window, min_periods=window).mean()
    sd = frame.rolling(window, min_periods=window).std()
    return (frame - mu) / sd.replace(0.0, np.nan)


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row z-score across tickers: how unusual is this ticker vs. its peers now."""
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mu, axis=0).div(sd, axis=0)


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row rank across tickers, scaled to [0, 1]. Robust to outliers."""
    ranked = frame.rank(axis=1, method="average")
    n = frame.notna().sum(axis=1)
    return ranked.sub(0.5).div(n.replace(0, np.nan), axis=0)


def dispersion(frame: pd.DataFrame) -> pd.Series:
    """Cross-sectional spread per bar. Wide dispersion is what pays for commission."""
    return frame.std(axis=1)


# ------------------------------------------------------------------------- calendar


def time_of_day(index: pd.Index) -> pd.Series:
    """Minutes since midnight, for intraday seasonality. NaN if index isn't datetime."""
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Series(np.nan, index=index, name="time_of_day")
    return pd.Series(index.hour * 60 + index.minute, index=index, name="time_of_day")


def bars_since_open(index: pd.Index) -> pd.Series:
    """Bar count within each calendar day. NaN if index isn't datetime."""
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Series(np.nan, index=index, name="bars_since_open")
    day = pd.Series(index.normalize(), index=index)
    return day.groupby(day).cumcount().rename("bars_since_open")


# ----------------------------------------------------------------------- ML panel


DEFAULT_RETURN_LAGS = (1, 2, 3, 5, 10, 20)


def _stack(frame: pd.DataFrame) -> pd.Series:
    """Stack keeping NaNs, across pandas versions (``future_stack`` is 2.1+)."""
    try:
        return frame.stack(future_stack=True)
    except TypeError:  # pandas < 2.1
        return frame.stack(dropna=False)


def build_panel(
    prices: pd.DataFrame,
    return_lags: Iterable[int] = DEFAULT_RETURN_LAGS,
    vol_window: int = 20,
    target_horizon: int = 1,
    dropna: bool = True,
) -> pd.DataFrame:
    """Stack all tickers into one long frame for pooled model training.

    Returns a MultiIndex (timestamp, ticker) frame of features plus a ``fwd_return``
    and ``target`` column. Pooling is the point: 8 tickers give 8x the training rows,
    which is the difference between a usable model and an overfit one.
    """
    feats = {}
    for lag in return_lags:
        feats["ret_%d" % lag] = returns(prices, lag)
    feats["vol_20"] = realized_vol(prices, vol_window)
    feats["ma_ratio_5_20"] = ma_ratio(prices, 5, 20)
    feats["rsi_14"] = rsi(prices, 14)
    feats["xs_rank_ret_1"] = cross_sectional_rank(returns(prices, 1))
    feats["xs_z_ret_5"] = cross_sectional_zscore(returns(prices, 5))

    # Vol-normalised short-horizon return is the reversion signal itself.
    feats["ret_5_volnorm"] = returns(prices, 5) / realized_vol(prices, vol_window)

    panel = pd.concat(
        {name: _stack(frame) for name, frame in feats.items()}, axis=1
    )
    panel.index = panel.index.set_names(["timestamp", "ticker"])

    fwd = _stack(forward_returns(prices, target_horizon))
    fwd.index = fwd.index.set_names(["timestamp", "ticker"])
    panel["fwd_return"] = fwd
    panel["target"] = (panel["fwd_return"] > 0).astype("float")
    panel.loc[panel["fwd_return"].isna(), "target"] = np.nan

    return panel.dropna() if dropna else panel


# ------------------------------------------------------------------- diagnostics


def xs_autocorrelation(prices: pd.DataFrame, lookback: int = 5, horizon: int = 5) -> float:
    """Correlation between *relative* past return and *relative* forward return.

    Negative means the cross-section reverts (a laggard tends to catch up); positive means
    it trends. This is the single number that says whether a reversion strategy has any
    business existing on this data — check it before running a single backtest.
    """
    past = returns(prices, lookback)
    past = past.sub(past.mean(axis=1), axis=0)
    fwd = prices.pct_change(horizon).shift(-horizon)
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)
    joined = pd.concat(
        [_stack(past).rename("past"), _stack(fwd).rename("fwd")], axis=1
    ).dropna()
    if len(joined) < 30:
        return float("nan")
    return float(joined.corr().iloc[0, 1])


def reversion_map(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (1, 2, 3, 5, 10, 20),
    horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
) -> pd.DataFrame:
    """Grid of :func:`xs_autocorrelation` over signal lookback x holding horizon.

    Rows are lookbacks, columns are horizons. Read it before choosing parameters: the
    most negative cell is where reversion is strongest, and it tells you the holding
    period too. An all-positive grid means trade momentum instead, and a grid that hovers
    around zero means neither works and you should not be trading this signal at all.
    """
    return pd.DataFrame(
        {
            h: {lb: xs_autocorrelation(prices, lb, h) for lb in lookbacks}
            for h in horizons
        }
    ).rename_axis(index="lookback", columns="horizon")
