"""Cross-sectional mean reversion.

Each bar, rank the universe by trailing return relative to the equal-weight basket, then
lean against it: buy the laggards, sell the leaders. Short-horizon reversal is one of the
most durable effects in equities, and unlike a decile sort it works on a small universe —
which is exactly what eight names is.

Three refinements matter more than the base signal:

* **Volatility scaling** — dividing trailing return by realised vol stops one volatile
  name from monopolising the book just because it moves more.
* **Dispersion gating** — the trade only pays when names have actually diverged. When
  everything moved together there is nothing to revert, and trading it just donates
  commission. ``dispersion_pct`` sits the strategy out on those bars.
* **top_k selection** — trading only the extremes cuts turnover hard, and the middle of
  the cross-section carries almost no signal anyway.

Set ``direction=+1`` to flip the whole thing into cross-sectional momentum. Running both
on the same data is the cheapest way to find out which regime the dataset is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from amp.features import realized_vol, returns


@dataclass
class CrossSectionalReversion:
    """Target weights from cross-sectional reversal.

    Args:
        lookback: bars of trailing return that form the signal.
        vol_window: window for the realised-vol scaler.
        vol_scale: divide the trailing return by realised vol before ranking.
        top_k: trade only the k strongest names per side. ``None`` = weight everything
            proportionally to signal strength.
        weighting: ``"equal"`` splits evenly across selected names, ``"signal"`` weights
            proportionally to the z-score. Equal is more robust; signal is more responsive.
        dispersion_pct: sit out unless cross-sectional dispersion is at least this
            percentile of its own trailing history (0 = always trade, 0.5 = top half).
        dispersion_window: lookback for that percentile.
        long_only: clip shorts, holding only the laggards.
        direction: -1 for reversion (default), +1 for cross-sectional momentum.
    """

    lookback: int = 5
    vol_window: int = 20
    vol_scale: bool = True
    top_k: Optional[int] = None
    weighting: str = "equal"
    dispersion_pct: float = 0.0
    dispersion_window: int = 100
    long_only: bool = False
    direction: int = -1
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.weighting not in ("equal", "signal"):
            raise ValueError("weighting must be 'equal' or 'signal'")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 (reversion) or +1 (momentum)")
        if self.name is None:
            kind = "xsec_rev" if self.direction == -1 else "xsec_mom"
            bits = ["%s_l%d" % (kind, self.lookback)]
            if self.top_k:
                bits.append("k%d" % self.top_k)
            if self.dispersion_pct:
                bits.append("d%.2f" % self.dispersion_pct)
            if self.long_only:
                bits.append("LO")
            self.name = "_".join(bits)

    @property
    def warmup(self) -> int:
        return max(self.lookback, self.vol_window if self.vol_scale else 0) + 1

    # -----------------------------------------------------------------------
    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        trailing = returns(prices, self.lookback)

        signal = trailing
        if self.vol_scale:
            vol = realized_vol(prices, self.vol_window)
            signal = trailing / vol.replace(0.0, np.nan)

        # Relative to the equal-weight basket, then standardised across names.
        centred = signal.sub(signal.mean(axis=1), axis=0)
        spread = centred.std(axis=1).replace(0.0, np.nan)
        z = centred.div(spread, axis=0)

        raw = self.direction * z

        if self.top_k:
            raw = self._select_extremes(raw, self.top_k, self.long_only)
        elif self.long_only:
            raw = raw.clip(lower=0.0)

        if self.weighting == "equal":
            raw = np.sign(raw) * (raw != 0)

        w = raw.fillna(0.0)

        gate = self._dispersion_gate(trailing)
        if gate is not None:
            w = w.where(gate, 0.0)

        w.iloc[: self.warmup] = 0.0
        return w

    # -----------------------------------------------------------------------
    @staticmethod
    def _select_extremes(raw: pd.DataFrame, k: int, long_only: bool) -> pd.DataFrame:
        """Keep the k highest (longs) and k lowest (shorts) per bar, zero the middle."""
        valid = raw.notna()
        high_rank = raw.rank(axis=1, ascending=False, method="first")
        low_rank = raw.rank(axis=1, ascending=True, method="first")

        keep_long = (high_rank <= k) & (raw > 0) & valid
        if long_only:
            return raw.where(keep_long, 0.0)
        keep_short = (low_rank <= k) & (raw < 0) & valid
        return raw.where(keep_long | keep_short, 0.0)

    def _dispersion_gate(self, trailing: pd.DataFrame) -> Optional[pd.Series]:
        """True on bars where the cross-section is dispersed enough to be worth trading."""
        if self.dispersion_pct <= 0:
            return None
        disp = trailing.std(axis=1)
        pct = disp.rolling(self.dispersion_window, min_periods=self.dispersion_window // 2).rank(pct=True)
        return pct >= self.dispersion_pct
