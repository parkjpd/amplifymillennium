"""Panel with LONG-horizon momentum and SHORT-horizon reversal at the same time.

This is the structure the round-2 regime table actually shows: lb60 strongly positive,
lb5/10/20 negative. Neither of the earlier generators had it, so neither could test a
short-term reversal overlay honestly.
"""
import numpy as np
import pandas as pd

TICKERS = ["NEXGEN", "QUANTUM", "VELOCITY", "PINNACLE",
           "ATLAS", "HORIZON", "ZENITH", "MERIDIAN"]


def dual_panel(n=350, seed=3, drift_vol=0.006, rev_strength=0.020, rev_phi=0.55,
               betas=(0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7)):
    rng = np.random.default_rng(seed)
    k = len(TICKERS)
    mkt = rng.normal(0.0004, 0.010, n)
    b = np.array(betas)

    # persistent drift -> long-horizon momentum
    drift = np.cumsum(rng.normal(0.0, drift_vol, (n, k)), axis=0) * 0.02

    # fast mean-reverting LEVEL -> short-horizon reversal
    state = np.zeros(k)
    rev = np.zeros((n, k))
    for t in range(n):
        state = rev_phi * state + rng.normal(0.0, rev_strength, k)
        rev[t] = state

    log_px = np.cumsum(mkt[:, None] * b[None, :] + drift, axis=0) + rev
    return pd.DataFrame(100.0 * np.exp(log_px),
                        index=pd.bdate_range("2024-01-02", periods=n), columns=TICKERS)
