def _chart(prices, height=16, width=68):
    """ASCII chart of all 8 names rebased to 100. Matplotlib won't render in the
    terminal pane, so draw it with characters instead."""
    cols = list(prices.columns)
    rebased = prices / prices.iloc[0] * 100.0

    step = max(1, len(rebased) // width)
    sampled = rebased.iloc[::step]
    lo = float(sampled.min().min())
    hi = float(sampled.max().max())
    if hi <= lo:
        return

    grid = [[" "] * len(sampled) for _ in range(height)]
    for i, ticker in enumerate(cols):
        mark = ticker[0]
        series = sampled[ticker].to_numpy()
        for x, value in enumerate(series):
            y = int((value - lo) / (hi - lo) * (height - 1))
            grid[height - 1 - y][x] = mark

    print("\n  price paths, rebased to 100   (%.0f - %.0f)" % (lo, hi))
    for row_i, row in enumerate(grid):
        label = "%6.0f |" % (hi - (hi - lo) * row_i / (height - 1))
        print(label + "".join(row))
    print("       +" + "-" * len(sampled))
    print("        " + " ".join("%s=%s" % (t[0], t) for t in cols))


def _regime(prices, horizon=10):
    """Are relative winners continuing or snapping back? One number per lookback."""
    import numpy as np
    import pandas as pd

    print("\n  regime check (positive = momentum, negative = reversion)")
    for lb in (5, 10, 20, 60):
        rel = prices.pct_change(lb)
        rel = rel.sub(rel.mean(axis=1), axis=0)
        fwd = prices.pct_change(horizon).shift(-horizon)
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)
        joined = pd.concat([rel.stack(), fwd.stack()], axis=1).dropna()
        if len(joined) < 50:
            continue
        corr = float(joined.corr().iloc[0, 1])
        bar = "#" * int(abs(corr) * 40)
        print("    lookback %3d -> %+.3f  %s" % (lb, corr, bar))


def _leaders(prices):
    """Total and recent moves per name, so you can see who is actually driving P&L."""
    total = (prices.iloc[-1] / prices.iloc[0] - 1.0) * 100
    recent = (prices.iloc[-1] / prices.iloc[-min(60, len(prices)) ] - 1.0) * 100
    vol = prices.pct_change().std() * (252 ** 0.5) * 100
    print("\n  %-10s %8s %8s %8s" % ("ticker", "total%", "last60%", "vol%"))
    for t in total.sort_values(ascending=False).index:
        print("  %-10s %8.1f %8.1f %8.1f" % (t, total[t], recent[t], vol[t]))
