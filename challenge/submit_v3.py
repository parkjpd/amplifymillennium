def momentum_strategy(engine, prices, timestamp):
    """Rank the 8 names by risk-adjusted momentum, go long the best, short the worst."""
    import numpy as np

    lookback    = 900   # effectively all available history
    skip        = 0     # include the most recent bar
    k           = 2     # names per side
    buffer_     = 0     # respond to ranking changes immediately
    vol_window  = 20
    gross       = 1     # commit more capital
    rebal_every = 5     # rebalance faster
    min_resize  = 0.01  # allow smaller useful adjustments
    adaptive    = True   # no-op when the market trends, saves you when it reverts
    verbose     = True
    regime_every = 100  # print the regime table every N bars; 0 turns it off

    tickers = list(prices.columns)
    n = len(prices)

    if n < 30:
        return
    if n % rebal_every != 0:
        return

    lb = int(min(lookback, n - skip - 2))
    if lb < 10:
        return

    last = prices.iloc[-1]
    if last.isna().any() or (last <= 0).any():
        return

    # Measure momentum through the most recent available bar.
    mom = (prices.iloc[-1 - skip] / prices.iloc[-1 - skip - lb]) - 1.0

    # Recent volatility.
    vol = prices.pct_change().iloc[-int(min(vol_window, n - 1)):].std().replace(0.0, np.nan)
    if vol.isna().all():
        return
    vol = vol.fillna(vol.median())

    # Give momentum the majority of the weight while still penalising very volatile names.
    score = (mom / np.sqrt(vol)).dropna()
    if len(score) < 2 * k:
        return

    # Keep pure momentum rather than flipping into mean reversion.
    direction = 1
    if adaptive and n >= 60:
        direction = _direction(prices, window=int(min(90, max(30, n // 3))))
    score = score * direction

    # What we're holding now.
    current = {}
    for ticker, pos in engine.current_positions.items():
        held = getattr(pos, "position_volume", 0) or 0
        label = getattr(getattr(pos, "direction", None), "value", None)
        if label == "LONG":
            current[ticker] = int(held)
        elif label == "SHORT":
            current[ticker] = -int(held)

    # Rank stocks from strongest to weakest.
    order = list(score.sort_values(ascending=False).index)
    longs = _pick(order, set(t for t, v in current.items() if v > 0), k, buffer_)
    shorts = _pick(order[::-1], set(t for t, v in current.items() if v < 0), k, buffer_)
    shorts = [t for t in shorts if t not in longs]
    if len(longs) < k or len(shorts) < k:
        return

    # Size off what we actually have.
    held_value = sum(abs(v) * float(last[t]) for t, v in current.items())
    capital = float(engine.balance) + held_value
    if capital <= 0:
        return

    targets = dict((t, 0) for t in tickers)

    # Half the gross exposure long, half short, inverse volatility within each side.
    for names, sign in ((longs, 1), (shorts, -1)):
        inv = dict((t, 1.0 / float(vol[t])) for t in names if float(vol[t]) > 0)
        total = sum(inv.values())
        if total <= 0:
            continue
        side_budget = capital * gross / 2.0
        for t in names:
            targets[t] = sign * int(side_budget * inv[t] / total / float(last[t]))

    if verbose:
        print("bar %4d  %s  dir %+d  L=%s  S=%s  bal=%.0f"
              % (n, str(timestamp)[:10], direction,
                 ",".join(longs), ",".join(shorts), engine.balance))

    # The engine only accepts ONE order per ticker per timestamp.
    # A flip therefore crosses through zero in one order.
    orders = []
    for ticker in tickers:
        target = targets[ticker]
        held = current.get(ticker, 0)
        delta = target - held
        if delta == 0:
            continue

        # Ignore very small trims.
        same_side = held != 0 and target != 0 and (held > 0) == (target > 0)
        if same_side and abs(delta) * float(last[ticker]) < min_resize * capital:
            continue

        orders.append((ticker, delta, abs(target) - abs(held)))

    # Two passes. Shrinks first, because they release balance. Then grow what is left
    # over, scaled by a SINGLE common factor so every name is cut by the same
    # proportion. Filling greedily in ticker order lets the first name eat all the cash
    # and leaves the rest unfilled, which quietly breaks the long/short hedge.
    shrinks = [o for o in orders if o[2] <= 0]
    growths = [o for o in orders if o[2] > 0]

    for ticker, delta, _ in shrinks:
        _trade(engine, ticker, abs(delta), "BUY" if delta > 0 else "SELL", timestamp)

    if growths:
        need = sum(abs(d) * float(last[t]) * 1.002 for t, d, _ in growths)
        available = max(0.0, float(engine.balance) * 0.98)
        scale = min(1.0, available / need) if need > 0 else 0.0

        if verbose and scale < 0.99:
            print("  balance limited: filling %.0f%% of new exposure (need %.0f, have %.0f)"
                  % (scale * 100, need, available))

        for ticker, delta, _ in growths:
            volume = int(abs(delta) * scale)
            if volume > 0:
                _trade(engine, ticker, volume, "BUY" if delta > 0 else "SELL", timestamp)


def _pick(order, holding, k, buffer_):
    """Pick k names, but let something we already own keep its slot if it's still close."""
    picks = [t for t in order[:k + buffer_] if t in holding][:k]
    for t in order:
        if len(picks) >= k:
            break
        if t not in picks:
            picks.append(t)
    return picks[:k]


def _direction(prices, window, horizon=10, threshold=-0.10):
    """+1 if winners keep winning, -1 if they snap back.

    The threshold is deliberately asymmetric. Flipping when you should not is expensive
    in a trending market, so it stays on momentum unless the evidence is clearly negative.
    -0.10 tested better than -0.06 (fewer false flips) and than -0.15 (which was so strict
    it sat through a genuine reversion).
    """
    import numpy as np

    rel = prices.pct_change(horizon)
    rel = rel.sub(rel.mean(axis=1), axis=0)

    past = rel.iloc[-(window + horizon):-horizon]
    future = rel.shift(-horizon).iloc[-(window + horizon):-horizon]
    if len(past) < 20:
        return 1

    a = past.to_numpy(dtype=float).ravel()
    b = future.to_numpy(dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 60:
        return 1

    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return 1
    return -1 if float(np.corrcoef(a, b)[0, 1]) < threshold else 1


def _trade(engine, ticker, volume, action, timestamp):
    """Never let one rejected order kill the whole run."""
    if volume <= 0:
        return
    try:
        engine.execute_order(ticker, int(volume), action, timestamp)
    except Exception as exc:
        print("  skipped %s %s %s: %s" % (action, volume, ticker, exc))


def _regime(prices, horizon=10):
    """Is the cross-section trending or snapping back? Read this before touching anything.

    Positive = momentum, negative = reversion. If the longer lookbacks go clearly
    negative, the ranking needs to invert, which is exactly what `adaptive` does.
    """
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
        print("    lookback %3d -> %+.3f  %s" % (lb, corr, "#" * int(abs(corr) * 40)))
    print("")
