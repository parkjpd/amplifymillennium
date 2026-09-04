def momentum_strategy(engine, prices, timestamp):
    """Rank the 8 names by risk-adjusted momentum, go long the best, short the worst."""
    import numpy as np

    lookback   = 90     # how far back we measure momentum
    skip       = 5      # ignore the last few bars, they tend to reverse
    k          = 2      # names per side
    buffer_    = 1      # rank slack before we drop something we already hold
    vol_window = 20
    gross      = 0.50   # share of capital we're willing to commit
    rebal_every = 10
    min_resize = 0.02   # don't bother trimming for less than this
    adaptive   = True
    verbose    = True
    chart_every = 100   # ASCII charts every N bars; 0 turns them off

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

    # Momentum, but measured up to `skip` bars ago rather than right up to today.
    # Very recent moves tend to snap back and they poison the ranking if you include them.
    mom = (prices.iloc[-1 - skip] / prices.iloc[-1 - skip - lb]) - 1.0

    vol = prices.pct_change().iloc[-int(min(vol_window, n - 1)):].std().replace(0.0, np.nan)
    if vol.isna().all():
        return
    vol = vol.fillna(vol.median())

    score = (mom / vol).dropna()
    if len(score) < 2 * k:
        return

    # Round 2 or 3 might not be a momentum market. Check whether recent winners actually
    # kept winning, and flip the ranking if they didn't.
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

    order = list(score.sort_values(ascending=False).index)
    longs = _pick(order, set(t for t, v in current.items() if v > 0), k, buffer_)
    shorts = _pick(order[::-1], set(t for t, v in current.items() if v < 0), k, buffer_)
    shorts = [t for t in shorts if t not in longs]
    if len(longs) < k or not shorts:
        return

    # Size off what we actually have. Shorts tie up balance the same as longs do, so
    # count both sides at absolute value.
    held_value = sum(abs(v) * float(last[t]) for t, v in current.items())
    capital = float(engine.balance) + held_value
    if capital <= 0:
        return

    targets = dict((t, 0) for t in tickers)
    for names, sign in ((longs, 1), (shorts, -1)):
        inv = dict((t, 1.0 / float(vol[t])) for t in names if float(vol[t]) > 0)
        total = sum(inv.values())
        if total <= 0:
            continue
        side_budget = capital * gross / 2.0
        for t in names:
            targets[t] = sign * int(side_budget * inv[t] / total / float(last[t]))

    if chart_every and n % chart_every == 0:
        print("\n===== bar %d  %s  =====" % (n, str(timestamp)[:10]))
        _chart(prices)
        _regime(prices)
        _leaders(prices)
        print("")

    if verbose:
        print("bar %4d  %s  dir %+d  L=%s  S=%s  bal=%.0f"
              % (n, str(timestamp)[:10], direction,
                 ",".join(longs), ",".join(shorts), engine.balance))

    # The engine only accepts ONE order per ticker per timestamp, so a flip has to be a
    # single order that crosses through zero (long 100 -> short 100 is SELL 200), not a
    # close followed by an open. The second order gets rejected and you end up flat.
    orders = []
    for ticker in tickers:
        target = targets[ticker]
        held = current.get(ticker, 0)
        delta = target - held
        if delta == 0:
            continue

        # Ignore small trims, they just pay commission.
        same_side = held != 0 and target != 0 and (held > 0) == (target > 0)
        if same_side and abs(delta) * float(last[ticker]) < min_resize * capital:
            continue

        orders.append((ticker, delta, abs(target) - abs(held)))

    # Anything that shrinks exposure frees up balance, so run those first.
    orders.sort(key=lambda o: o[2])

    for ticker, delta, growth in orders:
        price = float(last[ticker])
        volume = abs(delta)
        if growth > 0:
            affordable = int(max(0.0, float(engine.balance) * 0.95) / (price * 1.002))
            volume = min(volume, affordable)
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


def _direction(prices, window, horizon=10, threshold=-0.06):
    """+1 if winners keep winning, -1 if they snap back.

    Defaults to +1 unless the evidence is clearly negative, since the estimate is noisy
    and momentum is the premise we start from.
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
