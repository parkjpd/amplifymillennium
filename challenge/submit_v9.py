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
    adaptive    = True   # no-op while the market trends; the guard for round 3
    demean      = False  # strip the market out of the SIGNAL. Tested worse on its own in a
                         # trending market - the market component carries real information
                         # there. Best trending Sharpe came from demean+beta_neutral together,
                         # but that gave back reverting performance. Off is the robust default.
    beta_neutral = True  # size the two sides so the book carries no net market exposure
    beta_window = 60
    quality     = True  # weight momentum by how steady the trend was
    quality_window = 120
    quality_power  = 1.0
    min_bars    = 90    # sit out until there is enough history for the ranking to mean
                        # something. With lookback=900 the early bars rank on almost no
                        # data, which is where round 3's drawdown came from.
    verbose     = True
    regime_every = 100  # regime table every N bars; 0 turns it off

    tickers = list(prices.columns)
    n = len(prices)

    if n < min_bars:
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

    # Strip the market out before vol-scaling. `mom` carries the common market move, and
    # dividing that shared component by each name's own vol quietly tilts the ranking
    # toward low-vol names whenever the market is up. Subtracting the cross-sectional mean
    # first leaves only the part of the move that is specific to the stock, which is the
    # only part this strategy claims to predict.
    if demean:
        mom = mom - mom.mean()

    # Give momentum the majority of the weight while still penalising very volatile names.
    score = (mom / np.sqrt(vol)).dropna()
    if len(score) < 2 * k:
        return


    # How much of the move was a steady trend versus one lucky jump? Fit a line to the log
    # price path and keep its R-squared. A name that climbed steadily scores near 1; one
    # that gapped once and went flat scores near 0 despite an identical total return.
    # This asks HOW the return was earned, not WHICH ticker earned it, so it generalises
    # across rounds in a way that excluding a name never could.
    if quality:
        qwin = int(min(lb, quality_window))
        logp = np.log(prices.iloc[-qwin:].to_numpy(dtype=float))
        x = np.arange(qwin, dtype=float)
        xc = x - x.mean()
        denom = float((xc ** 2).sum())
        yc = logp - logp.mean(axis=0)
        ss = (yc ** 2).sum(axis=0)
        slope = (xc[:, None] * yc).sum(axis=0) / denom if denom > 0 else np.zeros(yc.shape[1])
        resid = yc - slope[None, :] * xc[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(ss > 0, 1.0 - (resid ** 2).sum(axis=0) / ss, 0.0)
        r2 = np.clip(np.nan_to_num(r2), 0.0, 1.0)
        qmap = dict(zip(list(prices.columns), r2))
        score = score * np.array([qmap.get(t, 0.0) for t in score.index]) ** quality_power

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

    # Dollar-neutral is not market-neutral. If the longs carry more beta than the shorts,
    # the book ends up quietly long the market - an uncompensated bet for a strategy whose
    # only claimed edge is relative performance. Weighting the sides by the opposite side's
    # beta makes the two exposures cancel instead of merely matching in dollars.
    long_share, short_share = 0.5, 0.5
    if beta_neutral:
        beta = _betas(prices, beta_window)
        if beta:
            bl = float(np.mean([abs(beta.get(t, 1.0)) for t in longs]))
            bs = float(np.mean([abs(beta.get(t, 1.0)) for t in shorts]))
            if bl > 0 and bs > 0:
                long_share = bs / (bl + bs)
                short_share = bl / (bl + bs)

    # Inverse volatility within each side.
    for names, sign, share in ((longs, 1, long_share), (shorts, -1, short_share)):
        inv = dict((t, 1.0 / float(vol[t])) for t in names if float(vol[t]) > 0)
        total = sum(inv.values())
        if total <= 0:
            continue
        side_budget = capital * gross * share
        for t in names:
            targets[t] = sign * int(side_budget * inv[t] / total / float(last[t]))

    if regime_every and n % regime_every == 0:
        _regime(prices)

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

    # Reduce exposure first so balance is available for new positions.
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


def _direction(prices, window, horizon=10, threshold=-0.10):
    return _direction_stable(prices, threshold)


def _direction_stable(prices, threshold, horizon=10):
    """Flip only when the FULL history says the cross-section reverts.

    The old version judged on the last ~100 bars, which is few enough that noise alone
    could push it past the threshold and short the winners in a trending market. This uses
    every bar available - the same measurement the regime table prints - and additionally
    requires the 60-bar horizon to agree, so a single noisy short-window reading cannot
    flip the book on its own.
    """
    import pandas as pd

    def corr(lb):
        rel = prices.pct_change(lb)
        rel = rel.sub(rel.mean(axis=1), axis=0)
        fwd = prices.pct_change(horizon).shift(-horizon)
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)
        j = pd.concat([rel.stack(), fwd.stack()], axis=1).dropna()
        return float(j.corr().iloc[0, 1]) if len(j) >= 50 else 0.0

    short_c, long_c = corr(10), corr(60)
    return -1 if (short_c < threshold and long_c < 0) else 1


def _direction_old(prices, window, horizon=10, threshold=-0.10):
    """+1 if winners keep winning, -1 if they snap back."""
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


def _betas(prices, window):
    """Each name's beta to the equal-weight basket. 1.0 means it moves with the market."""
    rets = prices.pct_change().iloc[-window:]
    mkt = rets.mean(axis=1)
    var = float(mkt.var())
    if not var or not (var == var):
        return {}
    return dict((t, float(rets[t].cov(mkt) / var)) for t in prices.columns)


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
