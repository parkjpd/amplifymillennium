# %% [markdown]
# # 00 - Environment probe
#
# Answers the questions the Quantify docs leave open (see
# `docs/quantify-api-reference.md` section 12). Run this FIRST, in the AmplifyME
# notebook, and paste the output back — several design decisions depend on it,
# especially whether shorting works.
#
# Everything here operates on a local `TradingEngine` object built from a price frame.
# It places no orders on the exchange and touches no live counterparty.

# %%
import sys, traceback
sys.path.insert(0, "..")   # adjust so `import amp` finds the package

import numpy as np
import pandas as pd

from quantify.TradingEngine import TradingEngine, PositionDirection
from quantify.Data import get_price_series

DATASET = "your_dataset"          # <-- set this
COMMISSION = 0.001                # <-- set to the rate your challenge uses

results = {}


def probe(label):
    """Decorator that runs a probe and records a one-line verdict."""
    def wrap(fn):
        print("\n" + "=" * 72)
        print(label)
        print("=" * 72)
        try:
            verdict = fn()
        except Exception as exc:
            verdict = "RAISED %s: %s" % (type(exc).__name__, exc)
            traceback.print_exc(limit=2)
        results[label] = verdict
        print("-> %s" % verdict)
        return fn
    return wrap


def fresh(balance=1_000_000):
    return TradingEngine(data=prices, starting_balance=balance,
                         commission_percentage=COMMISSION)


# %% [markdown]
# ## The data itself

# %%
prices = get_price_series(DATASET, is_dataframe=True)

print("shape      :", prices.shape)
print("index type :", type(prices.index).__name__)
print("index head :", list(prices.index[:3]))
print("index tail :", list(prices.index[-3:]))
print("tickers    :", list(prices.columns))
print("dtypes     :", prices.dtypes.unique())
print("\nNaNs per ticker:\n", prices.isna().sum().to_string())
print("\nhead:\n", prices.head())
print("\ndescribe:\n", prices.describe().T[["count", "mean", "min", "max"]])

if isinstance(prices.index, pd.DatetimeIndex):
    gaps = pd.Series(prices.index).diff().value_counts().head(5)
    print("\nmost common bar spacing:\n", gaps.to_string())
    print("unique calendar days:", prices.index.normalize().nunique())
    print("bars per day        : %.1f" % (len(prices) / prices.index.normalize().nunique()))

# %% [markdown]
# ## Probe 1 — does a naked SELL open a short?
#
# The single most important unknown. Every market-neutral idea depends on it.

# %%
@probe("1. naked SELL opens a short?")
def _():
    engine = fresh()
    ticker = prices.columns[0]
    engine.execute_order(ticker, 10, "SELL", prices.index[1])
    if ticker not in engine:
        return "NO SHORT - engine reports no position after a naked SELL"
    pos = engine[ticker]
    print("   direction=%s volume=%s open_price=%s"
          % (pos.direction, pos.position_volume, pos.open_price))
    if pos.direction == PositionDirection.SHORT:
        return "SHORTING WORKS - direction is SHORT, volume %s" % pos.position_volume
    return "UNEXPECTED - direction is %s" % pos.direction


# %% [markdown]
# ## Probe 2 — what does execute_order return?

# %%
@probe("2. execute_order return value")
def _():
    engine = fresh()
    out = engine.execute_order(prices.columns[0], 10, "BUY", prices.index[1])
    print("   repr:", repr(out)[:400])
    if out is None:
        return "returns None"
    return "returns %s with attrs %s" % (
        type(out).__name__,
        [a for a in dir(out) if not a.startswith("_")][:15],
    )


# %% [markdown]
# ## Probe 3 — is `profit_loss` marked to market, or realised only?
#
# Decides whether the engine's own equity curve is usable for Sharpe and drawdown.
# The harness keeps an independent mark-to-market ledger either way, so this is a
# cross-check rather than a blocker.

# %%
@probe("3. profit_loss marked to market?")
def _():
    engine = fresh()
    ticker = prices.columns[0]
    t0, t1 = prices.index[1], prices.index[min(30, len(prices) - 1)]
    p0, p1 = float(prices.at[t0, ticker]), float(prices.at[t1, ticker])
    engine.execute_order(ticker, 100, "BUY", t0)

    pnl_at_entry = engine.profit_loss
    expected_mtm = (p1 - p0) * 100
    print("   entry %.4f -> later %.4f | expected unrealised %.2f" % (p0, p1, expected_mtm))
    print("   engine.profit_loss right after the buy: %.2f" % pnl_at_entry)

    # nudge the engine to re-mark by trading 1 share at the later timestamp
    engine.execute_order(ticker, 1, "BUY", t1)
    print("   engine.profit_loss after a trade at the later bar: %.2f" % engine.profit_loss)

    if abs(engine.profit_loss - expected_mtm) < abs(expected_mtm) * 0.2 + 1:
        return "MARKED TO MARKET - profit_loss tracks unrealised P&L"
    return "REALISED ONLY (or marks differently) - do not use it as an equity curve"


# %% [markdown]
# ## Probe 4 — a timestamp that isn't in the index

# %%
@probe("4. unknown timestamp")
def _():
    engine = fresh()
    engine.execute_order(prices.columns[0], 10, "BUY", "definitely-not-a-timestamp")
    return "SILENT - no exception on an unknown timestamp (dangerous)"


# %% [markdown]
# ## Probe 5 — ordering more than the balance allows

# %%
@probe("5. insufficient balance")
def _():
    engine = fresh(balance=1_000)
    ticker = prices.columns[0]
    price = float(prices.at[prices.index[1], ticker])
    engine.execute_order(ticker, int(1_000_000 / price), "BUY", prices.index[1])
    print("   balance after oversized order: %.2f" % engine.balance)
    if engine.balance < 0:
        return "ALLOWED - balance went negative, so size positions yourself"
    return "ALLOWED - balance %.2f, no rejection" % engine.balance


# %% [markdown]
# ## Probe 6 — an engine built without `data=`
#
# The docs' streaming example does this, then trades on it.

# %%
@probe("6. engine without data=")
def _():
    engine = TradingEngine(starting_balance=1_000_000, commission_percentage=COMMISSION)
    engine.execute_order(prices.columns[0], 10, "BUY", prices.index[1])
    return "WORKS - engine prices trades without a data= frame"


# %% [markdown]
# ## Probe 7 — what is the ACTUAL fee structure?
#
# The docs describe exactly one rule, `shares * price * commission_percentage`, with no
# flat-fee parameter. This reverse-engineers what the engine really charges by setting the
# percentage to zero and reading `commission_costs` back for two different order sizes:
#
# ```
# fee(1)   = per_share * 1   + per_trade
# fee(100) = per_share * 100 + per_trade
# ```
#
# Two equations, two unknowns. If both come back zero, the fee is purely proportional and
# the docs are complete.

# %%
@probe("7. actual fee structure")
def _():
    ticker, ts = prices.columns[0], prices.index[1]
    price = float(prices.at[ts, ticker])

    def fee_for(volume, pct):
        eng = TradingEngine(data=prices, starting_balance=10_000_000,
                            commission_percentage=pct)
        eng.execute_order(ticker, volume, "BUY", ts)
        return float(eng.commission_costs)

    fee_1 = fee_for(1, 0.0)
    fee_100 = fee_for(100, 0.0)
    per_share = (fee_100 - fee_1) / 99.0
    per_trade = fee_1 - per_share

    fee_pct = fee_for(100, 0.001)
    expected_pct = 100 * price * 0.001

    print("   price %.4f" % price)
    print("   pct=0.000, volume=1   -> commission_costs %.6f" % fee_1)
    print("   pct=0.000, volume=100 -> commission_costs %.6f" % fee_100)
    print("   implied per-share fee : %.6f" % per_share)
    print("   implied per-trade fee : %.6f" % per_trade)
    print("   pct=0.001, volume=100 -> commission_costs %.6f (docs predict %.6f)"
          % (fee_pct, expected_pct + fee_100))

    parts = []
    if abs(per_trade) > 1e-9:
        parts.append("FLAT %.4f per trade" % per_trade)
    if abs(per_share) > 1e-9:
        parts.append("%.4f per share" % per_share)
    if abs(fee_pct - (expected_pct + fee_100)) < max(expected_pct * 0.01, 1e-6):
        parts.append("proportional term matches the documented formula")
    else:
        parts.append("PROPORTIONAL TERM DIFFERS from shares*price*pct")

    if not parts[:-1]:
        return "PURELY PROPORTIONAL - no flat fee; " + parts[-1]
    return "; ".join(parts)


# %% [markdown]
# ## Summary — paste this back

# %%
print("\n" + "=" * 72)
print("PROBE SUMMARY")
print("=" * 72)
for k, v in results.items():
    print("%-38s %s" % (k, v))
print("\ndataset : %s" % DATASET)
print("shape   : %s" % (prices.shape,))
print("tickers : %s" % list(prices.columns))
