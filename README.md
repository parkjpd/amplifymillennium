# amplifymillennium

Trading research toolkit for the AmplifyME **Quantify** simulation — an 8-stock universe,
a backtest harness that makes strategies comparable, and cross-sectional mean reversion as
the first strategy on top of it.

## Layout

```
amp/                        the library
  features.py               shared feature + diagnostic functions
  harness.py                Backtest, BacktestConfig, scorecard
  validation.py             out-of-sample selection, commission sensitivity
  engine_shim.py            offline stand-in for quantify.TradingEngine
  strategies/
    base.py                 Strategy contract, buy-and-hold benchmarks
    xsec_reversion.py       cross-sectional mean reversion / momentum
notebooks/
  00_probe.py               answers the questions the Quantify docs leave open
  01_xsec_reversion.py      data audit -> diagnostics -> benchmarks -> sweep -> validate
tests/
  synthetic.py              price panels with known statistical properties
  test_harness.py           21 offline tests, no pytest needed
docs/
  quantify-api-reference.md condensed Quantify API + the gaps in it
  strategy-ideas.md         the wider strategy backlog
```

## Running the tests

`quantify` only exists inside the AmplifyME Jupyter environment, so the harness falls back
to `amp.engine_shim` locally and the whole suite runs anywhere:

```bash
python3 tests/test_harness.py
```

## The design in one paragraph

A strategy is a pure function from prices to **target weights**; it never calls
`execute_order` itself. The harness owns execution lag, position sizing, the no-trade
band, order generation, and the scorecard — which is what makes turnover controllable and
every strategy comparable on the same axes. A weight row that is entirely `NaN` means
*hold the current book*; a row of zeros means *go flat*. The harness keeps its own
mark-to-market ledger alongside the engine, because the Quantify docs never say whether
`engine.profit_loss` is realised-only or marked to market, and Sharpe on a realised-only
curve is meaningless. `BacktestResult.reconciliation()` reports both side by side.

## Quick start

```python
from amp import Backtest, BacktestConfig, CrossSectionalReversion, reversion_map

# Does the effect exist before you backtest it?
print(reversion_map(prices).round(3))     # negative = reversion, positive = momentum

cfg = BacktestConfig(
    starting_balance=1_000_000,
    commission_percentage=0.001,
    execution_lag=1,        # decide on bar t, fill on bar t+1
    no_trade_band=0.05,     # skip orders under 5% of equity
    allow_short=True,
)
result = Backtest(prices, CrossSectionalReversion(lookback=5, top_k=2), cfg).run()
print(result.summary())
print(result.per_ticker_pnl())
```

## Two things worth knowing

**Commission is the binding constraint.** A round trip at `0.001` costs ~0.2% of notional.
Every scorecard reports gross and net P&L separately, and `breakeven_commission()` tells
you the rate at which a strategy stops making money. If breakeven sits just above the rate
you actually pay, it is not an edge.

**Equal-weight rebalancing is itself a reversion strategy.** Continuously restoring equal
weights mechanically sells winners and buys losers, so `BuyAndHold(rebalance=True)` earns
a real rebalancing premium on mean-reverting data. It is the zero-parameter benchmark a
bespoke strategy has to clear — beating plain buy-and-hold proves nothing.
