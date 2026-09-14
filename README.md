# How we traded the 8 stock momentum comp

Every five days we scored all eight stocks and rebuilt the book. The score was one line
of math:

    score = R / sqrt(sigma)

R is the stock's total return since the first day of data, so a name that opened at 100
and now trades at 149 has R = 0.49. Sigma is the standard deviation of its daily returns
over the last 20 days, so a stock bouncing around 2% a day has sigma near 0.02.

The square root is the part worth explaining. Rank on R alone and you end up buying
whatever moved the most, and the thing that moves the most is usually just the jumpiest
stock rather than the one genuinely trending. Divide by sigma and you overcorrect the
other way, favouring anything that barely moves at all. The square root sits between the
two. Work it through:

    stock A: up 30%, sigma 0.025  ->  0.30 / 0.158 = 1.90
    stock B: up 35%, sigma 0.040  ->  0.35 / 0.200 = 1.75

Stock A wins even though it gained less, because it got there steadily. That is the whole
idea in one comparison.

Top two by score went long, bottom two went short.

For sizing we weighted inverse to volatility inside each side, so a name's weight was
proportional to 1/sigma and the weights summed to one. Between the two sides we weighted
by beta so the market exposure cancelled out:

    long share  = beta_short / (beta_long + beta_short)
    short share = beta_long  / (beta_long + beta_short)

If our longs averaged a beta of 1.4 and our shorts 0.8, the book put 36% into longs and
64% into shorts. Equal dollars on each side is not the same thing as equal market
exposure, and we never claimed to know where the market was going, only which of the
eight would beat the other seven. Shares were then target notional divided by price, sent
as one order per stock per bar, crossing straight through zero when a position flipped
from long to short.

## Round 1

Choppy. When we measured whether recent winners kept winning, the short horizons sat
around +0.05 and even the 60 day horizon only reached +0.14, which is close to nothing.
Long horizon momentum was the only thing with any edge in the data, so that is what we
ranked on. Finished at $1.52M with a 6.1% drawdown and a 72% win rate.

## Round 2

Trended hard, and the same code with no changes did the work. VELOCITY went 20 for 20 on
the long side, ZENITH 14 for 14 on the short side. $3.28M, a 4.2% drawdown, and an 83%
win rate, our best round on every measure.

## Round 3

Carried almost entirely by the short book. MERIDIAN and ATLAS made about $5M between them
while the longs handed some of it back. We made one change mid round: skip the first 90
days completely. With a full history lookback, the early rankings are built on a few weeks
of prices and are basically noise, and that was exactly where the drawdown was coming
from. Sitting out that stretch added $641k and pulled the drawdown from 8.1% down to 7.1%.
Finished at $3.13M.

## Why it held up

Commission was 0.1% per trade, so a round trip costs 0.2% of whatever you traded.
Rebalancing daily would have handed most of the profit straight back in fees. Five days
was often enough to stay on the right names without paying for the privilege.

No machine learning, no engineered features, nothing fitted to a particular round. A
return, a volatility, a ranking, and the discipline not to trade it too often. The same
parameters ran in all three rounds and the only thing we ever changed was when to start.

## What is in this repo

The strategy we actually submitted is `challenge/submit_v8.py`. Everything else exists
because `quantify` only runs inside the AmplifyME notebook, so we needed a way to test
changes without burning submissions.

```
challenge/
  submit_v8.py        the version we submitted
  submit_v*.py        every variant we tried, kept so the comparisons stay checkable
  run_local.py        a copy of the challenge runner, plus StrictEngine
  dual_panel.py       synthetic prices with long horizon trend and short horizon reversal
amp/
  harness.py          Backtest, BacktestConfig, scorecard
  features.py         shared signals and the regime diagnostic
  validation.py       out of sample selection, commission sensitivity
  engine_shim.py      offline stand in for quantify.TradingEngine
  strategies/         the strategy contract and benchmarks
tests/
  test_harness.py     26 offline tests, no pytest needed
docs/
  quantify-api-reference.md   condensed API plus the gaps the official docs leave open
notebooks/
  scratchpad.ipynb    local sandbox for running variants against synthetic panels
```

`StrictEngine` in `run_local.py` is the piece that made local testing worth anything. It
mirrors the real engine's two constraints: every order needs its full cost available in
the balance, and a ticker can only be traded once per timestamp. We found the second one
the hard way, after a version that looked fine locally kept closing positions and never
reopening them.

```bash
python3 tests/test_harness.py
```
