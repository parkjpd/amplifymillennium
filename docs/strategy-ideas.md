# Strategy ideas — 8-stock universe on Quantify

Three facts about this setup drive every idea below.

1. **8 names is a tiny universe.** Anything needing cross-sectional breadth (decile
   sorts, factor models) is dead. Anything relative-value is ideal — 8 names give
   **28 pairs**, and you can afford to model each name individually.
2. **Commission is per trade.** A round trip at `0.001` costs ~0.2% of notional. Turnover
   is the enemy. Every backtest reports gross *and* net (`profit_loss_without_commission`
   vs `profit_loss`) so we can see exactly what the trading is costing.
3. **Pooling multiplies data 8x.** One stock's history is probably a few thousand bars —
   too few for LightGBM. Stack all 8 into one long frame with a `ticker` column and you
   have a real training set.

Open question that gates half of this: **does a naked SELL open a short?** Ideas marked
`[needs short]` collapse to a long-only variant if not.

---

## Tier 0 — Infrastructure (build first)

**0.1 Backtest harness.** One function: strategy signal -> engine -> scorecard.
Reports net P&L, gross P&L, commission drag, Sharpe, max drawdown, hit rate, turnover,
trade count, and a per-ticker breakdown. Without this, comparing strategies is guesswork.

**0.2 Feature library.** One module every strategy imports: multi-horizon returns,
MA ratios, realized vol, z-scores (time-series and cross-sectional), RSI, range/ATR,
time-of-day dummies, sentiment. Compute once, reuse everywhere, no leakage.

**0.3 Data audit.** Run the TraderLink logic against the *real* 8-stock data before
trusting it: `event_gaps` to find halts and stale feeds, `investable_universe` filters to
check none of the 8 are too illiquid to trade at size. Gaps that aren't market closes are
usually where backtests silently lie.

---

## Tier 1 — Highest expected value

**1.1 Pairs / spread reversion** `[needs short]`
Screen all 28 pairs for cointegration or rolling correlation. Trade the spread z-score:
enter at |z| > 2, exit at |z| < 0.5, hard stop at |z| > 4 (the relationship broke).
Market-neutral, naturally low-turnover, and the small universe is an *advantage* here.
*Long-only variant:* hold whichever leg is cheap on the ratio, rotate on crossings.

**1.2 Cross-sectional mean reversion**
Each bar, rank the 8 by short-horizon return vs. the equal-weight basket. Buy the worst,
sell the best. Short-horizon equity reversal is one of the most robust effects that
exists. The whole question is whether it survives commission — test at several horizons
and only trade when dispersion is wide enough to pay for the round trip.

**1.3 Vol-scaled trend with a commission gate**
The docs' MA crossover, fixed: size positions ∝ 1/realized vol so each name contributes
equal risk, and require the signal to exceed a threshold tied to the commission before
acting. Naive crossovers churn; this is the version that doesn't.

---

## Tier 2 — Model-driven

**2.1 Pooled LightGBM**
Stack all 8 tickers into one frame, train on the shared feature set, target = next-bar
return sign (or top-2-of-8 cross-sectionally). Chronological split, never shuffled.
Booster returns probabilities — trade only high-confidence tails (p > 0.6 / p < 0.4) so
turnover stays low. Feature importance doubles as the analysis deliverable.

**2.2 ARMA residual signal**
`run_arma` per stock; trade the sign of the forecast only when |forecast| clears the
commission hurdle. Also useful as a *filter* on other strategies rather than standalone.

**2.3 Volatility regime switch**
Classify each bar into high/low realized-vol regime; run trend in high vol, mean-reversion
in low vol. Meta-layer over 1.2 and 1.3 rather than a strategy of its own. Often the
single biggest improvement to a mediocre signal.

---

## Tier 3 — Event and alt-data

**3.1 News sentiment: drift vs. fade**
Score headlines, snap timestamps onto the price index, then measure forward returns at
several horizons. The real question is empirical: does a positive headline *lead* return
(drift — trade with it) or is it already priced and reverting (fade — trade against it)?
Answer that before writing any trading logic. Output: an event-study chart.

**3.2 Sentiment as a feature, not a signal**
Cheaper and usually better — feed the score into the 2.1 feature set instead of trading
it directly.

**3.3 Weather** — only if any of the 8 are energy, utility, agri, or retail names.
Otherwise skip; it's a commodity tool.

---

## Tier 4 — Microstructure (a different game)

**4.1 Inventory-skewed market making.** Quote around fair value using `get_spread(vol, ref,
symbol)`; skew quotes against inventory so you get pulled back to flat. Widen when vol
spikes. Track `mm.current_positions` for inventory risk.

**4.2 Adverse-selection filter.** Don't quote into toxic flow — pull quotes after a vol
spike or fresh news. Measure post-fill mark-out to see whether you're being picked off.

**4.3 Build the order book.** `LOB.Order` is just a dataclass — no book, no matching. If a
challenge needs one: price-time priority, add/cancel/match, best bid/offer.

---

## TraderLink challenges (standalone deliverables)

| Challenge | Core logic | Where it bites |
|---|---|---|
| Ticks -> OHLCV | resample; first/max/min/last/sum | empty intervals, single-tick bars, interval-boundary convention |
| Multi-venue NBBO | max bid, min ask per timestamp | locked (bid==ask), crossed (bid>ask), stale venues |
| Event gaps | diff timestamps, threshold | separating real halts from scheduled market closes |
| Investable universe | filter by volume/price/mcap/sector | matching the expected output format exactly |

---

## Recommended sequence

Tier 0 first — the harness makes everything after it measurable. Then **1.2 cross-sectional
mean reversion** as the first real strategy: it's fast to write, it uses all 8 names, and
whether it survives commission tells us immediately how much turnover this simulation can
bear. That answer determines whether we head toward low-turnover relative value (1.1) or
model-driven selection (2.1).
