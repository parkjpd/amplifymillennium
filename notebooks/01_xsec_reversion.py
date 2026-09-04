# %% [markdown]
# # 01 - Cross-sectional mean reversion on the 8-stock universe
#
# Order of operations matters here. We establish that the effect exists *before*
# backtesting it, benchmark against strategies with no parameters, and only then sweep —
# selecting on one slice of time and reporting on a slice the selection never saw.
#
# Run `00_probe.py` first. If shorting turns out not to work, set `ALLOW_SHORT = False`
# below and everything falls back to long-only automatically.

# %%
import sys
sys.path.insert(0, "..")   # adjust so `import amp` finds the package

import numpy as np
import pandas as pd

from quantify.Data import get_price_series

from amp import (
    Backtest, BacktestConfig, BuyAndHold, CrossSectionalReversion,
    breakeven_commission, commission_sensitivity, compare,
    reversion_map, select_and_validate, sweep,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

DATASET = "your_dataset"      # <-- set this
COMMISSION = 0.001            # <-- set to your challenge's rate
STARTING_BALANCE = 1_000_000
ALLOW_SHORT = True            # <-- set from probe 1

prices = get_price_series(DATASET, is_dataframe=True)
print(prices.shape, list(prices.columns))
prices.head()

# %% [markdown]
# ## 1. Data audit
#
# Anything odd here invalidates every backtest downstream, so look before modelling.

# %%
print("NaNs per ticker:\n", prices.isna().sum().to_string())
print("\nnon-positive prices:", int((prices <= 0).sum().sum()))
print("duplicated timestamps:", int(prices.index.duplicated().sum()))
print("monotonic index:", prices.index.is_monotonic_increasing)

rets = prices.pct_change()
print("\n1-bar return stats:\n", rets.describe().T[["std", "min", "max"]].round(4))

# Outsized single-bar moves are usually splits, bad ticks, or halts - not alpha.
extreme = (rets.abs() > 0.10).sum()
print("\nbars with >10%% single-bar move:\n", extreme.to_string())

if isinstance(prices.index, pd.DatetimeIndex):
    spacing = pd.Series(prices.index).diff()
    typical = spacing.median()
    gaps = spacing[spacing > typical * 3]
    print("\ntypical bar spacing:", typical)
    print("gaps longer than 3x typical: %d" % len(gaps))
    if len(gaps):
        print(gaps.head(10).to_string())

# %% [markdown]
# ## 2. Does the effect even exist?
#
# `reversion_map` correlates relative past return against relative forward return across
# a grid of lookbacks and horizons.
#
# * **Negative cells → reversion.** Trade the laggards. Use the most negative cell's
#   lookback, and its horizon as the holding period.
# * **Positive cells → momentum.** Set `direction=+1` and trade the leaders instead.
# * **Everything near zero → neither.** Stop here. No parameter sweep will rescue a
#   signal that isn't in the data, and a sweep run on a null signal will still hand you a
#   winner made of noise.

# %%
rmap = reversion_map(prices)
print(rmap.round(3).to_string())

best_cell = rmap.stack().idxmin() if rmap.stack().min() < 0 else rmap.stack().idxmax()
print("\nstrongest cell: lookback=%s horizon=%s corr=%+.3f"
      % (best_cell[0], best_cell[1], rmap.loc[best_cell[0], best_cell[1]]))
DIRECTION = -1 if rmap.stack().mean() < 0 else +1
print("mean corr %+.3f -> trading direction %s"
      % (rmap.stack().mean(), "REVERSION (-1)" if DIRECTION == -1 else "MOMENTUM (+1)"))

# %% [markdown]
# ## 3. Benchmarks
#
# Two zero-parameter baselines. The second one matters more than it looks: continuously
# restoring equal weights mechanically sells winners and buys losers, so it *is* a
# reversion strategy. Any bespoke strategy has to beat it, not just buy-and-hold.

# %%
base_cfg = BacktestConfig(
    starting_balance=STARTING_BALANCE,
    commission_percentage=COMMISSION,
    allow_short=ALLOW_SHORT,
    execution_lag=1,
)

bench = [BuyAndHold(rebalance=False), BuyAndHold(rebalance=True)]
bench[1].name = "equal_weight_rebalanced"
bench_table, bench_results = sweep(prices, bench, base_cfg)
print(bench_table[["net_pnl", "gross_pnl", "commission", "sharpe",
                   "max_drawdown_pct", "n_orders"]].round(2).to_string())

# %% [markdown]
# ## 4. Parameter sweep, selected out-of-sample
#
# `select_and_validate` fits on the first 60% of bars, picks the best by Sharpe, and
# reports that one strategy on the remaining 40%. The `decay` figure — how much of the
# in-sample Sharpe survived — is the number worth trusting.

# %%
candidates = [
    CrossSectionalReversion(
        lookback=lb, top_k=k, direction=DIRECTION,
        long_only=not ALLOW_SHORT, dispersion_pct=d,
    )
    for lb in (2, 3, 5, 10, 20)
    for k in (1, 2, 3)
    for d in (0.0, 0.5)
]
print("%d candidate configurations" % len(candidates))

sweep_cfg = BacktestConfig(
    starting_balance=STARTING_BALANCE,
    commission_percentage=COMMISSION,
    allow_short=ALLOW_SHORT,
    execution_lag=1,
    no_trade_band=0.05,     # skip orders under 5% of equity
    rebalance_every=1,
)

selection = select_and_validate(prices, candidates, sweep_cfg, train_frac=0.6)

print("\ntop 10 in-sample:")
print(selection["train_table"].head(10)[
    ["net_pnl", "sharpe", "max_drawdown_pct", "commission_drag_pct", "n_orders"]
].round(2).to_string())

print("\nselected      : %s" % selection["best_name"])
print("train sharpe  : %.2f" % selection["train_metric"])
print("test sharpe   : %.2f" % selection["test_metric"])
print("decay         : %.2f" % selection["decay"])
print("VERDICT       : %s" % selection["verdict"])

# %% [markdown]
# ## 5. Turnover control
#
# Commission is the binding constraint. These two knobs are what decide whether a real
# signal survives contact with it.

# %%
best = selection["best_strategy"]
rows = []
for band in (0.0, 0.02, 0.05, 0.10, 0.20):
    for every in (1, 5, 20):
        cfg = BacktestConfig(**{**sweep_cfg.__dict__,
                                "no_trade_band": band, "rebalance_every": every})
        card = Backtest(prices, best, cfg).run().scorecard()
        rows.append({"band": band, "rebalance_every": every,
                     "net_pnl": card["net_pnl"], "gross_pnl": card["gross_pnl"],
                     "commission": card["commission"],
                     "drag_pct": card["commission_drag_pct"],
                     "sharpe": card["sharpe"], "n_orders": card["n_orders"]})
turnover_table = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
print(turnover_table.round(2).to_string(index=False))

# %% [markdown]
# ## 6. How much commission can it take?
#
# If breakeven sits just above the rate you actually pay, this is not an edge — it is a
# rounding error that happened to land the right way.

# %%
print("breakeven commission: %.4f%%  (you pay %.4f%%)"
      % (breakeven_commission(prices, best, sweep_cfg) * 100, COMMISSION * 100))
print()
print(commission_sensitivity(prices, best, sweep_cfg).round(2).to_string())

# %% [markdown]
# ## 7. Final run and attribution

# %%
winner = turnover_table.iloc[0]
final_cfg = BacktestConfig(**{
    **sweep_cfg.__dict__,
    "no_trade_band": float(winner["band"]),
    "rebalance_every": int(winner["rebalance_every"]),
})

final = Backtest(prices, best, final_cfg).run()
print(final.summary())
print("\nper-ticker:\n", final.per_ticker_pnl().round(2).to_string())
print("\nharness vs engine reconciliation:\n", final.reconciliation().round(2).to_string())

# %%
ax = final.equity.plot(figsize=(12, 5), label="net of commission")
final.gross_equity.plot(ax=ax, alpha=0.6, label="gross")
for r in bench_results:
    r.equity.plot(ax=ax, alpha=0.5, label=r.name)
ax.set_title("%s - equity" % final.name)
ax.legend()
ax.grid(alpha=0.3)

# %%
final.exposure.plot(figsize=(12, 4), title="portfolio weights", alpha=0.8).grid(alpha=0.3)
