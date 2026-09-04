"""Offline verification of the harness, the engine shim, and the reversion strategy.

Run: ``python3 tests/test_harness.py``  (no pytest required, works under pytest too)

The point of these is falsifiability. The reversion strategy must make money on a panel
built to mean-revert AND lose money on one built to trend. A strategy that wins on both
is a broken backtest, not an edge.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import synthetic as syn
from amp.harness import Backtest, BacktestConfig, compare, infer_periods_per_year
from amp.strategies.base import BuyAndHold, SingleName
from amp.strategies.xsec_reversion import CrossSectionalReversion


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# --------------------------------------------------------------- exact arithmetic


def test_buy_and_hold_exact_pnl():
    px = syn.deterministic_panel([100.0, 100.0, 110.0])
    cfg = BacktestConfig(starting_balance=1_000_000, commission_percentage=0.0)
    res = Backtest(px, BuyAndHold(), cfg).run()
    card = res.scorecard()
    # 1,000,000 / 100 = 10,000 shares, +10/share = +100,000
    assert approx(card["net_pnl"], 100_000.0), card["net_pnl"]
    assert approx(card["commission"], 0.0)
    assert res.positions.iloc[1]["AAA"] == 10_000


def test_commission_matches_documented_formula():
    px = syn.deterministic_panel([100.0, 100.0, 110.0])
    cfg = BacktestConfig(starting_balance=1_000_000, commission_percentage=0.001)
    res = Backtest(px, BuyAndHold(), cfg).run()
    shares = res.positions.iloc[1]["AAA"]
    expected_fee = shares * 100 * 0.001 + shares * 110 * 0.001
    card = res.scorecard()
    assert approx(card["commission"], expected_fee), (card["commission"], expected_fee)
    assert approx(card["gross_pnl"], shares * 10.0), card["gross_pnl"]
    assert approx(card["net_pnl"], card["gross_pnl"] - card["commission"])
    # commission must never push cash negative
    assert res.equity.min() > 0


def test_ledger_reconciles_with_engine():
    px = syn.mean_reverting_panel(n=400)
    cfg = BacktestConfig(commission_percentage=0.001, allow_short=True)
    res = Backtest(px, CrossSectionalReversion(top_k=2), cfg).run()
    rec = res.reconciliation()
    assert approx(rec["ledger_commission"], rec["engine_commission"], tol=1e-6), rec
    assert rec["harness_order_count"] == rec["engine_trade_count"], rec
    # everything closed at the end, so realised P&L must match the ledger
    assert approx(rec["ledger_net_pnl"], rec["engine_net_pnl"], tol=1e-4), rec


# ------------------------------------------------------------------- mechanics


def test_execution_lag_delays_the_fill():
    px = syn.deterministic_panel([100.0] * 6)
    lagged = Backtest(px, SingleName("AAA"), BacktestConfig(execution_lag=1)).run()
    immediate = Backtest(px, SingleName("AAA"), BacktestConfig(execution_lag=0)).run()
    assert lagged.positions.iloc[0]["AAA"] == 0
    assert lagged.positions.iloc[1]["AAA"] > 0
    assert immediate.positions.iloc[0]["AAA"] > 0


def test_long_only_never_shorts():
    px = syn.mean_reverting_panel(n=500)
    cfg = BacktestConfig(allow_short=False)
    res = Backtest(px, CrossSectionalReversion(top_k=3, long_only=True), cfg).run()
    assert (res.positions >= 0).all().all()
    assert res.positions.abs().sum().sum() > 0, "long-only strategy never traded"


def test_top_k_limits_names_held():
    px = syn.mean_reverting_panel(n=500)
    cfg = BacktestConfig(allow_short=True, no_trade_band=0.0)
    res = Backtest(px, CrossSectionalReversion(top_k=2), cfg).run()
    held = (res.positions.abs() > 0).sum(axis=1)
    assert held.max() <= 4, held.max()  # 2 long + 2 short
    assert held.max() >= 2


def test_gross_exposure_is_respected():
    px = syn.mean_reverting_panel(n=400)
    cfg = BacktestConfig(allow_short=True, gross_exposure=0.5)
    res = Backtest(px, CrossSectionalReversion(top_k=2), cfg).run()
    gross = res.exposure.abs().sum(axis=1)
    assert gross.max() <= 0.52, gross.max()


def test_no_trade_band_cuts_turnover():
    px = syn.mean_reverting_panel(n=800)
    strat = CrossSectionalReversion(top_k=2)
    tight = Backtest(px, strat, BacktestConfig(allow_short=True, no_trade_band=0.0)).run()
    banded = Backtest(px, strat, BacktestConfig(allow_short=True, no_trade_band=0.05)).run()
    assert banded.scorecard()["n_orders"] < tight.scorecard()["n_orders"]
    assert banded.scorecard()["commission"] < tight.scorecard()["commission"]


def test_rebalance_every_cuts_orders():
    px = syn.mean_reverting_panel(n=800)
    strat = CrossSectionalReversion(top_k=2)
    fast = Backtest(px, strat, BacktestConfig(allow_short=True)).run()
    slow = Backtest(px, strat, BacktestConfig(allow_short=True, rebalance_every=10)).run()
    assert slow.scorecard()["n_orders"] < fast.scorecard()["n_orders"]


def test_dispersion_gate_sits_out():
    px = syn.mean_reverting_panel(n=800)
    strat_open = CrossSectionalReversion(top_k=2, dispersion_pct=0.0)
    strat_gated = CrossSectionalReversion(top_k=2, dispersion_pct=0.7)
    cfg = BacktestConfig(allow_short=True)
    a = Backtest(px, strat_open, cfg).run()
    b = Backtest(px, strat_gated, cfg).run()
    flat_a = (a.exposure.abs().sum(axis=1) < 1e-9).mean()
    flat_b = (b.exposure.abs().sum(axis=1) < 1e-9).mean()
    assert flat_b > flat_a, (flat_a, flat_b)


# ------------------------------------------------------------ signal falsifiability


def test_generators_have_the_properties_they_claim():
    assert syn.xs_autocorrelation(syn.mean_reverting_panel()) < -0.15
    assert syn.xs_autocorrelation(syn.trending_panel()) > 0.15


def test_reversion_wins_on_reverting_panel_and_loses_on_trending():
    cfg = BacktestConfig(commission_percentage=0.0, allow_short=True)
    rev = CrossSectionalReversion(top_k=2, lookback=5)

    on_mr = Backtest(syn.mean_reverting_panel(), rev, cfg).run().scorecard()
    on_tr = Backtest(syn.trending_panel(), rev, cfg).run().scorecard()

    assert on_mr["sharpe"] > 0.5, on_mr["sharpe"]
    assert on_tr["sharpe"] < 0.0, on_tr["sharpe"]


def test_momentum_direction_flips_the_result():
    cfg = BacktestConfig(commission_percentage=0.0, allow_short=True)
    mom = CrossSectionalReversion(top_k=2, lookback=5, direction=+1)
    on_tr = Backtest(syn.trending_panel(), mom, cfg).run().scorecard()
    assert on_tr["sharpe"] > 0.0, on_tr["sharpe"]


def test_commission_is_a_real_hurdle():
    """The whole reason the harness reports gross and net separately."""
    px = syn.mean_reverting_panel()
    rev = CrossSectionalReversion(top_k=2)
    free = Backtest(px, rev, BacktestConfig(commission_percentage=0.0, allow_short=True)).run()
    paid = Backtest(px, rev, BacktestConfig(commission_percentage=0.001, allow_short=True)).run()
    assert paid.scorecard()["net_pnl"] < free.scorecard()["net_pnl"]
    assert paid.scorecard()["commission"] > 0


# ------------------------------------------------------------------- utilities


def test_infer_periods_per_year():
    daily = pd.date_range("2024-01-01", periods=300, freq="B")
    assert approx(infer_periods_per_year(daily), 252.0, tol=0.05)

    five_min = syn.mean_reverting_panel(n=2000).index
    ppy = infer_periods_per_year(five_min)
    assert 15_000 < ppy < 25_000, ppy


def test_compare_ranks_runs():
    px = syn.mean_reverting_panel(n=600)
    cfg = BacktestConfig(allow_short=True)
    runs = [
        Backtest(px, CrossSectionalReversion(top_k=k, lookback=5), cfg).run()
        for k in (1, 2, 3)
    ]
    table = compare(runs)
    assert len(table) == 3
    assert table["sharpe"].is_monotonic_decreasing


def test_per_ticker_pnl_covers_universe():
    px = syn.mean_reverting_panel(n=500)
    res = Backtest(px, CrossSectionalReversion(top_k=2), BacktestConfig(allow_short=True)).run()
    table = res.per_ticker_pnl()
    assert set(table.index) == set(px.columns)
    assert approx(float(table["commission"].sum()), res.scorecard()["commission"], tol=1e-6)


def test_buy_and_hold_holds():
    """A NaN weight row means hold. Buy-and-hold must trade twice per name, not daily."""
    px = syn.mean_reverting_panel(n=600)
    cfg = BacktestConfig(commission_percentage=0.001)
    held = Backtest(px, BuyAndHold(rebalance=False), cfg).run()
    churned = Backtest(px, BuyAndHold(rebalance=True), cfg).run()

    hc, cc = held.scorecard(), churned.scorecard()
    # one buy and one sell per name, nothing in between
    assert hc["n_orders"] == 2 * len(px.columns), hc["n_orders"]
    assert cc["n_orders"] > 10 * hc["n_orders"], cc["n_orders"]
    assert cc["commission"] > hc["commission"], (hc["commission"], cc["commission"])
    # NOTE: deliberately no assertion that churning earns less. Continuous equal-weight
    # rebalancing mechanically sells winners and buys losers, so it IS a mean-reversion
    # strategy and beats true buy-and-hold on a reverting panel despite the extra fees.
    # See test_rebalancing_is_itself_a_reversion_strategy.


def test_rebalancing_is_itself_a_reversion_strategy():
    """Equal-weight rebalancing sells winners and buys losers by construction.

    That makes it a genuine zero-parameter reversion benchmark: it should beat true
    buy-and-hold on a reverting panel and lose to it on a trending one. Any bespoke
    reversion strategy has to clear THIS bar, not just buy-and-hold.
    """
    cfg = BacktestConfig(commission_percentage=0.0)

    mr = syn.mean_reverting_panel(n=1500)
    held_mr = Backtest(mr, BuyAndHold(rebalance=False), cfg).run().scorecard()
    rebal_mr = Backtest(mr, BuyAndHold(rebalance=True), cfg).run().scorecard()
    assert rebal_mr["net_pnl"] > held_mr["net_pnl"], (held_mr["net_pnl"], rebal_mr["net_pnl"])

    tr = syn.trending_panel(n=1500)
    held_tr = Backtest(tr, BuyAndHold(rebalance=False), cfg).run().scorecard()
    rebal_tr = Backtest(tr, BuyAndHold(rebalance=True), cfg).run().scorecard()
    assert rebal_tr["net_pnl"] < held_tr["net_pnl"], (held_tr["net_pnl"], rebal_tr["net_pnl"])


def test_hold_rows_leave_positions_untouched():
    px = syn.mean_reverting_panel(n=300)
    res = Backtest(px, BuyAndHold(rebalance=False), BacktestConfig()).run()
    interior = res.positions.iloc[2:-1]
    assert (interior.nunique() == 1).all(), "positions drifted during a hold"


def test_cagr_suppressed_on_short_samples():
    """Annualising six weeks of intraday data must not report billions of percent."""
    short = syn.mean_reverting_panel(n=600)          # well under a quarter of a year
    res = Backtest(short, CrossSectionalReversion(top_k=2),
                   BacktestConfig(allow_short=True)).run()
    card = res.scorecard()
    assert card["sample_years"] < 0.25, card["sample_years"]
    assert np.isnan(card["cagr_pct"]), card["cagr_pct"]
    assert np.isnan(card["calmar"]), card["calmar"]
    assert np.isfinite(card["total_return_pct"])     # this one still means something


# ------------------------------------------------------------------- flat fees


def test_flat_per_trade_fee_is_charged_once_per_order():
    px = syn.deterministic_panel([100.0, 100.0, 110.0])
    cfg = BacktestConfig(commission_percentage=0.0, commission_per_trade=0.10)
    res = Backtest(px, BuyAndHold(rebalance=False), cfg).run()
    card = res.scorecard()
    assert card["n_orders"] == 2, card["n_orders"]           # one buy, one sell
    assert approx(card["commission"], 0.20), card["commission"]


def test_per_share_fee_scales_with_volume():
    px = syn.deterministic_panel([100.0, 100.0, 110.0])
    cfg = BacktestConfig(commission_percentage=0.0, commission_per_share=0.01)
    res = Backtest(px, BuyAndHold(rebalance=False), cfg).run()
    shares = res.positions.iloc[1]["AAA"]
    assert approx(res.scorecard()["commission"], shares * 0.01 * 2), res.scorecard()["commission"]


def test_flat_fees_reconcile_with_the_shim():
    """Ledger and shim must agree on flat fees too, or the scorecard is fiction."""
    px = syn.mean_reverting_panel(n=400)
    cfg = BacktestConfig(
        commission_percentage=0.001, commission_per_trade=0.10,
        commission_per_share=0.005, allow_short=True,
    )
    res = Backtest(px, CrossSectionalReversion(top_k=2), cfg).run()
    rec = res.reconciliation()
    assert approx(rec["ledger_commission"], rec["engine_commission"], tol=1e-6), rec
    assert approx(rec["ledger_net_pnl"], rec["engine_net_pnl"], tol=1e-4), rec


def test_flat_fee_punishes_small_orders_not_turnover():
    """The strategic inversion: a flat fee makes ORDER COUNT the cost driver.

    Under a proportional fee, halving order size halves the cost. Under a flat per-trade
    fee it does not, so the same notional split across more orders costs strictly more.
    That is why the no-trade band matters even more when a flat fee is present.
    """
    px = syn.mean_reverting_panel(n=800)
    strat = CrossSectionalReversion(top_k=2)

    flat = BacktestConfig(commission_percentage=0.0, commission_per_trade=1.0, allow_short=True)
    many = Backtest(px, strat, BacktestConfig(**{**flat.__dict__, "no_trade_band": 0.0})).run()
    few = Backtest(px, strat, BacktestConfig(**{**flat.__dict__, "no_trade_band": 0.10})).run()

    # cost tracks order count exactly, not notional
    assert approx(many.scorecard()["commission"], many.scorecard()["n_orders"])
    assert approx(few.scorecard()["commission"], few.scorecard()["n_orders"])
    assert few.scorecard()["commission"] < many.scorecard()["commission"]


def test_breakeven_works_on_a_flat_fee_field():
    px = syn.mean_reverting_panel(n=800)
    from amp.validation import breakeven_commission
    cfg = BacktestConfig(commission_percentage=0.0, allow_short=True, no_trade_band=0.05)
    be = breakeven_commission(px, CrossSectionalReversion(top_k=2), cfg,
                              field="commission_per_trade", hi=500.0, tol=1e-3)
    assert be > 0, be


# ----------------------------------------------------------------------- runner


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as exc:
            failures.append((name, exc))
            print("  FAIL  %s -> %s" % (name, exc))
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print("  ERROR %s -> %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
