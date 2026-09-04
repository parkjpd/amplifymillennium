"""amp - trading research toolkit for the AmplifyME Quantify environment."""

from amp.features import build_panel, reversion_map, xs_autocorrelation
from amp.harness import Backtest, BacktestConfig, BacktestResult, compare, scorecard
from amp.strategies.base import BuyAndHold, SingleName
from amp.strategies.xsec_reversion import CrossSectionalReversion
from amp.validation import (
    breakeven_commission,
    commission_sensitivity,
    select_and_validate,
    split,
    sweep,
)

__all__ = [
    "Backtest",
    "BacktestConfig",
    "BacktestResult",
    "BuyAndHold",
    "CrossSectionalReversion",
    "SingleName",
    "breakeven_commission",
    "build_panel",
    "commission_sensitivity",
    "compare",
    "reversion_map",
    "scorecard",
    "select_and_validate",
    "split",
    "sweep",
    "xs_autocorrelation",
]
