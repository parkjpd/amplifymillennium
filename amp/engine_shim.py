"""Local stand-in for ``quantify.TradingEngine``.

The real engine only exists inside the AmplifyME Jupyter environment, so this mirrors
its documented public API well enough to test harness and strategy logic offline. It is
a **test double, not a replacement** — the accounting rules below are inferred from the
docs, not verified against the real engine, so headline P&L from a shim run is
indicative only. Anything that matters gets re-run in the notebook against the real
``TradingEngine``.

Assumptions made where the docs are silent (see docs/quantify-api-reference.md §12):
  * a SELL with no open long opens a SHORT (set ``allow_short=False`` to reject instead)
  * commission = |volume| * price * commission_percentage, charged on entry AND exit
  * ``profit_loss`` is realised P&L plus unrealised marked at the last seen timestamp
  * ``balance`` is cash: it falls on a BUY and rises on a SELL
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class PositionDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class CurrentPosition:
    """Mirrors the documented ``CurrentPosition`` attribute surface."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.signed_volume: int = 0
        self.open_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.commission_costs: float = 0.0
        self.trade_history: Dict[int, Dict[str, Any]] = {}
        self._mark: float = 0.0

    # -- documented surface -------------------------------------------------
    @property
    def position_volume(self) -> int:
        return abs(self.signed_volume)

    @property
    def direction(self) -> PositionDirection:
        if self.signed_volume > 0:
            return PositionDirection.LONG
        if self.signed_volume < 0:
            return PositionDirection.SHORT
        return PositionDirection.FLAT

    @property
    def unrealized_pnl(self) -> float:
        if self.signed_volume == 0:
            return 0.0
        return (self._mark - self.open_price) * self.signed_volume

    @property
    def profit_loss_without_commission(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def profit_loss(self) -> float:
        return self.profit_loss_without_commission - self.commission_costs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "CurrentPosition(%s, %s, %d @ %.4f)" % (
            self.ticker, self.direction.value, self.position_volume, self.open_price
        )


class TradingEngine:
    """Offline stand-in for the Quantify TradingEngine."""

    def __init__(
        self,
        starting_balance: float = 20_000_000.0,
        commission_percentage: float = 0.0,
        data: Optional[pd.DataFrame] = None,
        pivot: bool = True,
        units: str = "$",
        decimal_places: int = 2,
        allow_short: bool = True,
        commission_per_trade: float = 0.0,
        commission_per_share: float = 0.0,
    ) -> None:
        if data is None:
            raise ValueError("shim requires data= to price trades")
        self.data = data
        self.pivot = pivot
        self.units = units
        self.decimal_places = decimal_places
        self.starting_balance = float(starting_balance)
        self.commission_percentage = float(commission_percentage)
        # Not in the documented Quantify API; here so the shim can imitate a challenge
        # that charges flat fees. Both default to zero, matching the docs.
        self.commission_per_trade = float(commission_per_trade)
        self.commission_per_share = float(commission_per_share)
        self.allow_short = allow_short

        self._cash = float(starting_balance)
        self._positions: Dict[str, CurrentPosition] = {}
        self._trade_seq = 0

    # -- pricing ------------------------------------------------------------
    def price(self, ticker: str, timestamp: Any) -> float:
        try:
            value = self.data.at[timestamp, ticker]
        except KeyError as exc:
            raise KeyError(
                "no price for %r at %r (timestamp must exist in the price index)"
                % (ticker, timestamp)
            ) from exc
        if pd.isna(value):
            raise ValueError("price for %r at %r is NaN" % (ticker, timestamp))
        return float(value)

    def _mark_all(self, timestamp: Any) -> None:
        for ticker, pos in self._positions.items():
            try:
                pos._mark = self.price(ticker, timestamp)
            except (KeyError, ValueError):
                pass

    # -- orders -------------------------------------------------------------
    def execute_order(self, ticker: str, volume: int, action: str, timestamp: Any) -> Dict[str, Any]:
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError("action must be BUY or SELL, got %r" % action)
        volume = int(volume)
        if volume <= 0:
            raise ValueError("volume must be positive, got %r" % volume)

        price = self.price(ticker, timestamp)
        signed = volume if action == "BUY" else -volume
        pos = self._positions.setdefault(ticker, CurrentPosition(ticker))

        if not self.allow_short and pos.signed_volume + signed < 0:
            raise ValueError(
                "shorting disabled: %s %d would take %s to %d"
                % (action, volume, ticker, pos.signed_volume + signed)
            )

        commission = (
            abs(signed) * price * self.commission_percentage
            + abs(signed) * self.commission_per_share
            + self.commission_per_trade
        )
        self._apply_fill(pos, signed, price)

        self._cash -= signed * price
        self._cash -= commission
        pos.commission_costs += commission

        self._trade_seq += 1
        record = {
            "trade_id": self._trade_seq,
            "ticker": ticker,
            "volume": volume,
            "action": action,
            "timestamp": timestamp,
            "price": price,
            "commission": commission,
            "position_after": pos.signed_volume,
        }
        pos.trade_history[self._trade_seq] = record
        self._mark_all(timestamp)
        return record

    @staticmethod
    def _apply_fill(pos: CurrentPosition, signed: int, price: float) -> None:
        qty = pos.signed_volume
        if qty == 0 or (qty > 0) == (signed > 0):
            # opening or adding: roll the average entry price
            new_qty = qty + signed
            pos.open_price = (pos.open_price * qty + price * signed) / new_qty
            pos.signed_volume = new_qty
            return

        # reducing, closing, or reversing
        closing = min(abs(signed), abs(qty))
        pos.realized_pnl += (price - pos.open_price) * closing * (1 if qty > 0 else -1)
        new_qty = qty + signed
        if new_qty == 0:
            pos.open_price = 0.0
        elif (new_qty > 0) != (qty > 0):
            pos.open_price = price  # flipped through zero
        pos.signed_volume = new_qty

    def __call__(self, ticker: str, volume: int, action: str, timestamp: Any) -> Dict[str, Any]:
        return self.execute_order(ticker, volume, action, timestamp)

    def close_all_positions(self, timestamp: Any) -> None:
        for ticker, pos in list(self._positions.items()):
            if pos.signed_volume == 0:
                continue
            action = "SELL" if pos.signed_volume > 0 else "BUY"
            self.execute_order(ticker, abs(pos.signed_volume), action, timestamp)

    # -- position access ----------------------------------------------------
    def __getitem__(self, ticker: str) -> CurrentPosition:
        return self._positions[ticker]

    def __contains__(self, ticker: str) -> bool:
        pos = self._positions.get(ticker)
        return pos is not None and pos.signed_volume != 0

    def __len__(self) -> int:
        return sum(1 for p in self._positions.values() if p.signed_volume != 0)

    @property
    def current_positions(self) -> Dict[str, CurrentPosition]:
        return {t: p for t, p in self._positions.items() if p.signed_volume != 0}

    # -- accounting ---------------------------------------------------------
    @property
    def balance(self) -> float:
        return self._cash

    @property
    def commission_costs(self) -> float:
        return sum(p.commission_costs for p in self._positions.values())

    @property
    def profit_loss_without_commission(self) -> float:
        return sum(p.profit_loss_without_commission for p in self._positions.values())

    @property
    def profit_loss(self) -> float:
        return self.profit_loss_without_commission - self.commission_costs

    # -- history ------------------------------------------------------------
    def get_trade_history(self, ticker: str) -> Dict[int, Dict[str, Any]]:
        pos = self._positions.get(ticker)
        return dict(pos.trade_history) if pos else {}

    def get_all_trade_history(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pos in self._positions.values():
            out.extend(pos.trade_history.values())
        return sorted(out, key=lambda r: r["trade_id"])

    def get_total_trade_count(self) -> int:
        return self._trade_seq

    def has_trades(self) -> bool:
        return self._trade_seq > 0
