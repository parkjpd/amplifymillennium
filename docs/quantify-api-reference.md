# Quantify (AmplifyME) — API Reference

Condensed from https://app.amplifyme.com/quantify/docs. Package is pre-installed in the
AmplifyME Jupyter environment; no install step.

```python
from quantify import TradingEngine, Data, Exchange, Models
```

---

## 1. TradingEngine — `quantify.TradingEngine`

```python
from quantify.TradingEngine import TradingEngine, CurrentPosition, PositionDirection
```

### Constructor

```python
TradingEngine(
    starting_balance: float = 20_000_000.0,
    commission_percentage: float = 0.0,   # 0.001 == 0.1%
    data: pd.DataFrame = None,            # price data
    pivot: bool = True,                   # data is pivoted by ticker
    units: str = "$",
    decimal_places: int = 2,
)
```

### Orders

```python
engine.execute_order(ticker: str, volume: int, action: str, timestamp)   # action: "BUY" | "SELL"
engine("AAPL", 100, "BUY", prices.index[0])                              # callable shorthand
engine.close_all_positions(timestamp=prices.index[-1])
```

`timestamp` **must exist in the price-data index** — trades are priced by lookup, not by
a passed-in price.

### Position access (dunder API)

```python
engine["AAPL"]              # -> CurrentPosition
"AAPL" in engine            # bool
len(engine)                 # number of active positions
engine.current_positions    # dict[str, CurrentPosition]
```

### Properties

| Property | Meaning |
|---|---|
| `balance` | current cash balance |
| `profit_loss` | **net** P&L (after commission) |
| `profit_loss_without_commission` | gross P&L |
| `commission_costs` | total commission paid |
| `current_positions` | dict of all positions by ticker |

### Trade history

```python
engine.get_trade_history("AAPL")
engine.get_all_trade_history()
engine.get_total_trade_count()
engine.has_trades()
```

### CurrentPosition

| Attribute | Type | Meaning |
|---|---|---|
| `ticker` | str | symbol |
| `direction` | `PositionDirection` | LONG / SHORT / FLAT (use `.value` for the string) |
| `position_volume` | int | shares held |
| `open_price` | float | **average** entry price |
| `profit_loss` | float | net of commission |
| `profit_loss_without_commission` | float | gross |
| `commission_costs` | float | commission on this position |
| `trade_history` | dict | full audit trail |

```python
PositionDirection.LONG | PositionDirection.SHORT | PositionDirection.FLAT
```

### Commission

```
commission  = trade_value * commission_percentage
trade_value = shares * price_at_timestamp
```

Commission is charged **per trade**, so it is paid on entry *and* exit. A round trip at
`commission_percentage=0.001` costs ~0.2% of notional — that is the hurdle any signal
must clear.

---

## 2. Data — `quantify.Data`

```python
from quantify.Data import get_price_series, get_price_data, get_price_stream, get_news_series
```

### Historical prices

```python
get_price_series(file_name: str, is_dataframe: bool = False, pivot: bool = True)
get_price_data(dataset: str, round_number: int, is_dataframe: bool = False)   # challenge rounds
```

DataFrame is indexed by timestamp with one column per ticker (when `pivot=True`).

```python
prices  = get_price_series("your_dataset", is_dataframe=True)
aapl    = prices["AAPL"]
returns = prices.pct_change()
latest  = prices.iloc[-1]
```

### Streaming

```python
get_price_stream(file_name: str, is_dataframe: bool = False,
                 execution_times: list = [], base_delay: float = 0)

for index, row, exec_time in get_price_stream("your_dataset", base_delay=1.0):
    ...   # index = timestamp, row = tick prices, exec_time = timing info
```

Blocking generator — replays ticks in wall-clock time. `base_delay` sets a minimum gap
between updates (useful for debugging).

### News

```python
news = get_news_series("your_dataset", is_dataframe=True)
# columns seen in docs: timestamp, ticker, headline
```

---

## 3. Exchange — `quantify.Exchange`

```python
from quantify.Exchange import execute, get_spread

result = execute(trade)                                  # trade: ExchangeTrade -> ExecutedTrade
bid, offer = get_spread(vol: float, ref: float, symbol: str)
```

`ExecutedTrade`: `ticker`, `trade_volume`, `ref_price`, `trade_price`, `action`, `date`.

Spread widens with `vol`; `ref` is the mid/reference price. Slippage = `trade_price - ref_price`.

---

## 4. Models — `quantify.Models`

```python
from quantify.Models import lgbm, run_arma, get_ml_metrics
```

### LightGBM

```python
model, metrics = lgbm(X, y, features=None, target=None, test_size=0.2)
y_pred = model.predict(X_test)
```

Returns `(trained Booster, metrics_dict)`. Booster `.predict()` on a binary objective
returns **probabilities**, not classes — threshold them.

### ARMA

```python
result = run_arma(timeseries)   # -> dict of model results + forecasts
```

Order selection is left to the caller (ACF/PACF, AIC/BIC grid search).

### Metrics

```python
metrics = get_ml_metrics(model, X_test, y_test)
```

---

## 5. LOB — `quantify.LOB`

```python
from quantify.LOB import Order

o = Order(price=150.00, volume=100, side="BUY", order_id="order_001")
# attributes: price, volume (remaining), side, order_id, timestamp
```

Book operations (add / best bid / best offer / match) are **not** provided — you build them.

---

## 6. MarketMaker — `quantify.MarketMaker`

```python
from quantify.MarketMaker import mm

maker = mm(stocks=["AAPL", "MSFT", "GOOGL"])
result = maker.add_quoted_trade(quoted_trade)
maker.update_current_positions(completed_trade, mm_action)
maker.current_positions   # dict[ticker, position]
maker.quoted_trades       # outstanding quotes
maker.completed_trades    # executed
```

---

## 7. HedgeFund — `quantify.HedgeFund`

```python
from quantify.HedgeFund import show

response = show(quoted_trade)   # -> HfResponse
```

`HfResponse`: `ticker`, `trade_volume`, `trade_price`, `hf_action`, `ref_price`,
`bid_price`, `offer_price`, `date`. The counterparty decides whether to lift your quote.

---

## 8. SentimentAnalysis — `quantify.SentimentAnalysis`

```python
from quantify.SentimentAnalysis import sentiment_label, sentiment_score

sentiment_label(text)   # "Positive" | "Negative" | "Neutral"
sentiment_score(text)   # float in [-1.0, 1.0]
```

---

## 9. Weather — `quantify.Weather`

```python
from quantify.Weather import plot_region_forecasts, plot_heatmap

plot_region_forecasts(data, regions=["Northeast", ...], variable="temperature")
plot_heatmap(data, variable="precipitation")
```

Visualisation only. Use cases: power demand, crop yields, heating/cooling degree days.

---

## 10. TraderLink — `quantify.TraderLink`

Graded data-engineering challenges. Each sub-module ships **only** a plot helper — the
logic is yours, and output is validated against an expected format.

```python
from quantify.TraderLink.event_gaps          import plot_gaps          # plot_gaps(events)
from quantify.TraderLink.investable_universe import plot_prices        # plot_prices(securities)
from quantify.TraderLink.multi_venue_quote   import plot_venue_price   # plot_venue_price(quotes)
from quantify.TraderLink.ticks_to_ohlcv      import plot_trades        # plot_trades(ticks)
```

### Event gaps
Find periods where expected data points are missing. Used for trading halts, data-quality
issues, market-close periods, feed connectivity.
Approach: parse timestamps -> diff consecutive events -> flag diffs over a threshold ->
report gap start, end, duration.

### Investable universe
Filter a security master by liquidity (min volume / dollar volume), price (exclude penny
stocks), market cap, and sector.
Approach: load master -> apply filters -> return filtered set -> validate against the
expected output format.

### Multi-venue quote (NBBO)
Consolidate quotes across venues into the National Best Bid/Offer.

Input columns: `venue`, `symbol`, `bid_price`, `bid_size`, `ask_price`, `ask_size`,
`timestamp`.

Approach: for each timestamp, highest bid across venues and lowest ask across venues.
Watch for locked (bid == ask) and crossed (bid > ask) books, and for stale venues that
never update.

### Ticks to OHLCV
Aggregate tick trades into candlestick bars.

Output columns: `open` (first price), `high`, `low`, `close` (last price), `volume` (sum).
Common intervals: 1/5/15/30min, 1h, 4h, 1d.

```
Timestamp          | Open   | High   | Low    | Close  | Volume
2024-01-01 09:30   | 150.00 | 150.50 | 149.80 | 150.25 | 12500
2024-01-01 09:35   | 150.30 | 151.00 | 150.10 | 150.90 | 8700
```

Edge cases called out by the docs: empty intervals and single-tick intervals.

## 11. Canonical patterns

### Backtest loop (MA crossover)

```python
prices = get_price_series("your_dataset", is_dataframe=True)
engine = TradingEngine(data=prices, starting_balance=1_000_000, commission_percentage=0.001)

short_ma = prices["AAPL"].rolling(5).mean()
long_ma  = prices["AAPL"].rolling(20).mean()

for i in range(20, len(prices)):
    ts = prices.index[i]
    if short_ma.iloc[i] > long_ma.iloc[i]:
        if "AAPL" not in engine or engine["AAPL"].direction.value == "FLAT":
            engine.execute_order("AAPL", 100, "BUY", ts)
    elif short_ma.iloc[i] < long_ma.iloc[i]:
        if "AAPL" in engine and engine["AAPL"].direction.value == "LONG":
            engine.execute_order("AAPL", 100, "SELL", ts)

engine.close_all_positions(prices.index[-1])
print(engine.profit_loss, engine.get_total_trade_count())
```

### ML pipeline

```python
df = pd.DataFrame()
df["returns"] = prices["AAPL"].pct_change()
df["ma_5"]    = prices["AAPL"].rolling(5).mean()
df["ma_20"]   = prices["AAPL"].rolling(20).mean()
df["target"]  = (df["returns"].shift(-1) > 0).astype(int)
df = df.dropna()

n = int(len(df) * 0.8)                      # chronological split — never shuffle
feats = ["returns", "ma_5", "ma_20"]
model, train_metrics = lgbm(df[feats][:n], df["target"][:n])
test_metrics = get_ml_metrics(model, df[feats][n:], df["target"][n:])
```

### Sentiment-driven trading

```python
for _, article in news.iterrows():
    if sentiment_score(article["headline"]) > 0.5:
        engine.execute_order(article["ticker"], 10, "BUY", article["timestamp"])
```

---

## 12. Gaps in the official docs (verify in the notebook before relying on these)

1. **Shorting.** `PositionDirection.SHORT` exists, but no example sells without an open
   long. Unknown whether a naked SELL opens a short or is rejected. **Test first.**
2. **Return value of `execute_order`.** Undocumented — check whether it returns a fill
   object or raises on rejection (insufficient balance, unknown timestamp).
3. **Streaming + engine pricing.** The streaming example builds
   `TradingEngine(starting_balance=...)` with **no `data=`**, then the sentiment example
   calls `execute_order` on it. How the engine prices a trade with no `data` is
   unspecified — likely needs `data=` anyway, or prices come from the exchange.
4. **`lgbm` signature.** `X, y` positional plus `features`/`target` names implies two
   calling styles (pre-split frames vs. one frame + column names). Docs only show the
   first.
5. **`ExchangeTrade` constructor** is never shown — only `execute(trade)`. Same for the
   `quoted_trade` objects passed to `mm.add_quoted_trade` and `HedgeFund.show`.
6. **Position sizing / leverage.** No documented margin rules or rejection on
   insufficient balance.
7. **Timestamp alignment.** News/weather timestamps must be snapped onto the price index
   before they can be used in `execute_order`.
8. **Fill price.** Trades appear to fill at the price at `timestamp` — i.e. the bar you
   are making the decision on. Guard against look-ahead by acting on `index[i+1]`.
