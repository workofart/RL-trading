# Trading Environment

## Spread Trading `trading_spread.py`

`SpreadTrading` is a trading environment allowing to trade a *spread* (see https://en.wikipedia.org/wiki/Spread_trade). We feed the environment a time series of prices (bid and ask) for *n* different products (with a `DataGenerator`), as well as a list of *spread coefficients*. The state of the environment is defined as: prices, entry price and position (whether long, short or flat).

## Tick Trading `trading_tick.py`
`TickTrading` is a trading environment allowing to trade a *tick*. We feed the environment a time series of prices for *n* different products (with a `DataGenerator`). The state of the environment is defined as: prices, entry price and position (whether long, short or flat).

## Common for all Environments
The possible actions are then buying, selling or holding the spread.