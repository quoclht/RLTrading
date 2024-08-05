import numpy

from typing import List

from market_simulator.core.data_feed.binance import kline
from market_simulator.core.data_feed.load import unify_symbols
from market_simulator.core.data_feed.transform import hedge, trading_period


def update(symbols: List[str], with_zip: bool, interval: str = "4h"):
    for symbol in symbols:
        kline_fetch = kline.KLineDataFeed(symbol=symbol, interval=interval)
        kline_fetch.update(with_zip=with_zip)


def generate_dataset(symbols: List[str]):
    df = unify_symbols.unify(symbols=symbols)
    df["week_id"] = numpy.copy(
        trading_period.extract_week_id(open_time=df["open_time"])
    )

    df['basis'] = numpy.copy(df[f"close_{symbols[0]}"] - df[f"close_{symbols[1]}"])

    df['mvhr'] = numpy.copy(
        hedge.calculate_mvhr(
            close_price_1=df[f"close_{symbols[0]}"],
            close_price_2=df[f"close_{symbols[1]}"],
        )
    )

    delta, vega = \
        hedge.calculate_greeks(
            close_price_1=df[f"close_{symbols[0]}"],
            close_price_2=df[f"close_{symbols[1]}"],

    )

    df['delta'] = numpy.copy(delta)
    df['vega'] = numpy.copy(vega)

    return df.dropna().reset_index(drop=True)
