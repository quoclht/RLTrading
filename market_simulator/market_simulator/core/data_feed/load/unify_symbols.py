import pandas
from typing import Dict, List

from market_simulator.core.data_feed.binance.constants import KLINES_PATH


def _load_data(symbols: List[str]) -> Dict[str, pandas.DataFrame]:
    dfs_dict = {}
    for symbol in symbols:
        _df = pandas.read_parquet(f"{KLINES_PATH}/{symbol}.pq")
        dfs_dict[symbol] = _df.sort_values("open_time")

    return dfs_dict


def unify(symbols: List[str]) -> pandas.DataFrame:
    dfs_dict = _load_data(symbols)
    shared_df = pandas.merge(
        dfs_dict[symbols[0]],
        dfs_dict[symbols[1]],
        on="open_time",
        suffixes=(f"_{symbols[0]}", f"_{symbols[1]}"),
        how="outer"
    )
    shared_df.dropna(inplace=True)
    shared_df.sort_values(by='open_time', ignore_index=True, inplace=True)
    return shared_df.reset_index(drop=True)