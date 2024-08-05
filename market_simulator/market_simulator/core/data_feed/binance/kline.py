import time
from datetime import datetime
from glob import glob


import pandas
from binance.spot import Spot

from market_simulator.core.data_feed.common.request_data import download_file
from market_simulator.core.data_feed.binance.constants import PRICE_ZIP_URL, KLINES_PATH
from typing import Tuple
# from configs import data_source, path_file


class KLineDataFeed:

    def _get_url(self, year: str, month: str, day: str = None) -> Tuple[str, str]:

        file_name = (
            f"{self.symbol}-{self.interval}-{year}-{month}-{day}.zip"
            if day is not None
            else f"{self.symbol}-{self.interval}-{year}-{month}.zip"
        )

        url = f"{PRICE_ZIP_URL}/{self.market}/{self.period}/{self.candle_type}/{self.symbol}/{self.interval}/{file_name}"

        return url, file_name

    def _fetch_zip_file(self):

        current_year = datetime.now().year
        current_month = datetime.now().month

        year_iter = range(2021, current_year + 1)
        month_iter = range(1, 12 + 1)

        for yi in year_iter:
            for mi in month_iter:
                if yi == current_year and mi == current_month:
                    return 0
                else:
                    file_url, file_name = self._get_url(
                        year=str(yi).zfill(2),
                        month=str(mi).zfill(2)
                    )

                    file_path = f"{KLINES_PATH}/{file_name}"
                    download_file(
                        file_url=file_url,
                        file_path=file_path
                    )
                    print(f"Downloaded {file_name}")
                    time.sleep(2)

    def _fetch_api(self):
        client = Spot()
        mk_data = client.klines(symbol=self.symbol, interval=self.interval)
        mk_data = pandas.DataFrame(data=mk_data)

        file_name = f"{self.symbol}{self.interval}-api.csv"
        full_path = f"{KLINES_PATH}/{file_name}"
        mk_data.to_csv(full_path)
        time.sleep(2)

    def __init__(self, symbol: str, interval: str, market: str = "spot") -> None:

        self.symbol = symbol
        self.interval = interval
        self.market = market
        self.candle_type = "klines"
        self.period = "monthly"


    def _merge_data(self):
        file_name = f"{self.symbol}{self.interval}-api.csv"
        COLUMNS = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]

        # Read ZIP files
        df = pandas.DataFrame(columns=COLUMNS)
        for f in glob(f"{KLINES_PATH}/{self.symbol}-*.zip"):
            _df_ = pandas.read_csv(f, compression="zip")

            try:
                float(_df_.columns[0])
                _df_f_ = pandas.read_csv(f, compression="zip", header=None, names=COLUMNS)
                df = pandas.concat([df, _df_f_])

            except Exception:
                _df = pandas.read_csv(
                    f,
                    compression="zip",
                )
                assert len(df.columns) == len(
                    _df.columns
                ), "Expect columns have same length!"
                df = pandas.concat([df, _df])
        
        # Read API file
        _df_api = pandas.read_csv(
            f"{KLINES_PATH}/{file_name}",
            header=None,
            names=COLUMNS,
            skiprows=1
        )

        df = pandas.concat([df, _df_api])

        df.drop_duplicates(ignore_index=True, inplace=True)

        # Save merged data
        merged_file_full_path = f"{KLINES_PATH}/{self.symbol}.pq"
        df.to_parquet(merged_file_full_path, index=False)

    def update(self, with_zip: bool):
        if with_zip:
            self._fetch_zip_file()
        
        self._fetch_api()
        self._merge_data()