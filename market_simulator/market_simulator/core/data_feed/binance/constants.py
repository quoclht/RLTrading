import os
from pathlib import Path

PRICE_ZIP_URL = "https://data.binance.vision/data"
CACHE_DATA = "/home/quoclht/Projects/SafeAlpha/cache_data/"


def _get_klines_path():
    base_path = CACHE_DATA + "klines/"

    if not os.path.exists(base_path):
        Path(base_path).mkdir(parents=True, exist_ok=True)

    return base_path


KLINES_PATH = _get_klines_path()