import cython
from cython.cimports.libc import math

from decimal import Decimal


STEP_SIZE = {
    "BTCUSDT": 0.0001,
    "ETHUSDT": 0.001,
}

TICK_SIZE = {
    "BTCUSDT": 1,
    "ETHUSDT": 0.1,
}

MIN_ORDER_SIZE = {
    "BTCUSDT": 0.001,
    "ETHUSDT": 0.01,
}

def round_number(val: cython.float) -> cython.float:
    return math.round(val * 100000)/100000

def size(
        symbol: cython.basestring,
        size: cython.float,
        is_tick: cython.int) -> cython.float:

    step_size: cython.float = -99

    if is_tick == 1:
        step_size = TICK_SIZE[symbol]
    else:
        step_size = STEP_SIZE[symbol]

    min_order_size: cython.float = MIN_ORDER_SIZE[symbol]

    _size_: cython.float = size
    _size_moded_: cython.float = cython.cmod(_size_, step_size)
    if _size_ > 0:
        _size_ = _size_ - _size_moded_
    elif size < 0:
        _size_ = _size_ + math.fabsf(_size_moded_)

    if math.fabsf(_size_) > min_order_size:
        return _size_
    else:
        return 0
