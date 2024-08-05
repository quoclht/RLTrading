import pandas

from datetime import datetime
from typing import List

def extract_period(open_time: pandas.Series):
    open_time_dt = open_time.apply(
        lambda x: datetime.utcfromtimestamp(float(x) / 1000))
    
    period = (open_time_dt.dt.hour % 24 + 4) // 4 # 4h candle
    dummy_period_df = pandas.get_dummies(period, prefix="period")

    return dummy_period_df

def extract_week_id(open_time: pandas.Series):
    open_time_dt = open_time.apply(
        lambda x: datetime.utcfromtimestamp(float(x) / 1000))
    
    week_id: int = 0
    week_id_col: List[int] = []
    last_update_date: datetime.date = None
    for ot in open_time_dt:
        if ot.day_name() == 'Monday' and ot.date() != last_update_date:
            week_id += 1
            last_update_date = ot.date()

        week_id_col.append(week_id)
    
    return pandas.Series(week_id_col)