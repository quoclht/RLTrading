import pandas
from hampel import hampel

def calculate_mvhr(close_price_1: pandas.Series, close_price_2: pandas.Series):
    """
        Minimum Variance Hedge Ratio (MVHR) indicator is the ratio of
    futures position relative to the spot position that minimizes
    the variance of the position. It is calculated as:

                MVHR = -Cov(P_s, P_f)/Var(P_f)

    where where P_s, P_f are return of spot and futures positions respectively
    """

    return_P_s: pandas.Series = close_price_1.diff()
    return_P_f: pandas.Series = close_price_2.diff()
    Corr_P_s_P_f: pandas.Series = return_P_s.rolling(42).corr(return_P_f)
    Std_return_P_s = return_P_s.rolling(42).std()
    Std_return_P_f = return_P_f.rolling(42).std()

    mvhr = (
        Corr_P_s_P_f * Std_return_P_s
    ) / Std_return_P_f  # Minus sign is not included

    return mvhr


def calculate_greeks(close_price_1: pandas.Series, close_price_2: pandas.Series):
    """
        Greeks are the quantities representing the sensitivity of the
    price of derivatives such as options to a change in underlying parameters
    on which the value of an instrument or portfolio of financial instruments is dependent.
    See more: https://en.wikipedia.org/wiki/Greeks_(finance)


    + First order:
        Delta: measures the rate of change of the theoretical option value with respect to
    changes in the underlying asset's price
            Delta = Return of P_f / Return of P_s

        Lambda: is the percentage change in option value per percentage
    change in the underlying price
            Lambda = Delta * (P_f/P_s)

        Vega: measures an sensitivity to implied volatility.
            Vega = Return of P_f / STD of P_s

    + Second order
        Gamma: measures the rate of change in the delta with respect to
    changes in the underlying price
            Gamma = Delta / Return of P_s

        Vanna: measures sensitivity of the option delta with respect
    to change in volatility
            Vanna =  Delta / STD of P_s

        Vomma: measures second order sensitivity to volatility
            Vomma = Vega / STD of P_s

    + Third order
        Speed: measures the rate of change in Gamma with respect to
    changes in the underlying price.
            Speed = Gamma / Return of P_s

        Zomma: measures the rate of change of Gamma with respect to
    changes in volatility.
            Zomma = Gamma / STD of P_s
    """

    P_s: pandas.Series = close_price_1
    P_f: pandas.Series = close_price_2
    std_P_s = P_s.rolling(42).std()

    return_P_s: pandas.Series = P_s.diff()
    return_P_s[return_P_s == 0] = 1

    return_P_f: pandas.Series = P_f.diff()
    return_P_f[return_P_f == 0] = 1

    delta = return_P_f / return_P_s
    delta = hampel(delta, window_size=42 * 2, n=3, imputation=True)

    vega = return_P_f / std_P_s.diff()
    vega = hampel(vega, window_size=42 * 2, n=3, imputation=True)

    return delta, vega
