"""Creating indicators for the portfolio to create trading strategies from"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def compute_indicators(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), \
    syms=['JPM']):

    # Read in adjusted closing prices for given symbols, date range
    symbol = syms[0]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normalized = normalize_stocks(prices_SPY)
    prices_normalized = normalize_stocks(prices)

    rolling_days = 20

    sma = compute_sma(prices_normalized, rolling_days)
    columns = ['Price/SMA']
    prices_sma_ratio = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    prices_sma_ratio['Price/SMA'] = prices_normalized[symbol]/sma['SMA']
    bb = compute_bollinger_bands(prices_normalized, rolling_days, sma)
    momentum = compute_momentum(prices_normalized, rolling_days)

    sma_plot = pd.concat([prices_normalized, sma, prices_sma_ratio], axis=1)
    sma_plot.columns = [symbol, 'SMA', 'Price/SMA']
    sma_plot.plot(grid=True, title='Simple Moving Average', use_index=True)

    bb_percent = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    bb_percent['BBP'] = (prices_normalized[symbol] - bb['lower']) / (bb['upper'] - bb['lower'])
    bb_plot = pd.concat([prices_normalized, bb['lower'], bb['upper'], bb_percent['BBP']], axis=1)
    bb_plot.columns = [symbol, 'Lower band', 'Upper band', 'BBP']
    bb_plot.plot(grid=True, title='Bollinger Bands', use_index=True)

    momentum_plot = pd.concat([prices_normalized, momentum], axis=1)
    momentum_plot.plot(grid=True, title='Momentum', use_index=True)

    plt.show()


def compute_sma(prices_normalized, rolling_days):
    columns = ['SMA']
    sma = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    sma['SMA'] = prices_normalized.rolling(window=rolling_days, min_periods = rolling_days).mean()
    return sma

def compute_bollinger_bands(prices_normalized, rolling_days, sma, sd=2):
    columns = ['lower', 'upper']
    bb = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    bands = pd.DataFrame(0, index = prices_normalized.index, columns = ['band'])
    bands['band'] = prices_normalized.rolling(window = rolling_days, min_periods = rolling_days).std()
    bb['upper'] = sma['SMA'] + (bands['band'] * sd)
    bb['lower'] = sma['SMA'] - (bands['band'] * sd)
    return bb


def compute_momentum(prices_normalized, rolling_days):
    columns =['Momentum']
    momentum = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    momentum['Momentum'] = prices_normalized.diff(rolling_days)/prices_normalized.shift(rolling_days)
    return momentum

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def test_code():
    compute_indicators()

if __name__ == "__main__":
    test_code()