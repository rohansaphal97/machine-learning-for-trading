"""MC2-P1: Market simulator."""
#Name:Amogh Nalwaya, UserID:analwaya3
import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import marketsimcode as ms
from util import get_data, plot_data

def testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normalized = normalize_stocks(prices_SPY)
    prices_normalized = normalize_stocks(prices)

    rolling_days = 20

    #calculate SMA things
    sma = compute_sma(prices_normalized, rolling_days)
    columns = ['Price/SMA']
    prices_sma_ratio = pd.DataFrame(0, index = sma.index, columns = sma.columns)
    prices_sma_ratio['Price/SMA'] = prices_normalized[symbol]/sma['SMA']

    #calculate bollinger things
    bb = compute_bollinger_bands(prices_normalized, rolling_days, sma)
    columns = ['BBP']
    bb_percent = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    bb_percent['BBP'] = (prices_normalized[symbol] - bb['lower']) / (bb['upper'] - bb['lower'])

    #calculate momentum things
    momentum = compute_momentum(prices_normalized, rolling_days)

    orders = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
    buy_sell = pd.DataFrame('BUY', index = prices_normalized.index, columns = ['Order'])
    symbol_df = pd.DataFrame(symbol, index = prices_normalized.index, columns = ['Symbol'])
    total_holdings = 0


    for index, row in prices_normalized.iterrows():
        sma_value = sma.loc[index]['SMA']
        sma_ratio = prices_sma_ratio.loc[index]['Price/SMA']
        bbp_value = bb_percent.loc[index]['BBP']
        momentum_value = momentum.loc[index]['Momentum']
        current_price = row[symbol]
        if (momentum_value < -0.05) and (bbp_value < 0.0) and (sma_ratio < 0.95) and (total_holdings < 1000):
            buy_sell.loc[index]['Order'] = 'BUY'
            if total_holdings == 0:
                orders.loc[index]['Shares'] = 1000
                total_holdings += 1000
            else:
                orders.loc[index]['Shares'] = 2000
                total_holdings += 2000
        elif (momentum_value > 0.05) and (bbp_value > 1) and (sma_ratio > 1.05) and (total_holdings > -1000):
            buy_sell.loc[index]['Order'] = 'SELL'
            if total_holdings == 0:
                orders.loc[index]['Shares'] = 1000
                total_holdings = total_holdings - 1000
            else:
                orders.loc[index]['Shares'] = 2000
                total_holdings = total_holdings - 2000

    df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
    df_trades.columns = ['Symbol', 'Order', 'Shares']
    df_trades = df_trades[df_trades.Shares != 0]

    return df_trades


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

def author():
    return 'analwaya3'


def test_code():
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = sv)

    portvals = ms.compute_portvals(df_trades, start_val = sv)

    syms = ['SPY']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normalized = normalize_stocks(prices_SPY)
    prices_portval_normalized = normalize_stocks(portvals)

    chart_df = pd.concat([prices_portval_normalized, prices_SPY_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(grid=True, title='Comparing Manual strategy with Benchmark index', use_index=True, color=['Black', 'Blue'])

    for index, row in df_trades.iterrows():
        if df_trades.loc[index]['Order'] == 'BUY':
            plt.axvline(x=index, color='g', linestyle='-')
        elif df_trades.loc[index]['Order'] == 'SELL':
            plt.axvline(x=index, color='r', linestyle='-')

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = ms.compute_portfolio_stats(prices_SPY)

    print('In Sample stats:')

    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


    print('Out of Sample Stats:')


    sd=dt.datetime(2010,1,1)
    ed=dt.datetime(2011,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = sv)
    portvals = ms.compute_portvals(df_trades, start_val = sv)

    syms = ['SPY']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normalized = normalize_stocks(prices_SPY)
    prices_portval_normalized = normalize_stocks(portvals)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = ms.compute_portfolio_stats(prices_SPY)

    chart_df = pd.concat([prices_portval_normalized, prices_SPY_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(grid=True, title='Comparing Manual strategy with Benchmark index Out of Sample', use_index=True, color=['Black', 'Blue'])

    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    plt.show()

if __name__ == "__main__":
    test_code()
