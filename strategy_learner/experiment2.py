#Name:Amogh Nalwaya, UserID:analwaya3
import datetime as dt
import numpy as np
import pandas as pd
import random

import QLearner as ql
import marketsimcode as ms
import ManualStrategy as mst
import indicators as ind
import util as ut
import StrategyLearner as sl
from matplotlib import cm as cm
from matplotlib import style
import matplotlib.pyplot as plt

def plotStuff():
    slr = sl.StrategyLearner(verbose = False, impact=0.05)
    slr.addEvidence(symbol = "JPM", )
    df_trades_sl = slr.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 100000)
    df_trades_sl['Symbol'] = 'JPM'
    df_trades_sl['Order'] = 'BUY'
    df_trades_sl.loc[df_trades_sl.Shares < 0, 'Order'] = 'SELL'
    df_trades_sl = df_trades_sl[df_trades_sl.Shares != 0]
    df_trades_sl = df_trades_sl[['Symbol', 'Order', 'Shares']]

    portvals_sl = ms.compute_portvals(df_trades_sl, start_val = 100000, impact=0.05)

    df_trades = mst.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 100000)
    portvals = ms.compute_portvals(df_trades, start_val = 100000, impact=0.05)

    syms = ['SPY']
    dates = pd.date_range(dt.datetime(2008,1,1), dt.datetime(2009,1,1))
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normalized = normalize_stocks(prices_SPY)
    prices_portval_normalized = normalize_stocks(portvals)
    prices_sl_normalized = normalize_stocks(portvals_sl)

    chart_df = pd.concat([prices_portval_normalized, prices_SPY_normalized, prices_sl_normalized], axis=1)
    chart_df.columns = ['Manual Strategy', 'Benchmark', 'Strategy Learner']
    chart_df.plot(grid=True, title='Comparing Manual strategy with Strategy Learner', use_index=True, color=['Black', 'Blue', 'Red'])


    cum_ret_sl, avg_daily_ret_sl, std_daily_ret_sl, sharpe_ratio_sl = ms.compute_portfolio_stats(prices_sl_normalized)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(prices_portval_normalized)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = ms.compute_portfolio_stats(prices_SPY_normalized)
    print('Sharpe Ratio Strategy Learner:{}'.format(sharpe_ratio_sl))
    print('Sharpe Ratio Manual Strategy:{}'.format(sharpe_ratio))
    print('Sharpe Ratio Benchmark:{}'.format(sharpe_ratio_SPY))

    print('Cum Return Strategy Learner:{}'.format(cum_ret_sl))
    print('Cum Return Manual Strategy:{}'.format(cum_ret))
    print('Cum Return Benchmark:{}'.format(cum_ret_SPY))
    plt.show()

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def author():
    #Name:Amogh Nalwaya
    return 'analwaya3'

if __name__ == "__main__":
    plotStuff()