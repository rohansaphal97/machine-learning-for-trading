"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df_orders.sort_index(inplace=True)

    dates = pd.date_range(df_orders.first_valid_index(), df_orders.last_valid_index())

    syms = np.array(df_orders.Symbol.unique()).tolist()
    df_prices = get_data(syms, dates)
    columns = ['Cash']
    df_cash = pd.DataFrame(index=dates, columns=columns)
    df_cash = df_cash.fillna(1.0)
    df_prices = df_prices.join(df_cash)


    df_trades = create_trades(df_orders, df_prices, commission, impact)

    df_holdings = create_holdings(df_trades, start_val)

    df_values = create_values(df_prices, df_holdings)

    df_portval = cal_portval(df_values)

    return df_portval


def create_trades(df_orders, df_prices, commission, impact):
    df_trades = pd.DataFrame(0.0, columns=df_prices.columns, index=df_prices.index)
    columns = ['Commission']
    df_commission = pd.DataFrame(index=df_prices.index, columns=columns)
    df_commission = df_commission.fillna(0.0)
    columns_2 = ['Impact']
    df_impact = pd.DataFrame(index=df_prices.index, columns=columns_2)
    df_impact = df_impact.fillna(0.0)
    for index, row in df_orders.iterrows():
        sym = row['Symbol']
        shares = row['Shares']
        a = -1
        if (row['Order'] == 'BUY'):
            a = 1
        df_trades.loc[index][sym] = df_trades.loc[index][sym] + (a * shares)
        df_commission.loc[index]['Commission'] = df_commission.loc[index]['Commission'] + commission
        df_impact.loc[index]['Impact'] = df_impact.loc[index]['Impact'] + (df_prices.loc[index][sym] * shares * impact)

    df_temp = (df_prices * df_trades)

    df_trades['Cash'] = (-1.0 * df_temp.sum(axis = 1))

    df_trades['Cash'] = df_trades['Cash'] - df_commission['Commission'] - df_impact['Impact']

    return df_trades

def create_holdings(df_trades, start_val):
    start_date = df_trades.first_valid_index()
    df_holdings = pd.DataFrame(0.0, columns=df_trades.columns, index=df_trades.index)
    df_holdings.loc[start_date, 'Cash'] = start_val
    df_holdings = df_holdings + df_trades
    df_holdings = df_holdings.cumsum()
    return df_holdings

def create_values(df_prices, df_holdings):
    return df_prices * df_holdings

def cal_portval(df_values):
    return df_values.sum(axis = 1)

def compute_portfolio_stats(port_val, \
    rfr = 0.0, sf = 252.0):
    cr = get_cm_return(port_val)
    daily_returns = get_daily_returns(port_val)
    adr = daily_returns.mean()
    sddr = daily_returns.std()

    #calculating sr
    diff_returns = daily_returns - rfr
    diff_returns_mean = diff_returns.mean()

    sr = np.sqrt(sf) * (diff_returns_mean / sddr)
    return cr, adr, sddr, sr

def get_daily_returns(port_val):
    daily_returns = port_val.copy()
    daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
    daily_returns = daily_returns[1:]
    return daily_returns

def get_cm_return(port_val):
    return (port_val.ix[-1, :] / port_val.ix[0, :]) - 1

def author():
    return 'analwaya3'


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-12.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2008,6,1)
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)

    # Compare portfolio against $SPX
    #print "Date Range: {} to {}".format(start_date, end_date)
    #print
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

if __name__ == "__main__":
    test_code()
