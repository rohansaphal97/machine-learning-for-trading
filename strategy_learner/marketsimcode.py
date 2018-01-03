"""MC2-P1: Market simulator."""
#Name:Amogh Nalwaya, UserID:analwaya3
import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from util import get_data, plot_data

def compute_portvals(orders, start_val = 1000000, commission=0.00, impact=0.00):

    df_orders = orders
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

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def get_cm_return(port_val):
    return (port_val.ix[-1, :] / port_val.ix[0, :]) - 1

def author():
    return 'analwaya3'
