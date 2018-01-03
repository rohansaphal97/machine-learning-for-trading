"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    normed_prices = normalize_stocks(prices)
    allocated = get_allocated(allocs, normed_prices)
    allocated_amount = get_allocated_amount(allocated, sv)
    port_val = get_portfolio_values(allocated_amount)
    # Add code here to properly compute end value
    ev = port_val[-1]

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val = port_val, \
        allocs=allocs, \
        rfr = rfr, sf = sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], \
            keys=['Portfolio', 'SPY'], \
            axis=1)
        pass


    return cr, adr, sddr, sr, ev

def compute_portfolio_stats(port_val, \
    allocs=[0.1,0.2,0.3,0.4], \
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
    daily_returns = port_val
    daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
    daily_returns = daily_returns[1:]
    return daily_returns

def get_cm_return(port_val):
    return (port_val.ix[-1, :] / port_val.ix[0, :]) - 1

def get_portfolio_values(pos_values):
    return pos_values.sum(axis=1)

def get_allocated_amount(allocated, start_val):
    return allocated * start_val

def get_allocated(allocs, normed_prices):
    return normed_prices * allocs

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,6,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print "End Value:", ev

if __name__ == "__main__":
    test_code()
