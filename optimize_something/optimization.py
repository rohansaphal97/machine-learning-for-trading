"""MC1-P2: Optimize a portfolio."""
"""Using optimization functions within Scipy library to optimize a portfolio
    for volatility"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data

def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY = normalize_stocks(prices_SPY)

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    #Get optimal allocation and associated sddr
    allocs, sddr = min_volatility(syms, prices, sd, ed)

    #Compute other portfolio stats and potfolio value for plotting
    cr, adr, sr, port_val = compute_other_portfolio_stats(prices, sddr, allocs, \
        rfr=0.0, sf=252)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], \
            axis=1)
        df_temp = df_temp.reset_index(drop=True)
        df_plot = df_temp.plot(grid=True, title='Daily Potfolio Value and SPY', \
            use_index=True)
        df_plot.set_xlabel("Date")
        df_plot.set_ylabel("Price")
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr

#Function to cal portfolio stats other than sddr
def compute_other_portfolio_stats(prices, sddr, \
    allocs, \
    rfr = 0.0, sf = 252.0):

    normed_prices = normalize_stocks(prices)
    allocated = get_allocated(allocs, normed_prices)
    port_val = get_portfolio_values(allocated)

    daily_returns = get_daily_returns(port_val)

    cr = get_cm_return(port_val)
    daily_returns = get_daily_returns(port_val)
    adr = daily_returns.mean()

    #calculating sr
    diff_returns = daily_returns - rfr
    diff_returns_mean = diff_returns.mean()

    sr = np.sqrt(sf) * (diff_returns_mean / sddr)
    return cr, adr, sr, port_val

#Function returning the sddr value (which needs to be minimized)
def assess_portfolio_std(allocs, prices, sd, ed, syms):

    normed_prices = normalize_stocks(prices)
    allocated = get_allocated(allocs, normed_prices)
    port_val = get_portfolio_values(allocated)

    daily_returns = get_daily_returns(port_val)
    sddr = daily_returns.std()

    return sddr

#Function which returns optimal allocation and minimum sddr value
def min_volatility(syms, prices, sd, ed):
    n_syms = len(syms)
    allocs = np.ones((1, n_syms)) / n_syms
    allocs = np.asarray(allocs)
    bounds = [(0.0, 1.0) for _ in range(n_syms)]
    cons = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    options = {'disp': True}
    min_result = spo.minimize(assess_portfolio_std, allocs, args=(prices, sd, ed, syms,), constraints = cons, bounds = bounds, method='SLSQP')
    return min_result.x, min_result.fun

def get_cm_return(port_val):
    return (port_val.ix[-1, :] / port_val.ix[0, :]) - 1

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def get_portfolio_values(pos_values):
    return pos_values.sum(axis=1)

def get_allocated(allocs, normed_prices):
    return normed_prices * allocs

def get_daily_returns(port_val):
    daily_returns = port_val.copy()
    daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
    daily_returns = daily_returns[1:]
    return daily_returns

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()