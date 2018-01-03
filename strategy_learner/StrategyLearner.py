"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
"""Name:Amogh Nalwaya, UserID:analwaya3"""

"""Implementing trading strategy using a Q-learner"""

import datetime as dt
import numpy as np
import pandas as pd
import random

import QLearner as ql
import marketsimcode as ms
import ManualStrategy as mst
import indicators as ind
import util as ut

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.ql = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)

    def author():
        return 'analwaya3'

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        N = 30
        K = 2 * N
        sd_original = sd

        # add your code to do learning here
        sd = sd - dt.timedelta(K)


        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later


        prices_normalized, prices_sma_ratio, momentum, bb_percent = ind.compute_indicators(sd, ed, syms=[symbol], rolling_days=N)

        indicators = pd.concat([prices_sma_ratio, momentum, bb_percent['BBP']], axis=1)
        indicators = indicators.loc[sd_original:]
        prices_normalized = prices_normalized.loc[sd_original:]

        daily_price_change = self.get_daily_returns(prices_normalized)

        indicators = self.discretize(indicators)

        initial_state = indicators.iloc[0]['state']

        self.ql.querysetstate(int(float(initial_state)))


        orders = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
        buy_sell = pd.DataFrame('BUY', index = prices_normalized.index, columns = ['Order'])
        symbol_df = pd.DataFrame(symbol, index = prices_normalized.index, columns = ['Symbol'])

        df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
        df_trades.columns = ['Symbol', 'Order', 'Shares']

        df_trades_copy = df_trades.copy()

        i = 0

        while i < 500:
            i +=1
            reward = 0
            total_holdings = 0

            if(i > 20) and (df_trades.equals(df_trades_copy)):
                #print(i)
                break

            df_trades_copy = df_trades.copy()

            for index, row in prices_normalized.iterrows():
                reward = total_holdings * daily_price_change.loc[index] * (1 - self.impact)
                a = self.ql.query(int(float(indicators.loc[index]['state'])), reward)
                if(a == 1) and (total_holdings < 1000):
                    buy_sell.loc[index]['Order'] = 'BUY'
                    if total_holdings == 0:
                        orders.loc[index]['Shares'] = 1000
                        total_holdings += 1000
                    else:
                        orders.loc[index]['Shares'] = 2000
                        total_holdings += 2000
                elif (a == 2) and (total_holdings > -1000):
                    buy_sell.loc[index]['Order'] = 'SELL'
                    if total_holdings == 0:
                        orders.loc[index]['Shares'] = -1000
                        total_holdings = total_holdings - 1000
                    else:
                        orders.loc[index]['Shares'] = -2000
                        total_holdings = total_holdings - 2000

            df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
            df_trades.columns = ['Symbol', 'Order', 'Shares']


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        N = 30
        K = N + 30
        sd_original = sd

        # add your code to do learning here
        sd = sd - dt.timedelta(K)


        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later


        prices_normalized, prices_sma_ratio, momentum, bb_percent = ind.compute_indicators(sd, ed, syms=[symbol], rolling_days=N)

        indicators = pd.concat([prices_sma_ratio, momentum, bb_percent['BBP']], axis=1)
        indicators = indicators.loc[sd_original:]
        prices_normalized = prices_normalized.loc[sd_original:]

        daily_price_change = self.get_daily_returns(prices_normalized)

        indicators = self.discretize(indicators)


        orders = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
        buy_sell = pd.DataFrame('BUY', index = prices_normalized.index, columns = ['Order'])
        symbol_df = pd.DataFrame(symbol, index = prices_normalized.index, columns = ['Symbol'])


        reward = 0
        total_holdings = 0

        initial_state = indicators.iloc[0]['state']

        self.ql.querysetstate(int(float(initial_state)))

        for index, row in prices_normalized.iterrows():
            reward = total_holdings * daily_price_change.loc[index]
            #implement action
            a = self.ql.querysetstate(int(float(indicators.loc[index]['state'])))
            if(a == 1) and (total_holdings < 1000):
                buy_sell.loc[index]['Order'] = 'BUY'
                if total_holdings == 0:
                    orders.loc[index]['Shares'] = 1000
                    total_holdings += 1000
                else:
                    orders.loc[index]['Shares'] = 2000
                    total_holdings += 2000
            elif (a == 2) and (total_holdings > -1000):
                buy_sell.loc[index]['Order'] = 'SELL'
                if total_holdings == 0:
                    orders.loc[index]['Shares'] = -1000
                    total_holdings = total_holdings - 1000
                else:
                    orders.loc[index]['Shares'] = -2000
                    total_holdings = total_holdings - 2000

        df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
        df_trades.columns = ['Symbol', 'Order', 'Shares']

        df_trades = df_trades.drop('Symbol', axis=1)
        df_trades = df_trades.drop('Order', axis=1)

        #print(df_trades)

        return df_trades

    def discretize(self, indicators):
        bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        bins_bbp = [-2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2]
        bins_sma = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]
        group_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        indicators['sma_state'] = pd.cut(indicators['Price/SMA'], bins_sma, labels=group_names)
        indicators['momentum_state'] = pd.cut(indicators['Momentum'], bins, labels=group_names)
        indicators['bbp_state'] = pd.cut(indicators['BBP'], bins_bbp, labels=group_names)

        indicators = indicators.drop('Price/SMA', axis=1)
        indicators = indicators.drop('Momentum', axis=1)
        indicators = indicators.drop('BBP', axis=1)
        indicators['state'] = indicators['sma_state'].astype(str) + indicators['momentum_state'].astype(str) + indicators['bbp_state'].astype(str)
        indicators = indicators.drop('sma_state', axis=1)
        indicators = indicators.drop('momentum_state', axis=1)
        indicators = indicators.drop('bbp_state', axis=1)
        return indicators

    def get_daily_returns(self, port_val):
        daily_returns = port_val.copy()
        daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
        return daily_returns


if __name__=="__main__":
    sl = StrategyLearner()
    sl.addEvidence()
    sl.testPolicy()
    print "One does not simply think up a strategy"
