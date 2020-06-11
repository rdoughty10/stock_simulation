import numpy as np 
import pandas as pd
import pickle
from collections import Counter

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_adjClose_data.csv')
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace = True) # fills nA/NaN values using specified method (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html)
    for i in range(1, hm_days+1):
        df['{}_{}'.format(ticker, i)] = (df[ticker].shift(-i)-df[ticker])/df[ticker] # fills new column with percent change over i days

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.03
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extact_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, df['{}_1d'.format(ticker)] ,df['{}_2d'.format(ticker)], df['{}_3d'.format(ticker)], df['{}_4d'.format(ticker)], df['{}_5d'.format(ticker)], df['{}_6d'.format(ticker)], df['{}7d'.format(ticker)]))

    #this section gets the spread of buy, sells, and holds
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',  Counter(str_vals))

    df.fillna(0, inplace = True)

    df = df.replace([np.inf, -np.inf], np.nan) #replacing infinite changes with NaN
    df.dropna(inplace = True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace = True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

extact_featuresets('XOM')