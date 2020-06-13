import numpy as np 
import pandas as pd
import pickle
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_adjClose_data.csv')
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace = True) # fills nA/NaN values using specified method (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i)-df[ticker])/df[ticker] # fills new column with percent change over i days

    df.fillna(0, inplace=True)
    return tickers, df, hm_days

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.027
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extact_featuresets(ticker):
    tickers, df, hm_days = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)])) # list comprehension

    #print(df.head())

    #this section gets the spread of buy, sells, and holds
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',  Counter(str_vals))

    df.fillna(0, inplace = True)

    df = df.replace([np.inf, -np.inf], np.nan) #replacing infinite changes with NaN
    df.dropna(inplace = True)

    #print(tickers)
    df_vals = df[[ticker for ticker in tickers[1:]]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace = True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extact_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)  #this gives us our training and testing groups automatically

    #clf = neighbors.KNeighborsClassifier() #one specific type of classifier trainer (look up)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),('knn', neighbors.KNeighborsClassifier()),('rfor', RandomForestClassifier())]) # this takes in 3 different classifiers from scikit and votes on them
    

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test) #tells us our confidence
    predictions = clf.predict(X_test)
    print('Accuracy:', confidence)
    print('Predicted Spread:', Counter(predictions))

    return confidence

do_ml('BAC')