import pandas as pd 
import numpy as np 
import sklearn
from sklearn import svm, linear_model, neighbors, tree, discriminant_analysis, naive_bayes, preprocessing, utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import pickle

#final question: should I buy?
def get_data(ticker):
    df = pd.read_csv('{}.csv'.format(ticker))
    return df

def generate_columns(ticker):
    df = get_data(ticker)

    #machine learning label
    df['Day Change'] = (df['Adj Close'].shift(-1)-df['Adj Close'])
    
    #machine learning features
    df['10ma'] = df['Adj Close'].rolling(window = 10).mean()
    df['100ma'] = df['Adj Close'].rolling(window = 100).mean()
    df['100dac'] = df['Day Change'].shift(1).rolling(window = 100).mean()
    df['10dac'] = df['Day Change'].shift(1).rolling(window=10).mean()

    return df

def generate_feature_set(ticker):
    df = generate_columns(ticker)
    df = df[['Day Change', '10ma', '100ma', '100dac', '10dac']]
    return df

def train_data(ticker):
    df = generate_feature_set(ticker)
    df = df[2000:-1]

    X = np.array(df.drop(['Day Change'], 1)) #features array
    y = np.array(df['Day Change']) # label array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    clf = svm.SVR()

    clf.fit(X_train, y_train)

    pickle.dump(clf, open('stock_clfs/{}.sav'.format(ticker), 'wb'))
    return X_test, y_test

def score_data(ticker):
    X_test, y_test = train_data(ticker)
    clf = pickle.load(open('stock_clfs/{}.sav'.format(ticker), 'rb'))
    confidence = clf.score(X_test, y_test)
    print('Confidence:', confidence)

    predictions = clf.predict(X_test)
    #for x in range(len(predictions)):
    #    print(predictions[x], X_test[x], y_test[x])

score_data('UAL')


