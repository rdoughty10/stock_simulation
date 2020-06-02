#this file takes in data and retrieves the close price and volume for every single stock in the symbols.txt list and puts them into csv files to be used in main program

import pandas_datareader as web

stocks = []
f = open("symbols.txt", "r")
for line in f:
    stocks.append(line.strip())

f.close()

web.DataReader(stocks, "yahoo", start="2000-1-1", end="2020-5-29")["Adj Close"].to_csv("prices.csv")
web.DataReader(stocks, "yahoo", start="2000-1-1", end="2019-5-29")["Volume"].to_csv("volume.csv")

