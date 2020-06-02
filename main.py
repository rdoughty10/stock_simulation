import pandas as pd
import pandas_datareader as web
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
start = dt.datetime(2000, 1, 1)

today = datetime.today()
end = dt.datetime(today.year, today.month, today.day)

dfInput = web.DataReader('UAL', 'yahoo', start, end)
dfInput.to_csv('ual.csv')

#start reading and organizing the csv file
df = pd.read_csv('ual.csv', parse_dates=True, index_col=0)

#this adds another column for the 100 day rolling average 
df['100ma'] = df['Adj Close'].rolling(window = 100).mean()

#1 day change
df['Day Change'] = (df['Adj Close'].shift(-1)-df['Adj Close']).shift(1)

#average change over 1000 100 days and 10 days
df['1000dac'] = df['Day Change'].rolling(window = 1000).mean()
df['100dac'] = df['Day Change'].rolling(window = 100).mean()
df['10dac'] = df['Day Change'].rolling(window=10).mean()


print(df.tail(100))

#df['100dac'].plot()
#plt.show()

ax1 = plt.subplot2grid((10,1), (0,0), rowspan=5, colspan = 1)
ax2 = plt.subplot2grid((10,1), (5, 0), rowspan=2, colspan = 1, sharex = ax1)
ax3 = plt.subplot2grid((10, 1), (7, 0), rowspan=3, colspan= 1, sharex = ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
ax3.plot(df.index, df['100dac'])
ax3.plot(df.index, df['10dac'])

#plt.show

times_to_buy = df.loc[(df['1000dac']>0) & (df['100dac'] < -0.2) & (df['10dac'] < -0.5)]
print(times_to_buy)
