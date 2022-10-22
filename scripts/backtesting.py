
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

preprocessed_path = "/Users/egemenokur/PycharmProjects/D4PG_testingafternewclone/CSVs/State_Actions_Final.csv"

data = pd.read_csv(preprocessed_path)


#train[train.index % 3 == 0]  # Ex

"""
BuyStock1 = data['5']*0
SellStock1 = data['5']*0

BuyStock2 = data['5']*0
SellStock2 = data['5']*0

BuyStock3 = data['5']*0
SellStock3 = data['5']*0
"""

BuyStock1D = data['0'][data['4']>0]
BuyStock1P = data['8'][data['4']>0]
SellStock1D = data['0'][data['4']<0]
SellStock1P = data['8'][data['4']<0]

BuyStock2D = data['0'][data['5']>0]
BuyStock2P = data['9'][data['5']>0]
SellStock2D = data['0'][data['5']<0]
SellStock2P = data['9'][data['5']<0]

BuyStock3D = data['0'][data['6']>0]
BuyStock3P = data['10'][data['6']>0]
SellStock3D = data['0'][data['6']<0]
SellStock3P = data['10'][data['6']<0]

print([data['7']])

PriceData1 = data['8']
PriceData2 = data['9']
PriceData3 = data['10']
frame =data['0']

fig, ax1 = plt.subplots()
# Define position of 1st subplot

# Set the title and axis labels
plt.title('Backtesting')
plt.xlabel('Frames')

lns1 = ax1.plot(BuyStock1D,BuyStock1P,'ro',label='Buy',color='g', )
lns2 = ax1.plot(SellStock1D,SellStock1P,'ro',label='Sell',color='r')
lns3 = ax1.plot(BuyStock2D,BuyStock2P,'ro',label='Buy',color='g', )
lns4 = ax1.plot(SellStock2D,SellStock2P,'ro',label='Sell',color='r')
lns5 = ax1.plot(BuyStock3D,BuyStock3P,'ro',label='Buy',color='g', )
lns6 = ax1.plot(SellStock3D,SellStock3P,'ro',label='Sell',color='r')

lns01 = ax1.plot(frame,PriceData1,label='AAPL',color='b')
lns02 = ax1.plot(frame,PriceData2,label='MSFT',color='m')
lns03 = ax1.plot(frame,PriceData3,label='TSLA',color='y')

lns = lns01 + lns02 + lns03 +  lns1 +  lns2 +  lns3 +  lns4 +  lns5 +  lns6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.spines['right'].set_color('yellow')
plt.grid()
plt.show()
