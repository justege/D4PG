
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib.cm import get_cmap

metricList = ['sharpe', 'value_portfolio', 'trades', 'total_cost', 'variance', 'totalReturn', 'variance', 'totalReturn', 'drawdown','maxDD', 'meanReturn']

taus = [1.1, 1,0.75,0.5]
testOrTrain = ['train', 'test'] # 'Test
tauValue = 1
preprocessed_path = "0000_test.csv"

# pd.DataFrame({'tauValue':self.tauValue,'model_number': self.modelNumber,'sharpe':[sharpe],'value_portfolio':[self.P_t_0],'trades':[self.trades],'total_cost':[self.total_cost], 'variance': [df_total_value['daily_return'].std()], 'totalReturn': [total_return], 'drawdown': [drawdown], 'maxDD': [maxDD], 'mean_return':[df_total_value['daily_return'].mean()]}).to_csv("runs/resultsPortfolioValue" + self.testOrTrain + "_" + str(self.tauValue) + ".csv",index=False, mode='a', header=False)

#figTrainTest, axTrainTest = plt.subplots()

fig, ax = plt.subplots()
lnAll = ax.plot([1] * len(range(60, 541, 60)), label="1")


for testOrTrainValue in testOrTrain:

    metric = metricList[1]

    for i in range(60, 541, 60):
        for tauVal in taus:
            preprocessed_path = 'runs/account_value_train_' + testOrTrainValue  + str(i) + "_" + str(tauVal) +'.csv'
            data = pd.read_csv(preprocessed_path, index_col=0)

            ln = ax.plot(data[str(metric)], label= str(i) + "_" + str(tauVal))
            lnAll = lnAll + ln

    plt.title(str(tauValue) + '_' + metric + '_' + testOrTrainValue, loc='left')
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    labs = [l.get_label() for l in lnAll]
    ax.legend(lnAll, labs, loc=0)
    plt.grid()
    plt.show()

    plt.savefig('runs/Comparison' + metric + '_' + testOrTrainValue + '.png')
    plt.close()




