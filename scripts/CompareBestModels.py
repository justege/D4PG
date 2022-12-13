import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib.cm import get_cmap

metricList = ['sharpe', 'value_portfolio', 'trades', 'total_cost', 'variance', 'totalReturn', 'variance', 'totalReturn', 'drawdown','maxDD', 'meanReturn']

testOrTrainValue ='test' # 'Test
preprocessed_path = "0000_test.csv"

fig, ax = plt.subplots()
BestModel_1 = 300
BestModel_0_75 = 300
BestModel_0_5 = 300

for metric in metricList:

    preprocessed_path_1 = 'runs/account_value_train_' + testOrTrainValue + "_" + str(BestModel_1)+ "_" + str(1)  +'.csv'
    preprocessed_path_0_75 = 'runs/account_value_train_' + testOrTrainValue  + "_" + str(BestModel_0_75)+ "_" + str(0.75)  +'.csv'
    preprocessed_path_0_5 = 'runs/account_value_train_' + testOrTrainValue   + "_" + str(BestModel_0_5)+ "_" + str(0.5)  +'.csv'


    data_1 = pd.read_csv(preprocessed_path_1, index_col=0)
    data_0_75 = pd.read_csv(preprocessed_path_0_75, index_col=0)
    data_0_5 = pd.read_csv(preprocessed_path_0_5, index_col=0)

    ln1 = ax.plot(data_1[str(metric)], label= str(BestModel_1) + "_" + str(1))
    ln_0_75 = ax.plot(data_0_75[str(metric)], label=str(BestModel_0_75) + "_" + str(0.75))
    ln_0_50 = ax.plot(data_0_5[str(metric)], label=str(BestModel_0_5) + "_" + str(0.5))

    lnAll = ln1 + ln_0_75 + ln_0_50

    plt.title(str(metric) + '_' + 'Comparison', loc='left')
    plt.xlabel("Epoch")
    plt.ylabel(metric)

    labs = [l.get_label() for l in lnAll]
    ax.legend(lnAll, labs, loc=0)
    plt.grid()
    plt.show()

    plt.savefig('runs/Comparison' + metric + '.png')
    plt.close()




