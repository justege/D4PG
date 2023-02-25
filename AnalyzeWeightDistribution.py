import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = ['/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PriceAndWeightDistribution_test_360_0.375.csv',
'/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PriceAndWeightDistribution_test_480_0.5.csv',
'/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PriceAndWeightDistribution_test_450_0.75.csv',
'/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PriceAndWeightDistribution_test_390_1.csv']


df = pd.DataFrame()
data = pd.read_csv(files[0])
data = np.array(data)

# extract the weights and time
weights = np.array(data)[:,1:5]
prices = np.array(data)[:,5:9]

fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(5,1, figsize=(10, 11))

sum_of_weights_0 = [weights[:,0].sum()/250, weights[:,1].sum()/250, weights[:,2].sum()/250, weights[:,3].sum()/250]



ax1.stackplot(np.arange(weights.shape[0]), weights[:,0], weights[:,1], weights[:,2], weights[:,3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)

ax1.legend(loc='upper left')
ax1.set_xlim(0, weights.shape[0]-1)
ax1.set_ylim(0, 1)
ax1.set_ylabel('Weight')



data = pd.read_csv(files[1])
# extract the weights and time
weights = np.array(data)[:,1:5]
prices = np.array(data)[:,5:9]

sum_of_weights_1 = [weights[:,0].sum()/250, weights[:,1].sum()/250, weights[:,2].sum()/250, weights[:,3].sum()/250]

ax2.stackplot(np.arange(weights.shape[0]), weights[:,0], weights[:,1], weights[:,2], weights[:,3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)


data = pd.read_csv(files[2])
# extract the weights and time
weights = np.array(data)[:,1:5]
prices = np.array(data)[:,5:9]

sum_of_weights_2 = [weights[:,0].sum()/250, weights[:,1].sum()/250, weights[:,2].sum()/250, weights[:,3].sum()/250]


ax3.stackplot(np.arange(weights.shape[0]), weights[:,0], weights[:,1], weights[:,2], weights[:,3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)

data = pd.read_csv(files[3])
# extract the weights and time
weights = np.array(data)[:,1:5]
prices = np.array(data)[:,5:9]

sum_of_weights_3 = [weights[:,0].sum()/250, weights[:,1].sum()/250, weights[:,2].sum()/250, weights[:,3].sum()/250]


ax4.stackplot(np.arange(weights.shape[0]), weights[:,0], weights[:,1], weights[:,2], weights[:,3],
              labels=['AAPL', 'CL=F', 'TSLA', 'Cash'],
              edgecolor='black',
              alpha=0.75)



ax5.plot(np.arange(prices.shape[0]), prices[:,1], 'b-', label='AAPL')
ax5.plot(np.arange(prices.shape[0]), prices[:,2], 'y-', label='CL=F')
ax5.plot(np.arange(prices.shape[0]), prices[:,3], 'g-', label='TSLA')


ax1.set_title('0.375')
ax2.set_title('0.5')
ax3.set_title('0.75')
ax4.set_title('1')
ax5.set_title('Prices')
ax1.set_xlim(0, len(prices[:,1]))
ax2.set_xlim(0, len(weights[:,1]))
ax3.set_xlim(0, len(weights[:,1]))
ax4.set_xlim(0, len(weights[:,1]))
ax5.set_xlim(0, len(weights[:,1]))
# show the plot
plt.legend()
plt.show()


labels_Col = ['0.375', '0.5', '0.75', '1']
labels_Title = ['AAPL', 'CL=F', 'TSLA', 'Cash']

values = [sum_of_weights_0, sum_of_weights_1, sum_of_weights_2, sum_of_weights_3]

# create a bar chart for each column
n_columns = len(values[0])
colors = plt.cm.rainbow(np.linspace(0, 1, n_columns))  # generate n_columns colors from the rainbow color map


for col in range(len(values[0])):
    col_values = [row[col] for row in values]
    plt.bar(labels_Col, col_values, color=['red', 'green', 'blue', 'yellow'])
    plt.title(f"Invested in {labels_Title[col]}")
    plt.xlabel("Agents")
    plt.ylabel("Average Weights")
    plt.show()

