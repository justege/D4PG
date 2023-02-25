import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# list of filenames

"""
filenames = ['/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_360_0.375.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_390_1.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_420_0.875.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_450_0.75.csv',
             '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/PortfolioValueAndEqualWeight_test_480_0.5.csv',
             ] 
             
                          '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/DailyReturnDistribution_test_450_0.75.csv',
             
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filenames = [
    '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/DailyReturnDistribution_test_390_1.csv',
    '/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/test/DailyReturnDistribution_test_98_98.csv',
    ]

# empty dataframe to stre data from all files
df = pd.DataFrame()

labels = [
             'T=1',
             'meanVariance',
             ]

# loop through the filenames list
for file in filenames:
    # read each file as a dataframe
    data = pd.read_csv(file)

    # add the data to the empty dataframe as new columns
    df = pd.concat([df, data.iloc[:,1]], axis=1)

# calculate the mean and standard deviation for each column
means = df.mean(axis=0)
stds = df.std(axis=0)

df = df[df.iloc[:,1] > -0.05]
df = df[df.iloc[:,1] < 0.05]


# plotting the histograms

color = ['darksalmon','brown']

for i, column in enumerate(df.columns):
    plt.hist(df[column], bins=50, alpha=0.5, label=labels[i], color=color[i])
    plt.axvline(x=means[i], color=color[i], linestyle='--', label='mean')

plt.legend()
plt.show()


