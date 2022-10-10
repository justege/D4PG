action = -0.5

import pandas as pd



def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)

    # data  = data[final_columns]
    data.index = data.datadate.factorize()[0]

    return data




preprocessed_path = "/Users/egemenokur/PycharmProjects/D4PG/0000_test.csv"
data = pd.read_csv(preprocessed_path, index_col=0)

data = data.drop(columns=["datadate_full"])

data = data[["datadate", "tic", "close", "open", "high", "low", "volume", "macd", "rsi", "cci", "adx"]]
# print(data.to_string())
train = data_split(data, start=20191009, end=20211009)
test_d = data_split(data, start=20211010, end=20221009)

data = data.reset_index()  # make sure indexes pair with number of rows



print(data[data.index % 5 == 0].to_string())
