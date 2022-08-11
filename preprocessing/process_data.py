import pandas as pd
from preprocessing import normalization


def get_data(path, fields):
    return pd.read_csv(path, skipinitialspace=True, usecols=fields)


def normalize(data):
    # TODO: drop irrelevant data from dataframe (e.g. id not numeric, sexism not boolean)
    # use methods from normalization.py
    for index, row in data.iterrows():
        data.loc[index, "text"] = normalization.normalize(row["text"])

    return data
