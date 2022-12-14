# IMPORTS
import pandas as pd
from preprocessing import normalization
import nlpaug.augmenter.word as nlpaw
from data_augmentation import construction


def get_data(path, fields):
    """
        Read file from path.

        Input:
            - path: string
            - fields: list with important columns

        Output:
            - file as dataframe
    """
    return pd.read_csv(path, skipinitialspace=True, usecols=fields)


def normalize(data):
    """
        Normalize data.

           Input:
               - data: dataframe

           Output:
               - normalized dataframe
    """
    # use methods from normalization.py
    for index, row in data.iterrows():
        data.loc[index, "text"] = normalization.normalize(row["text"])

    return data


def data_augmentation(data, model_path):
    """
        Creation of synthetic data.

        Input:
            - data: dataframe
            - model_path: string

        Output:
            - balanced dataframe
    """
    # Use training data to create balanced dataset via data augmentation
    # Define nlpaug augmentation object
    aug10p = nlpaw.ContextualWordEmbsAug(model_path=model_path, aug_min=1, aug_p=0.1,
                                         action="substitute")

    # Upsample minority class ('sexist' == True) to create a roughly 50-50 class distribution
    balanced_df = construction.augment_text(data, aug10p, num_threads=8, num_times=3)

    return balanced_df
