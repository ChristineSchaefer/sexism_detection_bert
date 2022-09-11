# IMPORTS
import pandas as pd
import numpy as np
from tqdm import tqdm


# https://gist.github.com/RayWilliam46/a9a48998ace3cda38a47a4a87aa3d9b6#file-augment-py

def augment_tweet(tweet, aug, num_threads):
    """
        Construts a new sentence via text augmentation.

        Input:
            - sentence:     A string of text
            - aug:          An augmentation object defined by the nlpaug library
            - num_threads:  Integer controlling the number of threads to use if
                            augmenting text via CPU
        Output:
            - A string of text that been augmented
        """

    return aug.augment(tweet, num_thread=num_threads)


def augment_text(df, aug, num_threads, num_times):
    """
        Takes a pandas DataFrame and augments its text data.

        Input:
            - df:            A pandas DataFrame containing the columns:
                                    - 'text' containing strings of text to augment.
                                    - 'sexist' boolean target variable containing True and False.
            - aug:           Augmentation object defined by the nlpaug library.
            - num_threads:   Integer controlling number of threads to use if augmenting
                             text via CPU
            - num_times:     Integer representing the number of times to augment text.
        Output:
            - df:            Copy of the same pandas DataFrame with augmented data 
                             appended to it and with rows randomly shuffled.
        """

    # get rows of data to augment
    to_augment = df[df['sexist'] == True]
    to_augmentX = to_augment['text']
    to_augmentY = np.ones(len(to_augmentX.index) * num_times, dtype=bool)

    # build up dictionary containing augmented data
    aug_dict = {'text': [], 'sexist': to_augmentY}
    for i in tqdm(range(num_times)):
        augX = [augment_tweet(x, aug, num_threads) for x in to_augmentX]
        aug_dict['text'].extend(augX)

    # build DataFrame containing augmented data
    aug_df = pd.DataFrame.from_dict(aug_dict)

    return df.append(aug_df, ignore_index=True).sample(frac=1, random_state=42)
