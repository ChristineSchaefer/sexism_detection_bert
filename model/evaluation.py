# IMPORTS
from keras import backend as K


# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    """
        Compute recall for prediction.

           Input:
               - y_true:    numpy array(s) of true values
               - y_pred:    numpy array(s) of predictions

           Output:
               - float
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
        Compute precision for prediction.

           Input:
               - y_true:    numpy array(s) of true values
               - y_pred:    numpy array(s) of predictions

           Output:
               - float
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
        Compute f-score for prediction.

           Input:
               - y_true:    numpy array(s) of true values
               - y_pred:    numpy array(s) of predictions

           Output:
               - float
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
