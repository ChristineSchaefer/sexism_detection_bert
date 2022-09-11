# IMPORTS
import tensorflow as tf
from keras import backend as K
from model.evaluation import f1_m, precision_m, recall_m


# https://gist.github.com/RayWilliam46/07e8718b2e7b102b9617e06c9faca27c#file-build_model-py

def build_model(transformer, max_length, seed, learning_rate, compile):
    """
    Template for building a model off of the BERT or DistilBERT architecture
    for a binary classification task.

    Input:
        - transformer:      a base Hugging Face transformer model object (BERT or DistilBERT)
                            with no added classification head attached.
        - max_length:       integer controlling the maximum number of encoded tokens in a given sequence.
        - seed:             integer with random state
        - learning_rate:    float
        - compile:          boolean

    Output:
        - model:            a compiled tf.keras.Model with added classification layers
                            on top of the base pre-trained model architecture.
    """

    # define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed)

    # define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids',
                                            dtype='int32', )
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention',
                                                  dtype='int32')

    # distilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # it is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    # we only care about DistilBERT's output for the [CLS] token,
    # which is located at index 0 of every encoded sequence.
    # splicing out the [CLS] tokens gives us 2D data.
    cls_token = last_hidden_state[:, 0, :]

    # define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(cls_token)

    # define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    if compile is True:
        # compile the model
        model.compile(tf.keras.optimizers.Adam(learning_rate),
                      loss="binary_crossentropy",
                      metrics=['accuracy', f1_m, precision_m, recall_m])

    return model


# https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
def focal_loss(gamma, alpha):
    """
        Focal loss function to compute loss during model training. Use focal loss instead of binary cross entropy
        because of unbalanced dataset.

        Input:
            - gamma:    float
            - alpha:    float

        Output:
            - focal_loss_fixed: loss value
        """

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, True), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, False), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed
