import tensorflow as tf


def show_model_structure(model):
    return tf.keras.utils.plot_model(model)
