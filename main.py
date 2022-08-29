from matplotlib import pyplot as plt

from preprocessing import process_data, tokenizer
from utils import argparser
import nlpaug.augmenter.word as nlpaw
from data_augmentation import construction
from model import creation, config
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd


def main(arguments):
    # Load data from path and store it to a dataframe
    corpus = process_data.get_data(arguments.path, arguments.fields)

    # count sexism true and false tweets for under-/oversampling check
    sexist = corpus.query("sexist == True").shape[0]
    no_sexist = corpus.query("sexist == False").shape[0]
    print(f"Perc. of sexist examples: {(sexist / (no_sexist + sexist)) * 100}")
    print(f"Perc. of no_sexist examples: {(no_sexist / (no_sexist + sexist)) * 100}")

    # normalize tweets
    normalized_corpus = process_data.normalize(corpus)

    # Split data into test, validate and training_model set (unbalanced dataset: undersampling)
    # training_model
    training_data = normalized_corpus.sample(frac=0.7, random_state=25)
    x_train = training_data.iloc[:, 0]
    y_train = training_data.iloc[:, 1]

    # validation
    validate_data = normalized_corpus.drop(training_data.index).sample(frac=0.7, random_state=25)
    x_valid = validate_data.iloc[:, 0]
    y_valid = validate_data.iloc[:, 1]

    # test
    testing_data = normalized_corpus.drop(training_data.index).drop(validate_data.index)
    x_test = testing_data.iloc[:, 0]
    y_test = testing_data.iloc[:, 1]

    print(f"No. of training examples: {training_data.shape[0]}")
    print(f"No. of validation examples: {validate_data.shape[0]}")
    print(f"No. of testing examples: {testing_data.shape[0]}")

    # Use training data to create balanced dataset via data augmentation
    # Define nlpaug augmentation object
    # aug10p = nlpaw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_min=1, aug_p=0.1,
    # action="substitute")

    # Upsample minority class ('sexist' == True) to create a roughly 50-50 class distribution
    # balanced_df = construction.augment_text(training_data, aug10p, num_threads=8, num_times=3)
    # print(f"No. of balanced training examples: {balanced_df.shape[0]}")

    # set distilbert tokenizer
    tknz = tokenizer.get_distilbert_tokenizer('distilbert-base-uncased')

    # Encode x_train
    x_train_ids, x_train_attention = tokenizer.encode(tknz, x_train.tolist())

    # Encode x_valid
    x_valid_ids, x_valid_attention = tokenizer.encode(tknz, x_valid.tolist())

    # Encode x_test
    x_test_ids, x_test_attention = tokenizer.encode(tknz, x_test.tolist())

    # Initialize the Base Model
    DISTILBERT_DROPOUT = 0.2
    DISTILBERT_ATT_DROPOUT = 0.2
    distilBert = config.set_model(DISTILBERT_DROPOUT, DISTILBERT_ATT_DROPOUT, 'distilbert-base-uncased', False)

    # Add Classification Head
    MAX_LENGTH = 512
    LEARNING_RATE = 5e-5
    RANDOM_STATE = 42
    model = creation.build_model(distilBert, MAX_LENGTH, RANDOM_STATE, LEARNING_RATE, True)
    model.summary()
    # Save model
    tf.saved_model.save(model, 'models/unbalanced_model')

    # train model
    # Define callbacks
    """early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      min_delta=0,
                                                      patience=0,
                                                      restore_best_weights=True)"""
    # start training
    train_history = model.fit(
        x=[x_train_ids, x_train_attention],
        y=y_train.to_numpy(),
        epochs=6,
        batch_size=64,
        steps_per_epoch=(len(x_train.index) // 64),
        validation_data=([x_valid_ids, x_valid_attention], y_valid.to_numpy()),
        #callbacks=[early_stopping],
        verbose=2)

    history_df = pd.DataFrame(train_history.history)
    # Plot training and validation loss over each epoch
    history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.title(label='Training + Validation Loss Over Time', fontsize=17, pad=19)
    plt.xlabel('Epoch', labelpad=14, fontsize=14)
    plt.ylabel('Focal Loss', labelpad=16, fontsize=14)
    print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

    # Save figure
    # plt.savefig('figures/unbalanced_trainvalloss.png', dpi=300.0, transparent=True)

    # Evaluation
    # Generate predictions
    y_pred = model.predict([x_test_ids, x_test_attention])
    y_pred_thresh = np.where(y_pred >= 0.5, 1, 0)

    # Get evaluation results
    accuracy = accuracy_score(y_test, y_pred_thresh)
    auc_roc = roc_auc_score(y_test, y_pred)

    print('Accuracy:  ', accuracy)

    print('ROC-AUC:   ', auc_roc)


if __name__ == "__main__":
    parser = argparser.parse()
    args = parser.parse_args()
    main(args)
