# IMPORTS
from matplotlib import pyplot as plt

from preprocessing import process_data, tokenizer, normalization
from utils import argparser
from model import creation, config
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd


def main(arguments):
    """
    Main-Method to start application.
    Arguments:
        1. path to dataset
        2. column tweet
        3. column label
        4. data augmentation (True/False)
    """

    print("----- Start -----")
    # load data from path and store it to a dataframe
    corpus = process_data.get_data(arguments.path, arguments.fields)
    # get argument for data augmentation
    augmentation = arguments.augmentation

    # count sexism true and false tweets for under-/oversampling check
    sexist = corpus.query("sexist == True").shape[0]
    no_sexist = corpus.query("sexist == False").shape[0]
    print(f"Number of sexist examples: {sexist} \n Perc.: {(sexist / (no_sexist + sexist)) * 100}")
    print(f"Number. of no_sexist examples: {no_sexist} \n Perc.: {(no_sexist / (no_sexist + sexist)) * 100}")

    print("----- Normalization -----")
    # normalize tweets
    normalized_corpus = process_data.normalize(corpus)

    # check if data augmentation
    if augmentation == "True":
        print("----- Data augmentation -----")
        normalized_corpus = process_data.data_augmentation(normalized_corpus, 'distilbert-base-uncased')
        # count sexism true and false tweets after data augmentation
        sexist = normalized_corpus.query("sexist == True").shape[0]
        no_sexist = normalized_corpus.query("sexist == False").shape[0]
        print(f"Number of sexist examples: {sexist} \n Perc.: {(sexist / (no_sexist + sexist)) * 100}")
        print(f"Number. of no_sexist examples: {no_sexist} \n Perc.: {(no_sexist / (no_sexist + sexist)) * 100}")

    print("----- Split data into training, validation and test set -----")
    # split data into test, validate and training_model set
    # train
    training_data = normalized_corpus.sample(frac=0.7, random_state=25)
    x_train = training_data.iloc[:, 0]
    y_train = training_data.iloc[:, 1]
    print(f"No. of training examples: {training_data.shape[0]}")

    # validation
    validate_data = normalized_corpus.drop(training_data.index).sample(frac=0.7, random_state=25)
    x_valid = validate_data.iloc[:, 0]
    y_valid = validate_data.iloc[:, 1]
    print(f"No. of validation examples: {validate_data.shape[0]}")

    # test
    testing_data = normalized_corpus.drop(training_data.index).drop(validate_data.index)
    x_test = testing_data.iloc[:, 0]
    y_test = testing_data.iloc[:, 1]
    print(f"No. of testing examples: {testing_data.shape[0]}")

    print("----- Use distilbert tokenizer to encode data -----")
    # set distilbert tokenizer
    tknz = tokenizer.get_distilbert_tokenizer('distilbert-base-uncased')

    # encode x_train
    x_train_ids, x_train_attention = tokenizer.encode(tknz, normalization.normalize_list(x_train.tolist()))

    # encode x_valid
    x_valid_ids, x_valid_attention = tokenizer.encode(tknz, normalization.normalize_list(x_valid.tolist()))

    # encode x_test
    x_test_ids, x_test_attention = tokenizer.encode(tknz, normalization.normalize_list(x_test.tolist()))

    print("----- Initialize base model -----")
    # initialize the Base Model
    DISTILBERT_DROPOUT = 0.2
    DISTILBERT_ATT_DROPOUT = 0.2
    distilBert = config.set_model(DISTILBERT_DROPOUT, DISTILBERT_ATT_DROPOUT, 'distilbert-base-uncased', False)

    print("----- Add classification head -----")
    # add Classification Head
    MAX_LENGTH = 512
    LEARNING_RATE = 5e-5
    RANDOM_STATE = 42
    model = creation.build_model(distilBert, MAX_LENGTH, RANDOM_STATE, LEARNING_RATE, True)
    model.summary()

    print("----- Save model -----")
    # save model
    tf.saved_model.save(model, 'models/unbalanced_model')

    # train model
    print("----- Start training -----")
    # start training
    train_history = model.fit(
        x=[x_train_ids, x_train_attention],
        y=y_train.to_numpy(),
        epochs=10,
        batch_size=64,
        steps_per_epoch=(len(x_train.index) // 64),
        validation_data=([x_valid_ids, x_valid_attention], y_valid.to_numpy()),
        verbose=2)

    history_df = pd.DataFrame(train_history.history)
    # plot training and validation loss over each epoch
    history_df.loc[:, ['loss', 'val_loss']].plot()
    plt.title(label='Training + Validation Loss Over Time', fontsize=17, pad=19)
    plt.xlabel('Epoch', labelpad=14, fontsize=14)
    plt.ylabel('Focal Loss', labelpad=16, fontsize=14)
    print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

    # save figure
    plt.savefig('unbalanced_trainvalloss.png', dpi=300.0, transparent=True)

    print("----- Finish training. Start evaluation -----")
    # evaluation
    # generate predictions
    y_pred = model.predict([x_test_ids, x_test_attention])
    y_pred_thresh = np.where(y_pred >= 0.5, 1, 0)
    y_pred_bool = np.argmax(y_pred, axis=1)

    # get evaluation results
    accuracy = accuracy_score(y_test, y_pred_thresh)
    auc_roc = roc_auc_score(y_test, y_pred)

    print('Accuracy:  ', accuracy)
    print('ROC-AUC:   ', auc_roc)
    print(classification_report(y_test, y_pred_bool))


if __name__ == "__main__":
    parser = argparser.parse()
    args = parser.parse_args()
    main(args)
