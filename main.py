from preprocessing import process_data, tokenizer
from utils import argparser
import nlpaug.augmenter.word as nlpaw
from data_augmentation import construction
from model import creation, config


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

    # Split data into test, validate and training set (unbalanced dataset: undersampling)
    # training
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
    distilBert = config.set_model(DISTILBERT_DROPOUT, DISTILBERT_ATT_DROPOUT, 'distilbert-base-uncased')

    # Add Classification Head
    MAX_LENGTH = 512
    LEARNING_RATE = 5e-5
    RANDOM_STATE = 42
    model = creation.build_model(distilBert, MAX_LENGTH, RANDOM_STATE, LEARNING_RATE)

    # train model
    EPOCHS = 6
    BATCH_SIZE = 64
    NUM_STEPS = len(x_train.index) // BATCH_SIZE

    train_history1 = model.fit(
        x=[x_train_ids, x_train_attention],
        y=y_train.to_numpy(),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=NUM_STEPS,
        validation_data=([x_valid_ids, x_valid_attention], y_valid.to_numpy()),
        verbose=2
    )


if __name__ == "__main__":
    parser = argparser.parse()
    args = parser.parse_args()
    main(args)
