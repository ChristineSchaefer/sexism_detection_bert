from preprocessing import process_data
from utils import argparser
import nlpaug.augmenter.word as nlpaw
from data_augmentation import construction


def main(args):
    # Load data from path and store it to a dataframe
    corpus = process_data.get_data(args.path, args.fields)

    # count sexism true and false tweets for under-/oversampling check
    sexist = corpus.query("sexist == True").shape[0]
    no_sexist = corpus.query("sexist == False").shape[0]
    print(f"Perc. of sexist examples: {(sexist / (no_sexist + sexist)) * 100}")
    print(f"Perc. of no_sexist examples: {(no_sexist / (no_sexist + sexist)) * 100}")

    # normalize tweets
    normalized_corpus = process_data.normalize(corpus)

    # Split data into test and training set (unbalanced dataset: undersampling)
    training_data = normalized_corpus.sample(frac=0.8, random_state=25)
    testing_data = normalized_corpus.drop(training_data.index)

    print(f"No. of training examples: {training_data.shape[0]}")
    print(f"No. of testing examples: {testing_data.shape[0]}")

    # Use training data to create balanced dataset via data augmentation
    # Define nlpaug augmentation object
    aug10p = nlpaw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_min=1, aug_p=0.1,
                                         action="substitute")

    # Upsample minority class ('isToxic' == 1) to create a roughly 50-50 class distribution
    balanced_df = construction.augment_text(training_data, aug10p, num_threads=8, num_times=3)
    print(balanced_df)


if __name__ == "__main__":
    parser = argparser.parse()
    args = parser.parse_args()
    main(args)
