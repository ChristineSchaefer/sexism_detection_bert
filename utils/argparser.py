# IMPORTS
import argparse


def parse():
    """
        Instantiate argument parser.

        Output:
            - parser object
    """
    parser = argparse.ArgumentParser(description='Automatic Sexism Detection with BERT')
    parser.add_argument('path', metavar='path', type=str,
                        help='path to your corpus with annotated tweets')
    parser.add_argument('fields', metavar='fields', nargs='+', type=str,
                        help="list with names of fields in corpora to use in datframe")
    parser.add_argument('augmentation', metavar='augmentation', type=str, help='if unbalanced dataset then '
                                                                                'augmentation = True')
    return parser
