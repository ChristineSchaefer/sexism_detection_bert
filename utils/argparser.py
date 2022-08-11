import argparse


def parse():
    parser = argparse.ArgumentParser(description='Automatic Sexism Detection with BERT')
    parser.add_argument('path', metavar='path', type=str,
                        help='path to your corpus with annotated tweets')
    parser.add_argument('fields', metavar='fiels', nargs='+', type=str,
                        help="list with names of fields in corpora to use in datframe")
    return parser
