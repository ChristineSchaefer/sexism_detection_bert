# IMPORTS
from transformers import DistilBertTokenizerFast
import tensorflow as tf


def get_distilbert_tokenizer(model):
    """
        Return DistilBERT tokenizer object.

        Input:
            - model: string
    """
    # instantiate DistilBERT tokenizer...we use the Fast version to optimize runtime
    return DistilBertTokenizerFast.from_pretrained(model)


# https://gist.github.com/RayWilliam46/c2cdc2e41bef33b332151d7acc2afef2#file-batch_encode-py
def encode(tokenizer, texts):
    """
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.

        Input:
            - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
            - texts:       List of strings where each string represents a text
        Output:
            - input_ids:       sequence of texts encoded as a tf.Tensor object
            - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """

    input_ids = []
    attention_mask = []

    inputs = tokenizer(texts, padding='max_length', truncation=True, return_attention_mask=True, return_token_type_ids=True)
    input_ids.extend(inputs['input_ids'])
    attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)
