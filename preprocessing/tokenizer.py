from transformers import DistilBertTokenizerFast
import tensorflow as tf


def get_distilbert_tokenizer(model):
    # Instantiate DistilBERT tokenizer...we use the Fast version to optimize runtime
    return DistilBertTokenizerFast.from_pretrained(model)


# Define function to encode text data in batches
def encode(tokenizer, texts):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""

    input_ids = []
    attention_mask = []

    inputs = tokenizer(texts, padding='max_length', truncation=True, return_attention_mask=True, return_token_type_ids=True)
    input_ids.extend(inputs['input_ids'])
    attention_mask.extend(inputs['attention_mask'])
    # NOTE: convertion doesn't work, because lists in input_ids and attention_mask haven't the same length -> not rectangular

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)
