# IMPORTS
from transformers import TFDistilBertModel, DistilBertConfig


def set_model(dropout, att_dropout, model, trainable):
    """
            Define model.

            Input:
                - dropout:      float
                - att_dropout:  float
                - model:        string with model name
                - trainable:    boolean to (un-)freeze pretrained weights
            Output:
                - Pretrained model
            """
    # configure DistilBERT's initialization
    config = DistilBertConfig(dropout=dropout,
                              attention_dropout=att_dropout,
                              output_hidden_states=True)

    # the bare, pre-trained DistilBERT transformer model outputting raw hidden-states
    # and without any specific head on top.
    distilBERT = TFDistilBertModel.from_pretrained(model, config=config)

    # make DistilBERT layers untrainable
    # IMPORTANT: because of computational capacity it is not possible to unfreeze
    for layer in distilBERT.layers:
        layer.trainable = trainable

    return distilBERT
