from transformers import TFDistilBertModel, DistilBertConfig


def set_model(dropout, att_dropout, model, trainable):
    # Configure DistilBERT's initialization
    config = DistilBertConfig(dropout=dropout,
                              attention_dropout=att_dropout,
                              output_hidden_states=True)

    # The bare, pre-trained DistilBERT transformer model outputting raw hidden-states
    # and without any specific head on top.
    distilBERT = TFDistilBertModel.from_pretrained(model, config=config)

    # Make DistilBERT layers untrainable
    for layer in distilBERT.layers:
        layer.trainable = trainable

    return distilBERT
