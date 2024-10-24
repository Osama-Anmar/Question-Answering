
import tensorflow as tf
from tensorflow.keras.layers import Concatenate# type: ignore
import keras


def attention(attention_layer, decoder_output, encoder_outputs):
        attention_layer = attention_layer
        if type(attention_layer) == keras.src.layers.attention.attention.Attention:
            context_vector = attention_layer([decoder_output, encoder_outputs])
            concat_layer = Concatenate(axis=-1)([decoder_output, context_vector])

        elif type(attention_layer) == keras.src.layers.attention.additive_attention.AdditiveAttention:
            context_vector = attention_layer([decoder_output, encoder_outputs])
            concat_layer = Concatenate(axis=-1)([decoder_output, context_vector])

        elif type(attention_layer) == keras.src.layers.attention.multi_head_attention.MultiHeadAttention:
            attention_output, _ = attention_layer(query=decoder_output, value=encoder_outputs, key=encoder_outputs, return_attention_scores=True)
            concat_layer = Concatenate(axis=-1)([decoder_output, attention_output])
        return concat_layer