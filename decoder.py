import tensorflow as tf
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Embedding, Dense# type: ignore
import keras
from src.attention_mechanism import attention


def decoder_without_attention(encoder_states, input_dim_decoder, output_dim_decoder, unit, decoder_model):
    decoder_inputs = Input(shape= [None])
    decoder_embedding = Embedding(input_dim= input_dim_decoder, output_dim=output_dim_decoder, mask_zero=True)
    decoder_embedding_output = decoder_embedding(decoder_inputs)
    decoder_model = decoder_model

    if type(decoder_model) == keras.src.layers.rnn.lstm.LSTM:
        decoder_output,_,_ = decoder_model(decoder_embedding_output,initial_state=encoder_states)

    if type(decoder_model) == keras.src.layers.rnn.gru.GRU:
        decoder_output,_ = decoder_model(decoder_embedding_output,initial_state=encoder_states)

    decoder_dense = Dense(units= unit, activation='softmax')
    decoder_outputs = decoder_dense(decoder_output)
    return decoder_inputs, decoder_outputs



def decoder_with_attention(encoder_states, input_dim_decoder, output_dim_decoder, unit, decoder_model, encoder_outputs, attention_layer):
    decoder_inputs = Input(shape=[None])
    decoder_embedding = Embedding(input_dim=input_dim_decoder, output_dim=output_dim_decoder, mask_zero=False)
    decoder_embedding_output = decoder_embedding(decoder_inputs)

    if type(decoder_model) == keras.src.layers.rnn.lstm.LSTM:
        decoder_output,_,_ = decoder_model(decoder_embedding_output,initial_state=encoder_states)

    if type(decoder_model) == keras.src.layers.rnn.gru.GRU:
        decoder_output,_ = decoder_model(decoder_embedding_output,initial_state=encoder_states)

    concat_layer = attention(attention_layer,decoder_output, encoder_outputs)

    decoder_dense = Dense(units=unit, activation='softmax')
    decoder_outputs = decoder_dense(concat_layer)
    return decoder_inputs, decoder_outputs 