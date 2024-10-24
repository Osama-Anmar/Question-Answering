import tensorflow as tf
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Embedding, Concatenate# type: ignore
from keras import Model  # type: ignore
import keras

def encoder_without_attention(input_dim_encoder, output_dim_encoder, encoder_model):
    context_inputs = Input(shape=[None], name = 'context')
    question_inputs = Input(shape=[None], name = 'question')

    embeddings = Embedding(input_dim=input_dim_encoder, output_dim=output_dim_encoder, mask_zero=False)    
    context_emeddeing = embeddings(context_inputs)
    question_emeddeing = embeddings(question_inputs)

    encoder_embedding_output = Concatenate(axis=-1)([context_emeddeing, question_emeddeing])

    encoder_model = encoder_model

    if type(encoder_model) == keras.src.layers.rnn.lstm.LSTM:
        _, state_h, state_c = encoder_model(encoder_embedding_output)
        encoder_states = (state_h, state_c)


    if type(encoder_model) == keras.src.layers.rnn.gru.GRU:
        _, encoder_states= encoder_model(encoder_embedding_output)


    if type(encoder_model) == keras.src.layers.rnn.bidirectional.Bidirectional:
        _, forward_h, forward_c, backward_h, backward_c = encoder_model(encoder_embedding_output)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c] 

    return context_inputs, question_inputs, encoder_states



def encoder_with_attention(input_dim_encoder, output_dim_encoder, encoder_model):
    context_inputs = Input(shape=[None], name = 'context')
    question_inputs = Input(shape=[None], name = 'question')

    embeddings = Embedding(input_dim=input_dim_encoder, output_dim=output_dim_encoder, mask_zero=False)    
    context_emeddeing = embeddings(context_inputs)
    question_emeddeing = embeddings(question_inputs)

    encoder_embedding_output = Concatenate(axis=-1)([context_emeddeing, question_emeddeing])

    encoder_model = encoder_model


    if type(encoder_model) == keras.src.layers.rnn.lstm.LSTM:
        encoder_outputs, state_h, state_c = encoder_model(encoder_embedding_output)
        encoder_states = (state_h, state_c)

    if type(encoder_model) == keras.src.layers.rnn.gru.GRU:
        encoder_outputs, encoder_states= encoder_model(encoder_embedding_output)

    if type(encoder_model) == keras.src.layers.rnn.bidirectional.Bidirectional:
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_model(encoder_embedding_output)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c] 

    return context_inputs, question_inputs, encoder_states, encoder_outputs