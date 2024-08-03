import tensorflow as tf
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Embedding, Dense, Concatenate, TimeDistributed# type: ignore
from keras import Model  # type: ignore
import keras

class EncodeDecoder:
    def __init__(self, input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder, input_length_encoder, input_length_decoder, model_encoder, model_decoder, unit):
        self.input_dim_encoder = input_dim_encoder
        self.input_dim_decoder = input_dim_decoder
        self.output_dim_encoder = output_dim_encoder
        self.output_dim_decoder = output_dim_decoder
        self.input_length_decoder = input_length_decoder
        self.input_length_encoder = input_length_encoder
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.unit = unit
        

class EncoderDecoderWithoutAttention(EncodeDecoder):
    def __init__(self, input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder, input_length_encoder, input_length_decoder, model_encoder, model_decoder, unit):
        super().__init__(input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder, input_length_encoder, input_length_decoder, model_encoder, model_decoder, unit)
    
    def encoder(self):
        encoder_inputs = Input(shape = [None])
        encoder_embeddings = Embedding(input_dim = self.input_dim_encoder, output_dim =self.output_dim_encoder, input_length= self.input_length_encoder, mask_zero=True)
        encoder_embedding_output = encoder_embeddings(encoder_inputs)
        encoder_model = self.model_encoder
        if type(encoder_model) == keras.src.layers.rnn.lstm.LSTM:
            encoder_outputs, state_h, state_c = encoder_model(encoder_embedding_output)
            encoder_states = (state_h, state_c)

        elif type(encoder_model) == keras.src.layers.rnn.gru.GRU:
            encoder_outputs, encoder_states = encoder_model(encoder_embedding_output)
        return encoder_inputs, encoder_states
    
    def decoder(self, encoder_states):
        decoder_inputs = Input(shape=[None])
        decoder_embedding = Embedding(input_dim = self.input_dim_decoder, output_dim =self.output_dim_decoder, input_length= self.input_length_decoder, mask_zero=True)
        decoder_embedding_output = decoder_embedding(decoder_inputs)
        decoder_model = self.model_decoder
        if type(decoder_model) == keras.src.layers.rnn.lstm.LSTM:
            decoder_outputs,_,_ = decoder_model(decoder_embedding_output, initial_state=encoder_states)

        elif type(decoder_model) == keras.src.layers.rnn.gru.GRU:
            decoder_outputs,_ = decoder_model(decoder_embedding_output, initial_state=encoder_states)

        decoder_dense =  TimeDistributed(Dense(units = self.unit, activation='softmax'))
        output_ = decoder_dense(decoder_outputs)
        return decoder_inputs, output_
    
    def build_model(self):
        encoder_inputs, encoder_states = self.encoder()
        decoder_input, decoder_output = self.decoder(encoder_states)
        model = Model([encoder_inputs, decoder_input], decoder_output)
        return model
    
    def model_compile(self, model, optimizer, loss, metrics):
        model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
    
    def model_fit(self, model, epochs, early_stop,batch_size, encoder_input, decoder_input, decoder_output):       
        model.fit([encoder_input, decoder_input],
                                 decoder_output,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[early_stop])
        

class EncoderDecoderWithAttention(EncodeDecoder):
    def __init__(self, input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder, input_length_encoder, input_length_decoder, model_encoder, model_decoder, unit, attention_layer):
        super().__init__(input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder, input_length_encoder, input_length_decoder, model_encoder, model_decoder, unit)
        self.attention_layer = attention_layer

    
    def encoder_(self):
            encoder_inputs = Input(shape = [None])
            encoder_embeddings = Embedding(input_dim = self.input_dim_encoder, output_dim =self.output_dim_encoder, input_length= self.input_length_encoder, mask_zero=True)
            encoder_embedding_output = encoder_embeddings(encoder_inputs)
            encoder_model = self.model_encoder
            if type(encoder_model) == keras.src.layers.rnn.lstm.LSTM:
                encoder_outputs, state_h, state_c = encoder_model(encoder_embedding_output)
                encoder_states = (state_h, state_c)

            elif type(encoder_model) == keras.src.layers.rnn.gru.GRU:
                encoder_outputs, encoder_states = encoder_model(encoder_embedding_output)
            return encoder_outputs, encoder_inputs, encoder_states

    def decoder_(self, encoder_states, encoder_outputs):
            decoder_inputs = Input(shape=[None])
            decoder_embedding = Embedding(input_dim=self.input_dim_decoder, output_dim=self.output_dim_decoder, input_length=self.input_length_decoder, mask_zero=True)
            decoder_embedding_output = decoder_embedding(decoder_inputs)
            decoder_model = self.model_decoder
            if type(decoder_model) == keras.src.layers.rnn.lstm.LSTM:
                decoder_outputs,_,_ = decoder_model(decoder_embedding_output, initial_state=encoder_states)
            elif type(decoder_model) == keras.src.layers.rnn.gru.GRU:
                decoder_outputs,_ = decoder_model(decoder_embedding_output, initial_state=encoder_states)       
            concatenate_layer = self.attention(decoder_outputs, encoder_outputs)
            decoder_dense = TimeDistributed(Dense(units=self.unit, activation='softmax'))
            output_ = decoder_dense(concatenate_layer)
            return decoder_inputs, output_  

    def attention(self, decoder_output, encoder_outputs):
        attention_layer = self.attention_layer
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

    def build_model(self):
        encoder_outputs, encoder_inputs, encoder_states = self.encoder_()
        decoder_input, decoder_output = self.decoder_(encoder_states, encoder_outputs)
        model = Model([encoder_inputs, decoder_input], decoder_output)
        return model
    
    def model_compile(self, model, optimizer, loss, metrics):
        model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
    
    def model_fit(self, model, epochs, early_stop,batch_size, encoder_input, decoder_input, decoder_output):       
        model.fit([encoder_input, decoder_input],
                                 decoder_output,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[early_stop])