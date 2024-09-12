import tensorflow as tf
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Embedding, Dense, Concatenate, LSTM# type: ignore
from keras import Model  # type: ignore
import keras

class EncodeDecoder:
    def __init__(self, input_dim_encoder , input_dim_decoder  , output_dim_encoder,output_dim_decoder,  unit1, unit2 , early_stop):
    
        self.input_dim_encoder = input_dim_encoder
        self.input_dim_decoder = input_dim_decoder
        self.output_dim_encoder = output_dim_encoder
        self.output_dim_decoder = output_dim_decoder
        self.early_stop = early_stop
        self.unit1 = unit1
        self.unit2 = unit2

class EncoderDecoderWithoutAttention(EncodeDecoder):
    def __init__(self, input_dim_encoder, input_dim_decoder , output_dim_encoder, output_dim_decoder, unit1, unit2,  early_stop):
        super().__init__(input_dim_encoder, input_dim_decoder , output_dim_encoder, output_dim_decoder,  unit1, unit2, early_stop)
    
    def encoder(self):
        encoder_inputs = Input(shape = [None])
        encoder_embeddings = Embedding(input_dim = self.input_dim_encoder, output_dim =self.output_dim_encoder, mask_zero=True)
        encoder_embedding_output = encoder_embeddings(encoder_inputs)
        encoder_model = LSTM(units=self.unit1, return_sequences=True, return_state=True, seed = 33)
        _, state_h, state_c = encoder_model(encoder_embedding_output)
        encoder_states = (state_h, state_c)
        return encoder_inputs, encoder_states
    
    def decoder(self, encoder_states):
        decoder_inputs = Input(shape = [None])
        decoder_embedding = Embedding(input_dim = self.input_dim_decoder, output_dim =self.output_dim_decoder, mask_zero=True)
        decoder_embedding_output = decoder_embedding(decoder_inputs)
        decoder_model = LSTM(units=self.unit1, return_sequences=True, return_state=True, seed = 33)
        decoder_outputs,_,_ = decoder_model(decoder_embedding_output, initial_state=encoder_states)
        decoder_dense =  Dense(units = self.unit2, activation='softmax')
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
    
    def model_fit(self, model, epochs,batch_size, encoder_input, decoder_input, decoder_output):       
        model.fit([encoder_input, decoder_input],
                                 decoder_output,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks = self.early_stop)
    def save_model(self, model, name):
        model.save('./SavedModel/{}.keras'.format(name))


class EncoderDecoderWithAttention(EncodeDecoder):
    def __init__(self, input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder, unit1, unit2,  early_stop, attention_layer):
        super().__init__(input_dim_encoder, input_dim_decoder , output_dim_encoder,output_dim_decoder,  unit1, unit2, early_stop)
        self.attention_layer = attention_layer

    
    def encoder(self):
            encoder_inputs = Input(shape = [None])
            encoder_embeddings = Embedding(input_dim = self.input_dim_encoder, output_dim =self.output_dim_encoder, mask_zero=False)
            encoder_embedding_output = encoder_embeddings(encoder_inputs)
            encoder_model = LSTM(units=self.unit1, return_sequences=True, return_state=True, seed = 33)
            encoder_outputs, state_h, state_c = encoder_model(encoder_embedding_output)
            encoder_states = (state_h, state_c)
            return encoder_outputs, encoder_inputs, encoder_states

    def decoder(self, encoder_states, encoder_outputs):
            decoder_inputs = Input(shape = [None])
            decoder_embedding = Embedding(input_dim=self.input_dim_decoder, output_dim=self.output_dim_decoder,  mask_zero=False)
            decoder_embedding_output = decoder_embedding(decoder_inputs)
            decoder_model = LSTM(units=self.unit1, return_sequences=True, return_state=True, seed = 33)
            decoder_outputs,_,_ = decoder_model(decoder_embedding_output, initial_state=encoder_states)     
            concatenate_layer = self.attention(decoder_outputs, encoder_outputs)
            decoder_dense = Dense(units=self.unit2, activation='softmax')
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
        encoder_outputs, encoder_inputs, encoder_states = self.encoder()
        decoder_input, decoder_output = self.decoder(encoder_states, encoder_outputs)
        model = Model([encoder_inputs, decoder_input], decoder_output)
        return model
    
    def model_compile(self, model, optimizer, loss, metrics):
        model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
    
    def model_fit(self, model, epochs, batch_size, encoder_input, decoder_input, decoder_output):       
        model.fit([encoder_input, decoder_input],
                                 decoder_output,
                                 epochs=epochs,
                                 batch_size=batch_size, 
                                 callbacks = self.early_stop, 
                                 )
    def save_model(self, model, name):
        model.save('./SavedModel/{}.keras'.format(name))