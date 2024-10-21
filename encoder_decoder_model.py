from keras import Model  # type: ignore

def build_model(context_input, question_input , decoder_inputs, decoder_output):  
    model = Model(inputs=[context_input, question_input, decoder_inputs], outputs=decoder_output)
    return model

def model_compile(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def model_fit(model, epochs, batch_size, context_data, question_data, decoder_input, decoder_output, early_stop):      
    model.fit([context_data, question_data, decoder_input], decoder_output, epochs=epochs, batch_size=batch_size, callbacks=early_stop)

def summary(model):
    return model.summary()

def save_model(model, name):
    model.save('./SavedModel/{}.keras'.format(name))


