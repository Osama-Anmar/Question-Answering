
import tensorflow as tf
def check_point(name):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath = '{}_model_checkpoint.h5'.format(name),
        monitor='loss',
        save_best_only=True,
        mode='min')