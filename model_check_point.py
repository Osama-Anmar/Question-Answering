
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
def check_point(name):
    return ModelCheckpoint(
        filepath = '{}_model_checkpoint.keras'.format(name),
        monitor='loss',
        save_best_only=True,
        mode='min')