import os
import urllib.request as request
from zipfile import ZipFile
from brainTumorMRIClassification.entity.config_entity import PrepareCallbacksConfig
import tensorflow as tf
import time


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor="loss", min_delta=1e-9, patience=8, verbose=True
        )

    @property
    def _create_reduce_lr(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, verbose=True
        )

    def get_all_callbacks(self):
        return [self._create_early_stopping, self._create_reduce_lr]
