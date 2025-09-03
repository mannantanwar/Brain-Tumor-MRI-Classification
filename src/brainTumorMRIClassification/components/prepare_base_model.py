import os
import urllib.request as request
import zipfile
from pathlib import Path
from brainTumorMRIClassification import logger
import tensorflow as tf
from brainTumorMRIClassification.utils.common import get_size
from brainTumorMRIClassification.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        from tensorflow.keras import layers, models

        image_shape = tuple(self.config.params_image_size)
        SEED = self.config.params_seed
        N_TYPES = self.config.params_classes

        self.model = models.Sequential(
            [
                # Convolutional layer 1
                layers.Conv2D(32, (4, 4), activation="relu", input_shape=image_shape),
                layers.MaxPooling2D(pool_size=(3, 3)),
                # Convolutional layer 2
                layers.Conv2D(64, (4, 4), activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3)),
                # Convolutional layer 3
                layers.Conv2D(128, (4, 4), activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3)),
                # Convolutional layer 4
                layers.Conv2D(128, (4, 4), activation="relu"),
                layers.Flatten(),
                # Fully connected layers
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.4, seed=SEED),
                layers.Dense(N_TYPES, activation="softmax"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.params_learning_rate, beta_1=0.869, beta_2=0.995
        )
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model.summary()
        self.save_model(path=self.config.base_model_path, model=self.model)

    # def update_base_model(self):
    # For this custom Sequential model, update_base_model can simply save the compiled model again if needed
    # self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model=tf.keras.Model):
        model.save(path)
