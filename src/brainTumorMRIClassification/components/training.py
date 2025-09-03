import os
import urllib.request as request
import zipfile
from pathlib import Path
from brainTumorMRIClassification import logger
from brainTumorMRIClassification.entity.config_entity import TrainingConfig
import tensorflow as tf


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.base_model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.param_learning_rate,
                beta_1=0.869,
                beta_2=0.995,
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def train_valid_generator(self):
        train_dir = os.path.join(self.config.training_data, "Training")
        test_dir = os.path.join(self.config.training_data, "Testing")
        image_size = self.config.param_image_size
        target_size = tuple(
            image_size[:2]
        )  # Only height and width for flow_from_directory
        batch_size = self.config.param_batch_size
        SEED = self.config.param_seed
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Data augmentation and preprocessing for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            brightness_range=(0.85, 1.15),
            width_shift_range=0.002,
            height_shift_range=0.002,
            shear_range=12.5,
            zoom_range=0,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="nearest",
        )
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            seed=SEED,
        )

        # No augmentation for test data, just rescaling
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.valid_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
            seed=SEED,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info(f"Model saved at {path}")

    def train(self, callback_list: list):
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        logger.info("Training Started")
        self.model.fit(
            self.train_generator,
            epochs=self.config.param_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=callback_list,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
