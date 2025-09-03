from pathlib import Path
from brainTumorMRIClassification.utils.common import read_yaml, create_directories
from pathlib import Path
import os
from brainTumorMRIClassification import logger
from brainTumorMRIClassification.constants import *
from brainTumorMRIClassification.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_seed=self.params.SEED,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories(
            [Path(model_ckpt_dir), Path(config.tensorboard_root_log_dir)]
        )

        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir)
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(prepare_base_model.base_model_path),
            training_data=Path(training_data),
            param_epochs=params.EPOCHS,
            param_batch_size=params.BATCH_SIZE,
            param_is_augmentation=params.AUGMENTATION,
            param_image_size=params.IMAGE_SIZE,
            param_learning_rate=params.LEARNING_RATE,
            param_seed=params.SEED,
        )

    def get_validation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            path_of_model=Path("artifacts/training/brain_model.h5"),
            training_data=Path("artifacts/data_ingestion/brain-mri/Training"),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
