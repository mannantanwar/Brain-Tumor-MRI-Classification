from brainTumorMRIClassification.config.configuration import ConfigurationManager
from brainTumorMRIClassification.components.training import (
    Training,
)
from brainTumorMRIClassification.components.prepare_callback import (
    PrepareCallback,
)
from brainTumorMRIClassification import logger


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callbacks_list = prepare_callbacks.get_all_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callbacks_list)


if __name__ == "__main__":
    try:
        logger.info(f"*************************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
