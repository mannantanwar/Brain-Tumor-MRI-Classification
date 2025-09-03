import os
import kagglehub
import shutil
from brainTumorMRIClassification.entity.config_entity import DataIngestionConfig
from brainTumorMRIClassification import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):

            # Download latest version
            path = kagglehub.dataset_download(self.config.source_URL)
            logger.info(f"Downloaded file from kaggle hub: {path}")
            # Move the downloaded file to self.config.root_dir
            dest_path = os.path.join(self.config.root_dir, "brain-mri")
            os.makedirs(self.config.root_dir, exist_ok=True)
            shutil.move(path, dest_path)
            logger.info(f"Moved file to: {dest_path}")
        else:
            logger.info(f"File already exists: {self.config.local_data_file}")
