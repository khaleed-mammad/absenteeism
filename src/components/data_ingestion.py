import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainer
# from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfiguration:
    train_dataset_path:str = os.path.join("artifacts", 'train.csv')
    test_dataset_path:str = os.path.join("artifacts", 'test.csv')
    raw_dataset_path:str = os.path.join("artifacts", 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfiguration()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            df=pd.read_csv("notebook/Absenteeism_data.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_dataset_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_dataset_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=2)
            train_set.to_csv(self.ingestion_config.train_dataset_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_dataset_path, index=False, header=True)
            logging.info("Ingestion is completed")

            return (self.ingestion_config.train_dataset_path, 
                    self.ingestion_config.test_dataset_path, 
                    self.ingestion_config.raw_dataset_path
                    )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()