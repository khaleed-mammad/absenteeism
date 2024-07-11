import os
import sys
from dataclasses import dataclass
import numpy as np
# from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
    }
            params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_features': ['sqrt', 'log2', None],
            },
            "Random Forest": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'criterion': ['friedman_mse', 'squared_error'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear', 'saga', 'sag']
            },
            "AdaBoost": {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Find best model based on test F1 score
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_f1_score'])
            best_model = models[best_model_name]
            best_model_test_f1 = model_report[best_model_name]['test_f1_score']

            if best_model_test_f1 < 0.6:
                raise CustomException("No best model found with sufficient F1 score")

            logging.info(f"Best model found: {best_model_name} with F1 score: {best_model_test_f1}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            f1 = f1_score(y_test, predicted)

            return f1

        except Exception as e:
            raise CustomException(e, sys)
            



            
        except Exception as e:
            raise CustomException(e,sys)