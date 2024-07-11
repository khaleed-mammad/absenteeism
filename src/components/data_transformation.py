import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts", 'preprocessor.pickle')

# class CustomScaler(BaseEstimator, TransformerMixin): 
#     def __init__(self, columns, copy=True, with_mean=True, with_std=True):
#         self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
#         self.columns = columns
#         self.copy = copy
#         self.with_mean = with_mean
#         self.with_std = with_std
#         self.mean_ = None
#         self.var_ = None

#     def fit(self, X, y=None):
#         self.scaler.fit(X[self.columns], y)
#         self.mean_ = np.array(np.mean(X[self.columns]))
#         self.var_ = np.array(np.var(X[self.columns]))
#         return self

#     def transform(self, X, y=None, copy=None):
#         init_col_order = X.columns
#         X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
#         X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
#         return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            columns_to_scale = ['Month Value', 'Transportation Expense', 'Age', 'Body Mass Index', 'Education', 'Children', 'Pets']
            all_columns = ['Reasons_diseases', 'Reasons_pregnancy', 'Reasons_health_symptomps', 'Reasons_light', 
                           'Month Value', 'Day of the week', 'Transportation Expense', 'Distance to Work', 
                           'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets']

            unscaled_columns = [col for col in all_columns if col not in columns_to_scale]

            pipeline = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]), columns_to_scale),
                    ("cat", "passthrough", unscaled_columns)
                ]
            )

            logging.info("Pipeline Created")

            return pipeline
        except Exception as e:
            raise CustomException(e,sys)


        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data - Done")

            logging.info("Starting to transform 'Reason Column'")

            reason_column_train = pd.get_dummies(train_df['Reason for Absence'], drop_first=True).astype(int)
            reason_column_test = pd.get_dummies(test_df['Reason for Absence'], drop_first=True).astype(int)

            train_df = train_df.drop(['Reason for Absence'], axis=1)
            test_df = test_df.drop(['Reason for Absence'], axis=1)

            reason_diseases_train = reason_column_train.loc[:,:14].max(axis=1)
            reason_pregnancy_train = reason_column_train.loc[:,15:17].max(axis=1)
            reason_health_symptomps_train = reason_column_train.loc[:,18:21].max(axis=1)
            reason_lite_train = reason_column_train.loc[:,22:].max(axis=1)

            reason_diseases_test = reason_column_test.loc[:,:14].max(axis=1)
            reason_pregnancy_test = reason_column_test.loc[:,15:17].max(axis=1)
            reason_health_symptomps_test = reason_column_test.loc[:,18:21].max(axis=1)
            reason_lite_test = reason_column_test.loc[:,22:].max(axis=1)

            train_df = pd.concat([train_df,reason_diseases_train,reason_pregnancy_train,reason_health_symptomps_train,reason_lite_train], axis=1)
            test_df = pd.concat([test_df,reason_diseases_test,reason_pregnancy_test,reason_health_symptomps_test,reason_lite_test], axis=1)
            columns_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education',
                            'Children', 'Pets', 'Absenteeism Time in Hours', 'Reasons_diseases', 
                            'Reasons_pregnancy','Reasons_health_symptomps','Reasons_light']
            train_df.columns = columns_names
            test_df.columns = columns_names
            reordered_columns_names = ['Reasons_diseases', 'Reasons_pregnancy','Reasons_health_symptomps',
                                        'Reasons_light','Date', 'Transportation Expense', 'Distance to Work', 
                                        'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                        'Children', 'Pets', 'Absenteeism Time in Hours']
            
            train_df = train_df[reordered_columns_names]
            test_df = test_df[reordered_columns_names]

            logging.info("Finished with transforming 'Reason Column'")

            logging.info("Starting to transform 'Date' column")

            train_df['Date'] = pd.to_datetime(train_df['Date'], format='%d/%m/%Y')
            test_df['Date'] = pd.to_datetime(test_df['Date'], format='%d/%m/%Y')

            list_month = []

            for date in train_df['Date']:
                list_month.append(date.month)

            train_df['Month Value'] = list_month

            list_month = []
            for date in test_df['Date']:
                list_month.append(date.month)

            test_df['Month Value'] = list_month

            list_days_of_week = []
            for date in train_df['Date']:
                list_days_of_week.append(date.weekday())

            train_df['Day of the week'] = list_days_of_week

            list_days_of_week = []
            for date in test_df['Date']:
                list_days_of_week.append(date.weekday())

            test_df['Day of the week'] = list_days_of_week

            train_df = train_df.drop(['Date'], axis=1)
            test_df = test_df.drop(['Date'], axis=1)

            columns_reordered = ['Reasons_diseases', 'Reasons_pregnancy',
                                'Reasons_health_symptomps', 'Reasons_light', 'Month Value',
                                'Day of the week', 'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education',
                                'Children', 'Pets', 'Absenteeism Time in Hours']
            
            train_df=train_df[columns_reordered]
            test_df=test_df[columns_reordered]

            logging.info("Finished with transforming 'Date'")

            logging.info("Starting to transform 'Education' column")

            train_df['Education'] = train_df['Education'].map({1:0, 2:1, 3:1, 4:1})
            test_df['Education'] = test_df['Education'].map({1:0, 2:1, 3:1, 4:1})

            logging.info("Splitting Inputs and Targets started")


            
            target_feature_train_df = np.where(train_df['Absenteeism Time in Hours'] > train_df['Absenteeism Time in Hours'].median(), 1, 0)
            train_df['Excessive Absenteeism'] = target_feature_train_df
            target_feature_test_df = np.where(test_df['Absenteeism Time in Hours'] > test_df['Absenteeism Time in Hours'].median(), 1, 0)
            test_df['Excessive Absenteeism'] = target_feature_test_df 

            input_feature_train_df=train_df.drop(columns=['Absenteeism Time in Hours'],axis=1)
            input_feature_test_df=test_df.drop(columns=['Absenteeism Time in Hours'],axis=1)

            # unscaled_columns_train = train_df.iloc[:, :4]
            # unscaled_columns_test= test_df.iloc[:, :4]
            # columns_to_scale = ['Month Value', 'Transportation Expense', 'Age', 'Body Mass Index', 'Education', 'Children', 'Pets']

            # scaler = CustomScaler(columns=columns_to_scale)

            # input_feature_train_array = scaler.fit_transform(input_feature_train_df)
            # input_feature_test_array = scaler.transform(input_feature_test_df)

            # train_array = np.c_[
            #     unscaled_columns_train, input_feature_train_array, np.array(target_feature_train_df)
            # ]

            # test_array = np.c_[
            #     unscaled_columns_test, input_feature_test_array, np.array(target_feature_test_df)
            # ]

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Fit and transform the training data
            
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)

            # Transform the test data
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]

            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file)
        

        except Exception as e:
            raise CustomException(e, sys)
        

