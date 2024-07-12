import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pickle'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Reasons_diseases,
                 Reasons_pregnancy,
                 Reasons_health_symptomps,
                 Reasons_light,
                 Month_Value,
                 Day_of_the_Week,
                 Transportation_Expense,
                 Distance_to_Work,
                 Age,
                 Daily_Work_Load_Average,
                 Body_Mass_Index,
                 Education,
                 Children,
                 Pets):
        self.Reasons_diseases = Reasons_diseases
        self.Reasons_pregnancy = Reasons_pregnancy
        self.Reasons_health_symptomps = Reasons_health_symptomps
        self.Reasons_light = Reasons_light
        self.Month_Value = Month_Value
        self.Day_of_the_Week = Day_of_the_Week
        self.Transportation_Expense = Transportation_Expense
        self.Distance_to_Work = Distance_to_Work
        self.Age = Age
        self.Daily_Work_Load_Average = Daily_Work_Load_Average
        self.Body_Mass_Index = Body_Mass_Index
        self.Education = Education
        self.Children = Children
        self.Pets = Pets

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Reasons_diseases': [self.Reasons_diseases],
                'Reasons_pregnancy': [self.Reasons_pregnancy],
                'Reasons_health_symptomps': [self.Reasons_health_symptomps],
                'Reasons_light': [self.Reasons_light],
                'Month Value': [self.Month_Value],
                'Day of the Week': [self.Day_of_the_Week],
                'Transportation Expense': [self.Transportation_Expense],
                'Distance to Work': [self.Distance_to_Work],
                'Age': [self.Age],
                'Daily Work Load Average': [self.Daily_Work_Load_Average],
                'Body Mass Index': [self.Body_Mass_Index],
                'Education': [self.Education],
                'Children': [self.Children],
                'Pets': [self.Pets]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
