import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os 

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, G2: int, G1: int, Medu: int, higher: str, paid: str, studytime: int, Fedu: int, internet: str, goout: int, traveltime: int, romantic: str, age: int, failures: int):
        self.G2 = G2
        self.G1 = G1
        self.Medu = Medu
        self.higher = self.map_binary_feature(higher)
        self.paid = self.map_binary_feature(paid)
        self.studytime = studytime
        self.Fedu = Fedu
        self.internet = self.map_binary_feature(internet)
        self.goout = goout
        self.traveltime = traveltime
        self.romantic = self.map_binary_feature(romantic)
        self.age = age
        self.failures = failures

    @staticmethod
    def map_binary_feature(value: str) -> int:
        return 1 if value == 'yes' else 0

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "G2": [self.G2],
                "G1": [self.G1],
                "Medu": [self.Medu],
                "higher": [self.higher],
                "paid": [self.paid],
                "studytime": [self.studytime],
                "Fedu": [self.Fedu],
                "internet": [self.internet],
                "goout": [self.goout],
                "traveltime": [self.traveltime],
                "romantic": [self.romantic],
                "age": [self.age],
                "failures": [self.failures],
            }
        
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

