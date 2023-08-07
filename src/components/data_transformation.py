import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'goout']
            binary_features = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
            multi_cat_features = ['Mjob', 'Fjob', 'reason', 'guardian']

            num_pipeline = Pipeline([
                ("Imputer", SimpleImputer(strategy="median")),
                ("Scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("Imputer", SimpleImputer(strategy='most_frequent')),
                ('OneHotEncoder', OneHotEncoder())
            ])

            # Here, we're using OneHotEncoder for binary features as well.
            binary_pipeline = Pipeline([
                ("Imputer", SimpleImputer(strategy='most_frequent')),
                ('OneHotEncoder', OneHotEncoder(drop='if_binary'))  # This ensures that if the feature is binary, one of the encoded columns will be dropped
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, multi_cat_features),
                ('binary_pipeline', binary_pipeline, binary_features)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            print(train_df.head())

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="G3"
            numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'goout']

            
            print("G3" in train_df.columns)
            print("G3" in test_df.columns)
            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            