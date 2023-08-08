import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


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

    def get_data_transformer_object(self, numerical_columns, binary_features, multi_cat_features):
        try:
            # numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
            # binary_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
            # multi_cat_features = ['Mjob', 'Fjob', 'reason', 'guardian']

            num_pipeline = Pipeline([
                ("Imputer", SimpleImputer(strategy="median")),
                ("Scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("Imputer", SimpleImputer(strategy='most_frequent')),
                ('OneHotEncoder', OneHotEncoder())
            ])


            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, multi_cat_features),
                ('binary_pipeline', 'passthrough', binary_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'goout']
            binary_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
            multi_cat_features = ['Mjob', 'Fjob', 'reason', 'guardian']

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            for col in binary_features:
                train_df[col] = train_df[col].map({'yes': 1, 'no': 0}).fillna(0)
                test_df[col] = test_df[col].map({'yes': 1, 'no': 0}).fillna(0)

            

            target_column_name = "G3"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]


            correlation_with_target = input_feature_train_df.corrwith(target_feature_train_df).sort_values(ascending=False)

            # Selecting features with an absolute correlation greater than the threshold
            selected_features_names = correlation_with_target[correlation_with_target.abs() > .1].index.tolist()
            

            selected_numerical_columns = [col for col in numerical_columns if col in selected_features_names]
            selected_binary_features = [col for col in binary_features if col in selected_features_names]
            selected_multi_cat_features = [col for col in multi_cat_features if col in selected_features_names]

            logging.info("Obtaining preprocessing object")

            selected_preprocessor = self.get_data_transformer_object(selected_numerical_columns, selected_binary_features, selected_multi_cat_features)

            logging.info(f"Selected features based on correlation: {', '.join(selected_features_names)}")


            # Apply preprocessing using the selected preprocessor
            input_feature_train_arr = selected_preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = selected_preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=selected_preprocessor
            )


            logging.info(f"Saved preprocessing object.")



            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                selected_features_names
            )
        except Exception as e:
            raise CustomException(e, sys)