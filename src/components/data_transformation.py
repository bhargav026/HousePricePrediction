import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OrdinalEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transfomation is started')

            # devide features into ordinal encoded and which should be scaled
            #numerical_columns = X.columns[X.dtypes != 'object']
            #categorical_columns = X.columns[X.dtypes == 'object']
            numerical_columns = ['carat','depth','table','x','y','z']
            categorical_columns = ['cut','color','clarity']
            # define th custom ranking for each ordinal variable
            cut_map = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_map = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_map = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1','IF']

            logging.info('Pipeline Initiated')
            ## Numerical  Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            ## categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_map, color_map, clarity_map])),
                    ('scalar', StandardScaler())
                    # if one hot encoding is used then no need to do scaling for cat_features

                ]

            )
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            logging.info('Pipeline Completed')
            return preprocessor
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info('Read Train Test data Completed')
            logging.info(f'train DataFrame head: \n{train_data.head().to_string()}')
            logging.info(f'test DataFrame head: \n{test_data.head().to_string()}')

            logging.info("obtaining Preprocessor object")
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_colomns = [target_column_name, 'id']

            input_feature_train_data = train_data.drop(columns=drop_colomns, axis = 1)
            target_feature_train_data = train_data[target_column_name]
            input_feature_test_data = test_data.drop(columns=drop_colomns, axis = 1)
            target_feature_test_data = test_data[target_column_name]

            ## Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            save_object(file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj
                        )
            logging.info('Pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Exception occured in the Initiate_data_transformation')
            raise CustomException(e,sys)
