import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def model_eval(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train) #train model
           # y_train_pred = model.predict(X_train)   #predict Training data
            y_test_pred = model.predict(X_test) #predict testing Data

            #get r2 score for training and test data
            #train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report

    except Exception as e:
        logging.info('Exception occured in Model_Eval')
        raise CustomException(e,sys)