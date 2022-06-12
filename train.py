import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import math
import mlflow
DATA_DIR = './data'
DATASET_PATH = os.path.join(DATA_DIR,'Titanic+Data+Set.csv')
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR,'Titanic+Data+Set+Preprocess.csv')


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    rmse = math.sqrt(mean_squared_error(y_test,predictions))
    print(rmse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("precision", precision_score(y_test,predictions))
    mlflow.log_metric("acc_score", accuracy_score(y_test,predictions))
    mlflow.log_metric("f1_score", f1_score(y_test,predictions))
    mlflow.log_metric("recall_score", recall_score(y_test,predictions))




if __name__ == '__main__':
    data = pd.read_csv(PROCESSED_DATASET_PATH)


    ## Split the data 
    train, test = train_test_split(data, test_size=0.2)

    X = train.drop('Survived', axis=1)
    y = train['Survived']

    X_test = test.drop('Survived', axis=1)
    y_test = test['Survived']

    
    
    estimators= [10, 20, 30, 40]

    #mlflow.create_experiment("titanic")
    mlflow.set_experiment("titanic")
    for f in range(len(estimators)):
        
        with mlflow.start_run():
            n_estimators = estimators[f]
            mlflow.log_param("n_estimators",n_estimators)
            model = RandomForestClassifier(n_estimators=n_estimators,random_state=0)
            train_model(model, X, y)
            evaluate_model(model, X_test, y_test)
            mlflow.sklearn.log_model(model, "RandomForestClassifier")
            print("model run:", mlflow.active_run().info.run_uuid)
        mlflow.end_run()