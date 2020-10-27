import os
import warnings
import sys
from prepro import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from sklearn.metrics import precision_recall_fscore_support as score
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
import logging
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    train = preprocessing()

    train_labels = train['TARGET']
    train = train.drop(['TARGET'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.2, random_state=8)

    LR = float(sys.argv[1])
    NEST = int(sys.argv[2])
    MF = int(sys.argv[3])

    with mlflow.start_run():
        gb_clf = GradientBoostingClassifier(n_estimators=NEST, learning_rate=LR, max_features=MF)
        gb_clf.fit(X_train, y_train)
        pred_gb = gb_clf.predict(X_test)

        precision, recall, fscore, support = score(y_test, pred_gb)
        precision0 = precision[0]
        precision1 = precision[1]
        recall0 = recall[0]
        recall1 = recall[1]
        fscore0 = fscore[0]
        fscore1 = fscore[1]
        support0 = support[0]
        support1 = support[1]
        accuracy = accuracy_score(y_test, pred_gb)

        mlflow.log_param("Learning rates", LR)
        mlflow.log_param("n estimators", NEST)
        mlflow.log_param("max features", MF)
        mlflow.log_metric("precision 0", precision0)
        mlflow.log_metric("precision 1", precision1)
        mlflow.log_metric("recall 0", recall0)
        mlflow.log_metric("recall 1", recall1)
        mlflow.log_metric("fscore 0 ", fscore0)
        mlflow.log_metric("fscore 1 ", fscore1)
        mlflow.log_metric("support 0 ", support0)
        mlflow.log_metric("support 1 ", support1)
        mlflow.log_metric("accuracy ", accuracy)
