import warnings
import sys
from prepro import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
from sklearn.metrics import accuracy_score


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    train = preprocessing()

    train_labels = train['TARGET']
    train = train.drop(['TARGET'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.2, random_state=8)

    NEST = int(sys.argv[1])
    MWFL = float(sys.argv[2])
    MID = float(sys.argv[3])
    with mlflow.start_run():
        rf_clf = RandomForestClassifier(n_estimators=NEST, min_weight_fraction_leaf=MWFL,min_impurity_decrease=MID)

        rf_clf.fit(X_train, y_train)

        y_pred_rm = rf_clf.predict(X_test)

        precision, recall, fscore, support = score(y_test, y_pred_rm)
        precision0 = precision[0]
        precision1 = precision[1]
        recall0 = recall[0]
        recall1 = recall[1]
        fscore0 = fscore[0]
        fscore1 = fscore[1]
        support0 = support[0]
        support1 = support[1]
        accuracy=accuracy_score(y_test,y_pred_rm)

        mlflow.log_param("n estimators", NEST)
        mlflow.log_param("min weight fraction leaf", MWFL)
        mlflow.log_metric("precision 0", precision0)
        mlflow.log_metric("precision 1", precision1)
        mlflow.log_metric("recall 0", recall0)
        mlflow.log_metric("recall 1", recall1)
        mlflow.log_metric("fscore 0 ", fscore0)
        mlflow.log_metric("fscore 1 ", fscore1)
        mlflow.log_metric("support 0 ", support0)
        mlflow.log_metric("support 1 ", support1)
        mlflow.log_metric("accuracy ", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(rf_clf, "rf_clf", registered_model_name="Random forest")
        else:
            mlflow.sklearn.log_model(rf_clf, "rf_clf")
