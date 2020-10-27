import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def preprocessing():
    df_train = pd.read_csv('application_train.csv')
    df_test = pd.read_csv('application_test.csv')
    le = LabelEncoder()
    le_count = 0
    for col in df_train:
        if df_train[col].dtype == 'object':
                # If 2 or fewer unique categories
            if len(list(df_train[col].unique())) <= 2:
                    # Train on the training data
                le.fit(df_train[col])
                    # Transform both training and testing data
                df_train[col] = le.transform(df_train[col])
                df_test[col] = le.transform(df_test[col])
                le_count += 1
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    imp = SimpleImputer(strategy="most_frequent")
    train = pd.DataFrame(imp.fit_transform(df_train))
    train.columns = df_train.columns
    train.index = df_train.index
    train = pd.DataFrame(imp.fit_transform(df_train))
    train.columns = df_train.columns
    train.index = df_train.index
    print('%d columns were label encoded.' % le_count)

    print('%d columns were label encoded.' % le_count)
    return train
