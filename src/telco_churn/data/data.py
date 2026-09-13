from telco_churn.config import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def load_raw():
    df = pd.read_csv(settings.data_raw_path)
    return df


def clean(df):
    df = df.drop(columns=settings.cols_to_drop)
    return df


def split(df):
    X = df.drop(columns="Churn")
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_target_encoder(y):
    encoder = OrdinalEncoder(categories=[["No", "Yes"]])
    encoder.fit(y.values.reshape(-1, 1))
    return encoder