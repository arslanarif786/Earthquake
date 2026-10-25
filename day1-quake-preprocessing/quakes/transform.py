"""Task 5: split and preprocess."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from . import config


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test.

    Stratified on config.TARGET, seeded with config.SEED.
    Neither the target nor 'mag' may remain in X.
    """
    X = df.drop(columns=[config.TARGET, "mag"])
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.SEED,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(scaler: str = "robust") -> ColumnTransformer:
    """ColumnTransformer over config.NUMERIC and config.NOMINAL.

    numeric: median imputation, then a scaler chosen by name
             ('standard', 'minmax', 'robust')
    nominal: most-frequent imputation, then one-hot (handle_unknown='ignore')
    """

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    if scaler not in scalers:
        raise ValueError(
            f"Unknown scaler '{scaler}'. "
            "Choose from: standard, minmax, robust"
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scalers[scaler]),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC),
            ("nominal", nominal_pipeline, config.NOMINAL),
        ]
    )
