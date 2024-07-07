# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def data_cleaning(data):
    """
    Clean the given data by replacing infinite values with NaN, removing duplicates, and dropping rows with missing values.

    Args:
        data (pandas.DataFrame): The input data to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned data.
    """
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.drop_duplicates(inplace=True)
    data = data.dropna()

    return data


def preprocess_data(data):
    """
    Preprocess the data by selecting features and applying transformations.

    Args:
        data (pandas.DataFrame): The input data.
        selected_feature_names (list): List of selected feature names.
        binary_features (list): List of binary feature names.
        num_features (list): List of numerical feature names.

    Returns:
        numpy.ndarray: Preprocessed data.
    """
    binary_features = ["dual_sim", "four_g"]
    num_features = ["battery_power", "mobile_wt", "pc", "n_cores"]

    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('binary', 'passthrough', binary_features)
    ])

    X_preprocessed = preprocessor.fit_transform(data)
    return X_preprocessed, preprocessor
