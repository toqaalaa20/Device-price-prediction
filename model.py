from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle




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


data = pd.read_csv('train - train.csv')
data = data_cleaning(data)

selected_feature_names = ['battery_power', 'dual_sim', 'four_g', 'mobile_wt', 'n_cores', 'pc']

X = data.drop(['price_range'],axis=1)
y = data['price_range']

X = X[selected_feature_names]

binary_features = ["dual_sim",  "four_g"]
num_features= ["battery_power", "mobile_wt", "pc", "n_cores"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('binary', 'passthrough', binary_features)
])

X_preprocessed = preprocessor.fit_transform(X)

# Define the model
model = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False)


print(X_preprocessed)

model.fit(X_preprocessed, y)

# Define the file path to save the model
model_file_path = 'logistic_regression_model.pkl'

# Save the model to a file using pickle
with open(model_file_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved successfully to '{model_file_path}'")