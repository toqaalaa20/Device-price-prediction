# train_model.py
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from preprocessing import data_cleaning, preprocess_data

# Load the data
data = pd.read_csv('train - train.csv')
data = data_cleaning(data)

selected_feature_names = ['battery_power', 'dual_sim', 'four_g', 'mobile_wt', 'n_cores', 'pc']

X = data[selected_feature_names]
# Preprocess the data
X_preprocessed, preprocessor = preprocess_data(X)

# Define the target variable
y = data['price_range']

# Define the model
model = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False)

# Train the model
model.fit(X_preprocessed, y)

# Save the model and preprocessor to a file using pickle
model_file_path = 'logistic_regression_model.pkl'
preprocessor_file_path = 'preprocessor.pkl'

with open(model_file_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(preprocessor_file_path, 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

print(f"Model and preprocessor saved successfully to '{model_file_path}' and '{preprocessor_file_path}'")
