from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model_file_path = 'logistic_regression_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data (device specifications)
        device_specs = request.get_json()

        # Prepare the input data for prediction
        input_data = pd.DataFrame([device_specs])

        # Perform any necessary preprocessing on input_data here (e.g., feature scaling, encoding)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        binary_features = ["dual_sim", "four_g"]
        num_features = ["battery_power", "pc", "mobile_wt", "n_cores"]

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('binary', 'passthrough', binary_features)
        ])
        input_preprocessed = preprocessor.fit_transform(input_data)

        # Make a prediction using the loaded model
        predicted_price = model.predict(input_preprocessed)

        # Return the predicted price as a JSON response
        response = {'predicted_price': predicted_price[0]}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
