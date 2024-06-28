# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

model_file_path = 'logistic_regression_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)


price_categories = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction form page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the form data
        battery_power = int(request.form['battery_power'])
        dual_sim = int(request.form['dual_sim'])
        four_g = int(request.form['four_g'])
        mobile_wt = int(request.form['mobile_wt'])
        n_cores = int(request.form['n_cores'])
        pc = int(request.form['pc'])

        input_data = pd.DataFrame([[battery_power, dual_sim, four_g, mobile_wt, n_cores, pc]],
                                  columns=['battery_power', 'dual_sim', 'four_g', 'mobile_wt', 'n_cores', 'pc'])

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

        predicted_price = model.predict(input_preprocessed)

        predicted_price_category = int(predicted_price[0])

        price_categories = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3:"Very High Cost"}
        response = {'predicted_price_category': price_categories[predicted_price_category]}
        return jsonify(response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
