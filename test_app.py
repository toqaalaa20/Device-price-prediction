# test_app.py
import unittest
import json
from app import app  # Assuming app.py is your Flask application file
import pandas as pd
import numpy as np
from preprocessing import data_cleaning, preprocess_data
import json

class TestPreprocessing(unittest.TestCase):
    def test_data_cleaning(self):
        data = pd.DataFrame({
            'battery_power': [np.inf, 2000, 3000],
            'dual_sim': [0, 1, 1],
            'four_g': [1, 0, 1],
            'mobile_wt': [100, 200, 300],
            'n_cores': [4, 8, 12],
            'pc': [2, 4, 8],
            'price_range': [0, 1, 2]
        })
        cleaned_data = data_cleaning(data)
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)

    def test_preprocess_data(self):
        data = pd.DataFrame({
            'battery_power': [2000, 3000],
            'dual_sim': [1, 1],
            'four_g': [0, 1],
            'mobile_wt': [200, 300],
            'n_cores': [8, 12],
            'pc': [4, 8]
        })
        X_preprocessed, _ = preprocess_data(data)
        self.assertEqual(X_preprocessed.shape[1], 6)


class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        response = self.app.post('/predict', data={
            'battery_power': 2000,
            'dual_sim': 1,
            'four_g': 1,
            'mobile_wt': 200,
            'n_cores': 8,
            'pc': 4
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('predicted_price_category', data)
        self.assertIn(data['predicted_price_category'], ['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'])


if __name__ == '__main__':
    unittest.main()