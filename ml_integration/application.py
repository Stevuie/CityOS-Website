from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from flask_cors import CORS  # Add CORS support
import os
import sys
import csv

# Initialize the Flask application
app = Flask(__name__)
CORS(app, origins=["http://localhost:8000", "https://stevuie.github.io"])  # Enable CORS for the GitHub Pages frontend
application = app

# Load the trained model and the day map
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, 'parking_model.pkl'))
    day_map = joblib.load(os.path.join(base_dir, 'day_map.pkl'))
    print("Model and day map loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please run train_model.py first.")
    exit()


@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    try:
        spot_id = data['spot_id']
        day_name = data['day']
        hour = int(data['hour'])
        minute = int(data['minute'])

        # Create a DataFrame from the input
        current_time = datetime.now()
        example_data = {
            'SpotID': [spot_id],
            'Month': [current_time.month],
            'Day': [day_name],
            'Year': [current_time.year],
            'Hour': [hour],
            'Minute': [minute]
        }
        new_data_point = pd.DataFrame(example_data)

        # One-hot encode 'Day' just like in training
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        new_data_point['Day'] = pd.Categorical(new_data_point['Day'], categories=days_of_week, ordered=True)
        new_data_point = pd.get_dummies(new_data_point, columns=['Day'], drop_first=True)

        # SpotID: extract numeric part if needed
        if 'SpotID' in new_data_point.columns:
            new_data_point['SpotID'] = new_data_point['SpotID'].str.replace('spot_', '').astype(int)

        # Remove columns not used in training
        features_to_drop = ['Year', 'Date', 'Month', 'Second', 'Status']
        for col in features_to_drop:
            if col in new_data_point.columns:
                new_data_point = new_data_point.drop(columns=[col])

        # Ensure all expected columns are present
        expected_features = list(model.feature_names_in_)
        for col in expected_features:
            if col not in new_data_point.columns:
                new_data_point[col] = 0
        new_data_point = new_data_point[expected_features]

        # --- Make Prediction ---
        prediction_numeric = model.predict(new_data_point)[0]
        prediction_proba = model.predict_proba(new_data_point)[0]

        prediction_status = 'Occupied' if prediction_numeric == 1 else 'Free'
        confidence = prediction_proba[np.argmax(prediction_proba)]

        print(f"Input: {data}, Prediction: {prediction_status}, Confidence: {confidence:.2%}")

        return jsonify({
            'status': prediction_status,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    try:
        spot_ids = data['spot_ids']  # List of spot IDs
        day_name = data['day']
        hour = int(data['hour'])
        minute = int(data['minute'])

        current_time = datetime.now()
        # Build a DataFrame for all spots
        example_data = {
            'SpotID': spot_ids,
            'Month': [current_time.month] * len(spot_ids),
            'Day': [day_name] * len(spot_ids),
            'Year': [current_time.year] * len(spot_ids),
            'Hour': [hour] * len(spot_ids),
            'Minute': [minute] * len(spot_ids)
        }
        new_data_point = pd.DataFrame(example_data)

        # One-hot encode 'Day' just like in training
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        new_data_point['Day'] = pd.Categorical(new_data_point['Day'], categories=days_of_week, ordered=True)
        new_data_point = pd.get_dummies(new_data_point, columns=['Day'], drop_first=True)

        # SpotID: extract numeric part if needed
        if 'SpotID' in new_data_point.columns:
            new_data_point['SpotID'] = new_data_point['SpotID'].str.replace('spot_', '').astype(int)

        # Remove columns not used in training
        features_to_drop = ['Year', 'Date', 'Month', 'Second', 'Status']
        for col in features_to_drop:
            if col in new_data_point.columns:
                new_data_point = new_data_point.drop(columns=[col])

        # Ensure all expected columns are present
        expected_features = list(model.feature_names_in_)
        for col in expected_features:
            if col not in new_data_point.columns:
                new_data_point[col] = 0
        new_data_point = new_data_point[expected_features]

        # --- Make Predictions ---
        prediction_numeric = model.predict(new_data_point)
        prediction_proba = model.predict_proba(new_data_point)

        results = {}
        for i, spot_id in enumerate(spot_ids):
            status = 'Occupied' if prediction_numeric[i] == 1 else 'Free'
            confidence = prediction_proba[i][np.argmax(prediction_proba[i])]
            results[spot_id] = {
                'status': status,
                'confidence': f"{confidence:.2%}"
            }
        return jsonify(results)
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.after_request
def add_private_network_header(response):
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 