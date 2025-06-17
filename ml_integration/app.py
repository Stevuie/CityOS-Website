from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from flask_cors import CORS  # Add CORS support

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and the day map
try:
    model = joblib.load('parking_model.pkl')
    day_map = joblib.load('day_map.pkl')
    print("Model and day map loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please run train_model.py first.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # --- Prepare data for prediction ---
    try:
        spot_id = data['spot_id']
        day_name = data['day']
        hour = int(data['hour'])
        minute = int(data['minute'])

        # Create a DataFrame from the input
        # Use current month and year for the prediction
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

        # Apply the same transformations as in training
        new_data_point['SpotID_numeric'] = new_data_point['SpotID'].str.split('_').str[1].astype(int)
        new_data_point['Day_numeric'] = new_data_point['Day'].map(day_map)
        
        features_for_prediction = ['SpotID_numeric', 'Month', 'Day_numeric', 'Year', 'Hour', 'Minute']
        new_data_features = new_data_point[features_for_prediction]

        # --- Make Prediction ---
        prediction_numeric = model.predict(new_data_features)[0]
        prediction_proba = model.predict_proba(new_data_features)[0]

        prediction_status = 'Occupied' if prediction_numeric == 1 else 'Free'
        confidence = prediction_proba[np.argmax(prediction_proba)]

        # Return the result as JSON
        return jsonify({
            'status': prediction_status,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000) 