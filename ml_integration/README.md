# CityOS ML Integration

This directory contains the machine learning integration for the CityOS Parking Predictor.

## Setup

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Place your `video.csv` file in this directory.

3. Train the model:
```bash
python train_model.py
```

4. Start the Flask server:
```bash
python app.py
```

The server will run on http://127.0.0.1:5000

## API Endpoints

### POST /predict
Predicts parking availability for a given spot, day, and time.

Request body:
```json
{
    "spot_id": "spot_1",
    "day": "Monday",
    "hour": "14",
    "minute": "30"
}
```

Response:
```json
{
    "status": "Free",
    "confidence": "85.00%"
}
```

## Integration with Frontend

The frontend can make predictions by sending POST requests to the `/predict` endpoint. The frontend code is already set up to handle the responses and display them appropriately. 