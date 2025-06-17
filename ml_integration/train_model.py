import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib # Used for saving the model

print("Starting model training...")

# --- Data Loading and Preprocessing ---
file_path = 'video.csv'
df = pd.read_csv(file_path)

# --- Feature Engineering ---
df.dropna(subset=['SpotID', 'Day'], inplace=True)
df['SpotID_numeric'] = df['SpotID'].str.split('_', expand=True)[1]
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['Day_numeric'] = df['Day'].map(day_map)
df.dropna(subset=['SpotID_numeric', 'Day_numeric'], inplace=True)
df['SpotID_numeric'] = df['SpotID_numeric'].astype(int)
df['Day_numeric'] = df['Day_numeric'].astype(int)

features = ['SpotID_numeric', 'Month', 'Day_numeric', 'Year', 'Hour', 'Minute']
target = 'Status'
X = df[features]
y = df[target]

# We identified Random Forest as the best model, so we will train it on all data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Model training complete.")

# --- Save the Trained Model ---
joblib.dump(model, 'parking_model.pkl')
print("Model saved to parking_model.pkl")

# --- Save the feature map for later use ---
joblib.dump(day_map, 'day_map.pkl')
print("Day map saved to day_map.pkl") 