import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import sys
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    df = pd.read_csv('masterdata1.csv')
except FileNotFoundError:
    print("Error: 'masterdata1.csv' not found.")
    sys.exit()

# --- Data Preparation (No Changes) ---
print("Head:")
print(df.head())
if df.isnull().sum().any():
    print("\nWarning: Missing values detected. Dropping rows with null values.")
    df.dropna(inplace=True)
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['Day'] = pd.Categorical(df['Day'], categories=days_of_week, ordered=True)
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['Day'], drop_first=True)
features_to_drop = ['Year', 'Date', 'Month', 'Second', 'Status']
features = [col for col in df_model.columns if col not in features_to_drop and col != 'Day']
X = df_model[features]
y = df_model['Status']
print("\nFeatures for training:")
print(X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- Model Training (No Changes) ---
print("\n--- Training Model with Pre-defined Best Parameters ---")
best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)
best_rf.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation (No Changes) ---
print("\n--- Model Evaluation on Test Set ---")
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
class_names = ['Vacant', 'Occupied']
print(classification_report(y_test, y_pred, target_names=class_names))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n--- Feature Importances ---")
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df)
print(f"\nTraining Accuracy: {accuracy_score(y_train, best_rf.predict(X_train)):.4f}")

# --- Save the Trained Model ---
joblib.dump(best_rf, 'parking_model.pkl')
print("\nModel saved to parking_model.pkl")

# --- Save the day map for later use ---
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
joblib.dump(day_map, 'day_map.pkl')
print("Day map saved to day_map.pkl")

print("\nModel training and evaluation complete!") 