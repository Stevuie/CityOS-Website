import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib

try:
    df = pd.read_csv('masterdata.csv')
except FileNotFoundError:
    print("Error: 'masterdata.csv' not found.")
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

# --- MODIFICATION START: Clean, Single-Line Prediction Visualization ---
print("\n--- Generating Per-Spot Predicted Occupancy Graphs ---")

test_data_with_day = df.loc[X_test.index].copy()
test_data_with_day['Predicted_Status'] = y_pred

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for day in day_order:
    print(f"Generating subplot page for {day}...")
    day_specific_data = test_data_with_day[test_data_with_day['Day'] == day]

    if day_specific_data.empty:
        print(f"No test data available for {day}, skipping.")
        continue

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(22, 28))
    fig.suptitle(f'Predicted Parking Occupancy for {day}', fontsize=24)
    axes = axes.flatten()

    for i, spot_id in enumerate(range(1, 15)):
        ax = axes[i]
        spot_data = day_specific_data[day_specific_data['SpotID'] == spot_id].copy()

        ax.set_title(f'Spot {spot_id}', color='white', fontsize=14)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(6.5, 19.5)
        ax.set_yticks([0, 1])
        ax.set_xticks(np.arange(7, 20, 2))
        ax.grid(True, linestyle='--', alpha=0.2)

        if i % 2 == 0:
            ax.set_yticklabels(['Vacant', 'Occupied'], fontsize=12)
        else:
            ax.set_yticklabels(['', ''])

        if not spot_data.empty:
            spot_data['time'] = spot_data['Hour'] + spot_data['Minute'] / 60.0
            spot_data = spot_data.sort_values('time')

            # --- USE STEP PLOT FOR A CLEAN, SINGLE LINE ---
            # This creates a single, continuous line that jumps between 0 and 1
            ax.step(spot_data['time'], spot_data['Predicted_Status'], where='post', color='#3399FF', linewidth=2.5)

    # --- LEGEND & LAYOUT ---
    # Simplified legend for a single prediction line
    legend_elements = [
        Line2D([0], [0], color='#3399FF', lw=2.5, label='Predicted Status')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=1, fontsize=16)

    fig.text(0.5, 0.02, 'Time of Day (Hour)', ha='center', va='center', fontsize=20)
    fig.text(0.06, 0.5, 'Status', ha='center', va='center', rotation='vertical', fontsize=20)

    fig.subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.1, hspace=0.5, wspace=0.15)
    
    plt.show()
    print(f"Displayed subplot page for {day}.")
# --- MODIFICATION END ---

print("\nModel training and evaluation complete!") 