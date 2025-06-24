import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint


try:
    df = pd.read_csv('allDataCombined.csv')
except FileNotFoundError:
    print("Error: 'combinedData.csv' not found.")
    print("Please make sure the CSV file is in the correct directory.")
    sys.exit()

print("--- Data Head ---")
print(df.head())
print("\n--- Data Info ---")
df.info()

if df.isnull().sum().any():
    print("\nWarning: Missing values detected. Consider a strategy to handle them")
    df.dropna(inplace=True)

days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['Day'] = pd.Categorical(df['Day'], categories=days_of_week, ordered=False)

df = pd.get_dummies(df, columns=['Day'], drop_first=True)

features_to_drop = ['SpotID', 'Year', 'Second', 'Status']
features = [col for col in df.columns if col not in features_to_drop]
X = df[features]
y = df['Status']

print("\n--- Features being used for training ---")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


param_dist = {
    'n_estimators': randint(50, 301),
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100,
                                   cv=4, n_jobs=-1, verbose=2, scoring='accuracy', random_state=42)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
print(random_search.best_params_)

print("\n--- Model Evaluation on Test Set ---")
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
class_names = ['0', '1'] 
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

print("\n--- Generating Prediction Plots for Individual Spots ---")

results_df = X_test.copy()
results_df['SpotID'] = df.loc[X_test.index, 'SpotID']
results_df['Actual_Status'] = y_test
results_df['Predicted_Status'] = y_pred

results_df['TimeDecimal'] = results_df['Hour'] + results_df['Minute'] / 60.0

sns.set_style("whitegrid")

spots_to_plot = [1, 2, 3, 4, 5]

for spot_id in spots_to_plot:
    spot_df = results_df[results_df['SpotID'] == spot_id].sort_values('TimeDecimal')
    if spot_df.empty:
        print(f"No data for SpotID {spot_id} in the test set. Skipping plot.")
        continue

    plt.figure(figsize=(16, 7))

    plt.scatter(spot_df['TimeDecimal'], spot_df['Actual_Status'],
                label='Actual Status', color='dodgerblue', alpha=0.8, s=100, marker='o')


    plt.scatter(spot_df['TimeDecimal'], spot_df['Predicted_Status'] + 0.05,
                label='Predicted Status', color='darkorange', alpha=0.7, s=50, marker='X')

    plt.title(f'Parking Status vs. Time of Day for Spot {spot_id}', fontsize=16)
    plt.xlabel('Time of Day (Hour)', fontsize=12)
    plt.ylabel('Status (0=Occupied, 1=Vacant)', fontsize=12)
    plt.yticks([0, 1], ['Occupied', 'Vacant'])
    plt.xticks(np.arange(0, 25, 1)) 
    plt.legend()
    plt.tight_layout()
    plt.show()