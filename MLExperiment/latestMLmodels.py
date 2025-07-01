import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import matplotlib.colors as mcolors

try:
    df = pd.read_csv('master_shuffled_data.csv')
except FileNotFoundError:
    print("Error: 'master_shuffled_data.csv' not found.")
    sys.exit()

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


param_grid = {
    'n_estimators': [50, 100, 200, 300], 
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],     
    'min_samples_leaf': [1, 2, 4],       
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1, verbose=2) 

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("\nBest Parameters found by RandomizedSearchCV:")
print(grid_search.best_params_)

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


print("\n--- Generating Parking Status Heatmap for Each Day ---")

test_data_with_day = df.loc[X_test.index].copy()
test_data_with_day['Predicted_Status'] = y_pred

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for day in day_order:
    print(f"Generating heatmap for {day}...")
    
    day_specific_data = test_data_with_day[test_data_with_day['Day'] == day]
    
    if day_specific_data.empty:
        print(f"No test data available for {day}, skipping heatmap.")
        continue

    heatmap_data = day_specific_data.pivot_table(
        index='SpotID', 
        columns='Hour', 
        values='Predicted_Status', 
        aggfunc=lambda x: x.mode()[0] if not x.empty else np.nan
    )

    all_hours = np.arange(0, 24)
    heatmap_data = heatmap_data.reindex(columns=all_hours).sort_index()

    cmap = mcolors.ListedColormap(['green', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(20, 12))
    sns.heatmap(
        heatmap_data, 
        cmap=cmap, 
        norm=norm, 
        cbar_kws={'ticks': [0, 1]}, 
        linewidths=.5, 
        linecolor='lightgray'
    )

    cbar = plt.gca().collections[0].colorbar
    if cbar is not None:
        cbar.set_ticklabels(['Vacant', 'Occupied'])

    plt.title(f'Predicted Parking Status for {day}', fontsize=18, pad=20)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Parking Spot ID', fontsize=14)
    plt.xticks(ticks=np.arange(0.5, 24.5), labels=[str(i) for i in range(24)], rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()