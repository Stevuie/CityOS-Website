import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
try:
    df = pd.read_csv('master_shuffled_data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'master_shuffled_data.csv' not found.")
    exit()

print(f"Dataset shape: {df.shape}")
print(f"Unique spots: {sorted(df['SpotID'].unique())}")

# Feature Engineering
def create_features(df):
    """Create enhanced features for better prediction"""
    df = df.copy()
    
    # Time-based features
    df['TimeDecimal'] = df['Hour'] + df['Minute'] / 60.0
    
    # Rush hour features
    df['MorningRush'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)).astype(int)
    df['EveningRush'] = ((df['Hour'] >= 17) & (df['Hour'] <= 19)).astype(int)
    df['LunchTime'] = ((df['Hour'] >= 11) & (df['Hour'] <= 14)).astype(int)
    
    # Weekend vs weekday
    df['Weekend'] = df['Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Business hours
    df['BusinessHours'] = ((df['Hour'] >= 8) & (df['Hour'] <= 18) & 
                          ~df['Day'].isin(['Saturday', 'Sunday'])).astype(int)
    
    # Time of day categories
    df['EarlyMorning'] = (df['Hour'] < 6).astype(int)
    df['LateNight'] = (df['Hour'] >= 22).astype(int)
    
    # Cyclical time features (better for ML)
    df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek'] = pd.Categorical(df['Day']).codes
    
    # Month features
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    return df

# Apply feature engineering
df = create_features(df)

# Define features to use
feature_columns = [
    'Hour', 'Minute', 'TimeDecimal', 'DayOfWeek',
    'MorningRush', 'EveningRush', 'LunchTime', 'Weekend',
    'BusinessHours', 'EarlyMorning', 'LateNight',
    'HourSin', 'HourCos', 'MonthSin', 'MonthCos'
]

print(f"\nFeatures being used: {feature_columns}")

class PerSpotModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def train_spot_model(self, spot_id, X, y):
        """Train a model for a specific spot"""
        print(f"\n{'='*50}")
        print(f"Training model for Spot {spot_id}")
        print(f"{'='*50}")
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        # Handle stratification - only stratify if both classes have at least 2 samples
        stratify_param = y if min(class_counts) >= 2 else None
        if stratify_param is None:
            print("Warning: Cannot stratify due to insufficient samples in one class")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.1,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, random_state=42, class_weight='balanced', max_iter=1000
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', random_state=42, class_weight='balanced'
            )
        }
        
        best_model = None
        best_score = 0
        best_model_name = None
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Cross-validation - handle small datasets
                if len(y_train) >= 10:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(y_train)//2), scoring='accuracy')
                    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                else:
                    print("Skipping cross-validation - insufficient data")
                    cv_scores = np.array([0])
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                print(f"Test accuracy: {test_accuracy:.4f}")
                
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        if best_model is None:
            print(f"Failed to train any model for Spot {spot_id}")
            return None, None
        
        # Store best model and results
        self.models[spot_id] = best_model
        self.scalers[spot_id] = scaler
        
        # Final evaluation
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        self.results[spot_id] = {
            'model_name': best_model_name,
            'test_accuracy': best_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_scores': cv_scores,
            'feature_importance': self._get_feature_importance(best_model, feature_columns) if hasattr(best_model, 'feature_importances_') else None
        }
        
        print(f"\nBest model for Spot {spot_id}: {best_model_name}")
        print(f"Final test accuracy: {best_score:.4f}")
        
        return best_model, scaler
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return None
    
    def train_all_spots(self, df, feature_columns):
        """Train models for all spots"""
        spot_ids = sorted(df['SpotID'].unique())
        
        for spot_id in spot_ids:
            spot_data = df[df['SpotID'] == spot_id].copy()
            
            if len(spot_data) < 50:  # Skip spots with too little data
                print(f"Skipping Spot {spot_id} - insufficient data ({len(spot_data)} samples)")
                continue
            
            X = spot_data[feature_columns]
            y = spot_data['Status']
            
            # Check if we have both classes
            if len(y.unique()) < 2:
                print(f"Skipping Spot {spot_id} - only one class present")
                continue
            
            self.train_spot_model(spot_id, X, y)
    
    def plot_results(self):
        """Plot comprehensive results"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        # Overall accuracy comparison
        accuracies = {spot: results['test_accuracy'] for spot, results in self.results.items()}
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy by spot
        plt.subplot(2, 2, 1)
        spots = list(accuracies.keys())
        scores = list(accuracies.values())
        plt.bar(spots, scores, color='skyblue', alpha=0.7)
        plt.title('Accuracy by Parking Spot')
        plt.xlabel('Spot ID')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, v in enumerate(scores):
            plt.text(spots[i], v + 0.01, f'{v:.3f}', ha='center')
        
        # Model types used
        plt.subplot(2, 2, 2)
        model_counts = {}
        for results in self.results.values():
            model_name = results['model_name']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        if model_counts:
            plt.pie(list(model_counts.values()), labels=list(model_counts.keys()), autopct='%1.1f%%')
            plt.title('Model Types Used')
        
        # Feature importance (for Random Forest models)
        plt.subplot(2, 2, 3)
        rf_spots = [spot for spot, results in self.results.items() 
                   if results['model_name'] == 'RandomForest' and results['feature_importance']]
        
        if rf_spots:
            # Average feature importance across RF models
            avg_importance = {}
            for spot in rf_spots:
                for feature, importance in self.results[spot]['feature_importance'].items():
                    avg_importance[feature] = avg_importance.get(feature, 0) + importance
            
            # Normalize
            total = sum(avg_importance.values())
            avg_importance = {k: v/total for k, v in avg_importance.items()}
            
            # Plot top 10 features
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            
            plt.barh(range(len(features)), importances, color='lightgreen')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importances (Random Forest)')
        
        # Confusion matrix for best performing spot
        plt.subplot(2, 2, 4)
        if self.results:
            best_spot = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
            best_results = self.results[best_spot]
            
            cm = confusion_matrix(best_results['y_test'], best_results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Spot {best_spot} (Best: {best_results["test_accuracy"]:.3f})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}")
        print(f"Total spots modeled: {len(self.results)}")
        if accuracies:
            print(f"Average accuracy: {np.mean(list(accuracies.values())):.4f}")
            print(f"Best accuracy: {max(accuracies.values()):.4f} (Spot {best_spot})")
            print(f"Worst accuracy: {min(accuracies.values()):.4f}")
            print(f"Standard deviation: {np.std(list(accuracies.values())):.4f}")
        
        print(f"\nModel distribution:")
        for model_name, count in model_counts.items():
            print(f"  {model_name}: {count} spots")

# Train models
trainer = PerSpotModelTrainer()
trainer.train_all_spots(df, feature_columns)

# Plot results
trainer.plot_results()

# Save models (optional)
import joblib
for spot_id, model in trainer.models.items():
    joblib.dump(model, f'spot_{spot_id}_model.pkl')
    joblib.dump(trainer.scalers[spot_id], f'spot_{spot_id}_scaler.pkl')

print(f"\nModels saved for spots: {list(trainer.models.keys())}") 