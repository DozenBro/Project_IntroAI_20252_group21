import yaml
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class PredictionModel:
    """Handles training and evaluation of the core Ensemble Classification model."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        seed = self.config['preprocessing']['random_state']
        
        # Initialize Base Classifiers with balanced weights to handle qcut imperfections
        rf = RandomForestClassifier(
            n_estimators=100, # enough trees for variance reduction
            random_state=seed, # fixed seed 
            class_weight='balanced' # handle dataset imbalance
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, # small tree to avoid overfitting
            learning_rate=0.1, # safe learning rate
            random_state=seed,
            objective='multi:softprob' 
        )
        
        cat = CatBoostClassifier(
            iterations=100, 
            depth=6, 
            learning_rate=0.1, 
            random_seed=seed, 
            verbose=0, # no output during training
            loss_function='MultiClass', 
            auto_class_weights='Balanced'
        )
        
        # Logistic Regression (Classification)
        lr = LogisticRegression(
            max_iter=1000, # ensure convergence
            random_state=seed, 
            class_weight='balanced'
        )

        # Initialize Voting Classifier
        # voting='soft' - %
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('cat', cat),
                ('lr', lr)
            ],
            voting='soft'
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Fits the Ensemble model and evaluates classification metrics."""
        print("Training Ensemble Voting Classifier...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\n" + "="*40)
        print(" ENSEMBLE CLASSIFICATION MODEL RESULTS ")
        print("="*40)
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Export model
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(self.model, 'outputs/models/ensemble_model.pkl')
        
        print("\n[DIAGNOSTIC] Generating and saving evaluation figures...")
        
        # Vẽ Confusion Matrix (Ma trận nhầm lẫn)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Needs Impr (0)', 'Average (1)', 'Good (2)', 'Excellent (3)'], 
                    yticklabels=['Needs Impr (0)', 'Average (1)', 'Good (2)', 'Excellent (3)'])
        plt.title('Confusion Matrix: Model Confusion between Tier 1 & Tier 2')
        plt.ylabel('Actual Tier')
        plt.xlabel('Predicted Tier')
        plt.tight_layout()
        plt.savefig('outputs/figures/confusion_matrix.png')
        plt.close()

        # Vẽ Feature Importance (Lấy từ mô hình Random Forest trong Ensemble)
        rf_model = self.model.named_estimators_['rf']
        importances = rf_model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1][:10] # Lấy top 10
        
        plt.figure(figsize=(10, 6))
        plt.title("Top 10 Feature Importances (Random Forest Base Model)")
        plt.bar(range(10), importances[indices], align="center", color='coral')
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('outputs/figures/feature_importance.png')
        plt.close()
        
        print("[DIAGNOSTIC] SUCCESS: Figures saved to 'outputs/figures/'")
        
        return self.model