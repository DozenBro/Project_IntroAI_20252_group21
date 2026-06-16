import yaml
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

class PredictionModel:
    """Handles training and evaluation of the core Ensemble Classification model."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        seed = self.config['preprocessing']['random_state']
        
        # 1. Initialize Base Classifiers with balanced weights to handle qcut imperfections
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=seed, 
            class_weight='balanced'
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=seed,
            objective='multi:softprob' 
        )
        
        cat = CatBoostClassifier(
            iterations=100, 
            depth=6, 
            learning_rate=0.1, 
            random_seed=seed, 
            verbose=0,
            loss_function='MultiClass',
            auto_class_weights='Balanced'
        )
        
        # Logistic Regression (Classification)
        lr = LogisticRegression(
            max_iter=1000, 
            random_state=seed, 
            class_weight='balanced'
        )

        # 2. Initialize Voting Classifier
        # voting='soft' is explicitly required for LIME predict_proba later
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
        
        return self.model