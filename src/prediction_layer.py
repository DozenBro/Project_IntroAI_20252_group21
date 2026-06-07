import xgboost as xgb
from sklearn.metrics import accuracy_score
import yaml
import joblib
import os

class PredictionModel:
    """Handles the training and evaluation of the core classification model."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # [FIX BUG] Updated the config path to match the nested ['xgboost'] structure in config.yaml
        self.model = xgb.XGBClassifier(
            n_estimators=self.config['model']['xgboost']['n_estimators'],
            max_depth=self.config['model']['xgboost']['max_depth'],
            learning_rate=self.config['model']['xgboost']['learning_rate'],
            random_state=self.config['preprocessing']['random_state'],
            eval_metric='mlogloss'
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Fits the model and evaluates its accuracy."""
        print("Training XGBoost classification model...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        
        # Export trained model for inference
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(self.model, 'outputs/models/xgboost_model.pkl')
        
        return self.model