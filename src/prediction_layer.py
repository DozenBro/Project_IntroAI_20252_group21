import yaml
import joblib
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from catboost import CatBoostRegressor

class PredictionModel:
    """Handles the training and evaluation of the core Ensemble Regression model."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        seed = self.config['preprocessing']['random_state']
        
        # 1. Initialize Base Regressors (Thuật toán dự đoán số)
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            random_state=seed
        )
        
        cat = CatBoostRegressor(
            iterations=100, depth=6, learning_rate=0.1, 
            random_seed=seed, verbose=0
        )
        
        # Thay thế Logistic Regression bằng Ridge Regression cho bài toán số
        ridge = Ridge(alpha=1.0, random_state=seed)

        # 2. Initialize Voting Regressor
        # Lưu ý: VotingRegressor tự động lấy trung bình cộng dự đoán của 4 mô hình, không cần voting='soft' nữa
        self.model = VotingRegressor(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('cat', cat),
                ('ridge', ridge)
            ]
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Fits the Ensemble model and evaluates its regression metrics."""
        print("Training Ensemble Voting Regressor (RF, XGB, CatBoost, Ridge)...")
        self.model.fit(X_train, y_train)
        
        # Dự đoán điểm số trên tập Test
        y_pred = self.model.predict(X_test)
        
        # Chấm điểm mô hình
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\n" + "="*40)
        print(" ENSEMBLE REGRESSION MODEL RESULTS ")
        print("="*40)
        print(f"Mean Absolute Error (MAE): {mae:.2f} points")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} points")
        print(f"R-squared (R2 Score): {r2:.4f}")
        print("="*40 + "\n")
        
        # Export trained model
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(self.model, 'outputs/models/ensemble_model.pkl')
        
        return self.model