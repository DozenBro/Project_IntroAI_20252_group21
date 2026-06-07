import yaml
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

class PredictionModel:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.model = xgb.XGBClassifier(
            n_estimators=self.config['model']['xgboost']['n_estimators'],
            max_depth=self.config['model']['xgboost']['max_depth'],
            learning_rate=self.config['model']['xgboost']['learning_rate'],
            random_state=self.config['preprocessing']['random_state']
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        print("2. Đang huấn luyện mô hình XGBoost...")
        self.model.fit(X_train, y_train)

        # Dự đoán
        y_pred = self.model.predict(X_test)
        
        # Đánh giá
        acc = accuracy_score(y_test, y_pred)
        print(f"-> Huấn luyện xong! Độ chính xác (Accuracy): {acc:.4f}")
        
        # Lưu mô hình lại để ứng dụng Web sử dụng
        joblib.dump(self.model, 'outputs/models/xgboost_model.pkl')
        print("-> Đã lưu mô hình XGBoost vào outputs/models/xgboost_model.pkl")
        
        return self.model

# Code test thử sự kết nối giữa Layer 1 và Layer 2
if __name__ == "__main__":
    from data_layer import DataProcessor
    
    # 1. Chạy Tầng Data
    dp = DataProcessor(config_path='config.yaml')
    X_train, X_test, y_train, y_test, cols = dp.prepare_data()
    
    # 2. Chạy Tầng Prediction
    pm = PredictionModel(config_path='config.yaml')
    model = pm.train_and_evaluate(X_train, y_train, X_test, y_test)