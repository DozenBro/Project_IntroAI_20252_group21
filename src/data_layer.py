import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Đảm bảo thư mục lưu model tồn tại
        os.makedirs('outputs/models', exist_ok=True)

    def prepare_data(self):
        print("1. Đang tải và tiền xử lý dữ liệu...")
        # Đọc dữ liệu
        df = pd.read_csv(self.config['data']['raw_path'], sep=';')
        
        X = df.drop(columns=[self.config['data']['target_col']])
        y = df[self.config['data']['target_col']]

        # Label Encoding
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        joblib.dump(le, 'outputs/models/label_encoder.pkl') # Lưu lại cho UI dùng

        # Chia tập Train/Test (Stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config['preprocessing']['test_size'], 
            random_state=self.config['preprocessing']['random_state'],
            stratify=y_encoded
        )

        # SMOTE
        smote = SMOTE(random_state=self.config['preprocessing']['random_state'])
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, 'outputs/models/scaler.pkl') # Lưu lại cho UI dùng

        # Chuyển lại thành DataFrame để giữ tên cột (Cần thiết cho XGBoost và LIME)
        X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

        print("-> Data Layer hoàn tất! Đã lưu Scaler và LabelEncoder.")
        return X_train_final, X_test_final, y_train_res, y_test, X.columns.tolist()

# Test thử file nếu chạy trực tiếp
if __name__ == "__main__":
    dp = DataProcessor()
    X_train, X_test, y_train, y_test, cols = dp.prepare_data()