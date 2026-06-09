import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering for Regression Task."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
    def prepare_data(self):
        print(" Starting Data Layer: Preprocessing for Regression Task...")
        
        # 1. Load data
        data_path = self.config['data']['raw_path'] 
        df = pd.read_csv(data_path)
        
        print(f"Initial dataset shape: {df.shape}")
        
        # 2. Handle Missing Values (Điền bằng giá trị xuất hiện nhiều nhất - Mode)
        # Trong dataset này có 3 cột bị thiếu dữ liệu (Teacher_Quality, Parental_Education, Distance)
        missing_cols = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
        for col in missing_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        print(f"Total missing values after imputation: {df.isnull().sum().sum()}")
        
        # 3. Smart Ordinal Encoding (Giữ nguyên ý nghĩa phân cấp của biến)
        mapping_dict = {
            'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
            'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
            'Extracurricular_Activities': {'No': 0, 'Yes': 1},
            'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
            'Internet_Access': {'No': 0, 'Yes': 1},
            'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
            'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
            'School_Type': {'Public': 0, 'Private': 1},
            'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
            'Learning_Disabilities': {'No': 0, 'Yes': 1},
            'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
            'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
            'Gender': {'Male': 0, 'Female': 1}
        }
        
        for col, mapping in mapping_dict.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                
        # 4. Separate Features (X) and Target (y)
        target_col = self.config['data']['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 5. Train/Test Split (BỎ SMOTE)
        seed = self.config['preprocessing']['random_state']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['preprocessing']['test_size'], 
            random_state=seed
        )
        
        # 6. Selective Scaling (Chỉ chuẩn hóa các biến liên tục thực sự)
        numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
        
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        # Export mapping & scaler để Tầng Web App / Tư vấn dùng sau này
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(scaler, 'outputs/models/scaler.pkl')
        joblib.dump(mapping_dict, 'outputs/models/mapping_dict.pkl')
        
        feature_names = X.columns.tolist()
        print("Data Layer complete! Data is ready for Regression Modeling.")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names