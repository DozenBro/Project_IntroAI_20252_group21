import pandas as pd
import yaml
import os
import joblib
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering for Ordinal Classification Task."""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
    def prepare_data(self):
        print(" Starting Data Layer: Preprocessing for Classification Task...")
        
        # Load data
        #data_path = self.config['data']['raw_path'] 
        #df = pd.read_csv(data_path)
        db_path = self.config['data']['db_path']
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Không tìm thấy Database tại {db_path} chạy 'python create_db.py' trước!")
            
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM students_performance", conn)
        conn.close()
        print(f"Initial dataset shape: {df.shape}")
        
        target_col = self.config['data']['target_col']
        
        # Dùng pd.qcut để tự động chia điểm thành 4 nhóm có số lượng sinh viên bằng nhau:
        # 0: Needs Improvement (Dưới nhóm 25%)
        # 1: Average (25% - 50%)
        # 2: Good (50% - 75%)
        # 3: Excellent (Top 25%)
        df['Target_Class'], bins = pd.qcut(df[target_col], q=4, labels=[0, 1, 2, 3], retbins=True)
        
        print(f"\n [DATA INSIGHT] Dynamic Class Boundaries (Bin Edges): {np.round(bins, 2)}")
        print("Class Distribution:")
        print(df['Target_Class'].value_counts().sort_index())
        
        #  Tách Features (X) và Target (y)
        X = df.drop(columns=[target_col, 'Target_Class'])
        y = df['Target_Class']
        
        # Lưu lại điểm gốc
        actual_scores = df[target_col] 
        
        seed = self.config['preprocessing']['random_state']
        
        # Tách Train/Test NGAY TỪ ĐẦU và dùng stratify=y để đảm bảo 

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, df.index, # Truyền thêm df.index để theo dõi vị trí dòng dữ liệu
            test_size=self.config['preprocessing']['test_size'], 
            random_state=seed,
            stratify=y 
        )
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        #  Handle Missing Values (Chỉ học Mode từ Train)
        missing_cols = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
        imputation_values = {}
        
        for col in missing_cols:
            if col in X_train.columns:
                mode_val = X_train[col].mode()[0]
                imputation_values[col] = mode_val # Lưu lại làm lịch sử
                
                # Áp dụng giá trị của Train cho cả Train và Test
                X_train[col] = X_train[col].fillna(mode_val)
                X_test[col] = X_test[col].fillna(mode_val)
                
        #  Smart Ordinal Encoding
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
            if col in X_train.columns:
                X_train[col] = X_train[col].map(mapping)
                X_test[col] = X_test[col].map(mapping)
                
        #  Selective Scaling 
        numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
        
        scaler = StandardScaler() 
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        #  Export Artifacts (Scaler, Mapping Dict, Imputation Values, Bin Edges) 
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(scaler, 'outputs/models/scaler.pkl')
        joblib.dump(mapping_dict, 'outputs/models/mapping_dict.pkl')
        joblib.dump(imputation_values, 'outputs/models/imputation_values.pkl')
        joblib.dump(bins, 'outputs/models/bin_edges.pkl') 
        
        feature_names = X.columns.tolist()
        
        # Trích xuất điểm gốc của tập Test để lát in ra báo cáo
        y_test_actual_scores = actual_scores.loc[indices_test]
        
        print("\n Data Layer complete! Data is correctly binned and scaled without leakage.")
        
        # Trả về thêm tập y_test_actual_scores để hàm main() tái sử dụng
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, y_test_actual_scores