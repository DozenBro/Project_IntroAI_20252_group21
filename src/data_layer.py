import pandas as pd
import yaml
import os
import joblib
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
    def prepare_data(self):
        print("Starting Data Layer: Preprocessing...")
        
        # Load Data
        db_path = self.config['data']['db_path']
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM students_performance", conn)
        conn.close()
        
        target_col = self.config['data']['target_col']
        
        # Xử lý nhiễu (Outliers)
        df[target_col] = df[target_col].clip(upper=100)
        
        # Train/Test
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # (Fill missing values dùng mode của tập TRAIN)
        missing_cols = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
        imputation_values = {}
        for col in missing_cols:
            mode_val = X_train[col].mode()[0]
            imputation_values[col] = mode_val
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
                
        # Mapping
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
                
        # Scaling
        numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        bins = [0, 64, 69, 74, 100]
        y_train_binned = pd.cut(y_train, bins=bins, labels=[0, 1, 2, 3], include_lowest=True)
        y_test_binned = pd.cut(y_test, bins=bins, labels=[0, 1, 2, 3], include_lowest=True)
        
        # SMOTE sẽ nội suy dữ liệu cho nhóm thiểu số (<60 và >80) một cách tự nhiên
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_binned)
        
        # Trích xuất tên cột để làm feature_names cho LIME
        feature_names = X.columns.tolist()
        
        #  Lưu lại điểm số gốc của tập Test trước khi bị chia rổ
        y_test_actual_scores = y_test.copy()
        
        # Export
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(scaler, 'outputs/models/scaler.pkl')
        joblib.dump(bins, 'outputs/models/bin_edges.pkl')
        
        print("Data Layer complete!")
        
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test_binned, feature_names, y_test_actual_scores