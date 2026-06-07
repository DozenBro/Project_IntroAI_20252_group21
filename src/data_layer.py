import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

class DataProcessor:
    """Handles data loading, preprocessing, and resampling."""
    
    def __init__(self, config_path='config.yaml'):
        # Load configurations from YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def prepare_data(self):
        """Executes the full data preprocessing pipeline."""
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.config['data']['raw_path'], sep=';')
        
        # Drop columns that are not relevant to the local context
        cols_to_drop = ['Age at enrollment', 'Inflation rate', 'GDP', 'Unemployment rate']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Separate features (X) and target (y)
        X = df.drop(columns=[self.config['data']['target_col']])
        y = df[self.config['data']['target_col']]
        feature_names = X.columns.tolist()

        # Encode categorical target into numerical values (0, 1, 2)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Save LabelEncoder for UI mapping later
        os.makedirs('outputs/models', exist_ok=True)
        joblib.dump(le, 'outputs/models/label_encoder.pkl')

        # Split data using stratify to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        # Apply SMOTE to oversample minority classes (e.g., Dropout) in Training data
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Apply Z-score standardization to all features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=feature_names)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
        
        # Save Scaler for real-time inference in UI
        joblib.dump(scaler, 'outputs/models/scaler.pkl')

        return X_train_scaled, X_test_scaled, y_train_res, y_test, feature_names