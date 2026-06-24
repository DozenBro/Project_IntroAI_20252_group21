import os
import joblib
import pandas as pd
from src.utils import ensure_directories
from src.data_layer import DataProcessor
from src.prediction_layer import PredictionModel
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

def run_automated_pipeline():
    ensure_directories()
    print("INITIATING AUTOMATED BACKEND PIPELINE (CLASSIFICATION)...")

# Data Layer
    dp = DataProcessor(config_path='config.yaml')
    
    X_train_resampled, X_test_scaled, y_train_resampled, y_test_binned, feature_names, y_test_actual_scores = dp.prepare_data()

    # Prediction Layer
    pm = PredictionModel(config_path='config.yaml')
    
    model = pm.train_and_evaluate(X_train_resampled, y_train_resampled, X_test_scaled, y_test_binned)

    # XAI Layer
    print("\nExtracting specific student profiles for 3 core Case Studies...")
    y_pred = model.predict(X_test_scaled)
    
    # (Needs Improvement)
    tier_0_indices = [i for i, pred in enumerate(y_pred) if pred == 0]
    
    if len(tier_0_indices) > 0:
        # Load scaler & chuẩn bị "Target Profile"  trước khi vào vòng lặp
        scaler = joblib.load('outputs/models/scaler.pkl')
        numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
        
        target_idx = (y_train_resampled == 2) | (y_train_resampled == 3)
        target_profile_scaled = X_train_resampled.loc[target_idx].mean()
        
        # 3 case studies đại diện
        min_income_encoded = X_test_scaled['Family_Income'].min()
        max_income_encoded = X_test_scaled['Family_Income'].max()
        
        case_1_idx = None # Sinh viên nghèo
        case_2_idx = None # Sinh viên giàu
        case_3_idx = None # Nghịch lý Burnout 
        
        for idx in tier_0_indices:
            student_row = X_test_scaled.iloc[idx]
            # Săn Case 1
            if case_1_idx is None and student_row['Family_Income'] == min_income_encoded:
                case_1_idx = idx
            # Săn Case 2
            if case_2_idx is None and student_row['Family_Income'] == max_income_encoded:
                case_2_idx = idx
            # Săn Case 3 
            if case_3_idx is None and student_row['Hours_Studied'] > 0.5: 
                case_3_idx = idx
                
        # Fallback 
        case_1_idx = case_1_idx if case_1_idx is not None else tier_0_indices[0]
        case_2_idx = case_2_idx if case_2_idx is not None else (tier_0_indices[1] if len(tier_0_indices) > 1 else tier_0_indices[0])
        case_3_idx = case_3_idx if case_3_idx is not None else tier_0_indices[-1]
        
        case_studies = [
            ("Case1_LowIncome", case_1_idx),
            ("Case2_HighIncome", case_2_idx),
            ("Case3_BurnoutParadox", case_3_idx)
        ]
        
        # Vòng lặp XAI & LLM cho 3 Case
        for case_name, sample_idx in case_studies:
            print(f"\n---> Executing {case_name} (Test Set Index: {sample_idx})")
            
            student_data_scaled = X_test_scaled.iloc[sample_idx]
            actual_score = float(y_test_actual_scores.iloc[sample_idx])
            
            # Giải thích mô hình
            xai = XAILayer(model_path='outputs/models/ensemble_model.pkl')
            lime_reasons, predicted_class_idx, probabilities = xai.explain_student(X_train_resampled, student_data_scaled, feature_names)

            # Inverse Transform 
            student_unscaled = student_data_scaled.copy()
            target_unscaled = target_profile_scaled.copy()
            
            student_num_df = pd.DataFrame([student_data_scaled[numeric_cols].values], columns=numeric_cols)
            target_num_df = pd.DataFrame([target_profile_scaled[numeric_cols].values], columns=numeric_cols)
            
            student_unscaled[numeric_cols] = scaler.inverse_transform(student_num_df)[0]
            target_unscaled[numeric_cols] = scaler.inverse_transform(target_num_df)[0]

            # Tư vấn học tập bằng LLM
            adv = AdvisoryLayer()
            prompt, advice = adv.get_advice(
                lime_reasons=lime_reasons, 
                predicted_class=predicted_class_idx, 
                probabilities=probabilities,
            #    actual_score=actual_score, 
                student_data=student_unscaled,
                target_profile=target_unscaled,
                use_mock=False 
            )

            # Lưu File Markdown
            report_filename = f"outputs/reports/{case_name}_Student{sample_idx}_Diagnostic.md"
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(advice)
                
            print(f"SUCCESS: Report saved to '{report_filename}'!")
            
    else:
        print("No students in the 'Needs Improvement' class found in the test set.")

    print("\nPIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    run_automated_pipeline()