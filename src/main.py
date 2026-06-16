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

    # 1. Execute Data Layer
    dp = DataProcessor(config_path='config.yaml')
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, y_test_actual_scores = dp.prepare_data()

    # 2. Execute Prediction Layer
    pm = PredictionModel(config_path='config.yaml')
    model = pm.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

    # 3. Execute XAI Layer
    print("\nExtracting a student profile for case study analysis...")
    y_pred = model.predict(X_test_scaled)
    
    # Tim sinh vien duoc du doan thuoc nhom 0 (Needs Improvement)
    low_score_indices = [i for i, pred in enumerate(y_pred) if pred == 0]
    
    if len(low_score_indices) > 0:
        sample_idx = low_score_indices[0]
        student_data_scaled = X_test_scaled.iloc[sample_idx]
        predicted_class = int(y_pred[sample_idx])
        actual_score = float(y_test_actual_scores.iloc[sample_idx])
        
        print(f"Analyzing student at index {sample_idx}. Predicted Class: {predicted_class}, Actual Score: {actual_score:.1f}")
        
        xai = XAILayer(model_path='outputs/models/ensemble_model.pkl')
        lime_reasons, predicted_class_idx, probabilities = xai.explain_student(X_train_scaled, student_data_scaled, feature_names)

        # 4. Tinh toan Gap Analysis bang du lieu goc (Chua Scale)
        scaler = joblib.load('outputs/models/scaler.pkl')
        numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
        
        # Lay muc tieu la nhung sinh vien thuoc nhom 2 (Good) hoac 3 (Excellent)
        target_idx = (y_train == 2) | (y_train == 3)
        target_profile_scaled = X_train_scaled.loc[target_idx].mean()
        
        # Inverse transform de LLM doc duoc con so thuc te (Vi du: 5 gio hoc, thay vi -1.2)
        student_unscaled = student_data_scaled.copy()
        target_unscaled = target_profile_scaled.copy()
        
        student_num_df = pd.DataFrame([student_data_scaled[numeric_cols].values], columns=numeric_cols)
        target_num_df = pd.DataFrame([target_profile_scaled[numeric_cols].values], columns=numeric_cols)
        
        student_unscaled[numeric_cols] = scaler.inverse_transform(student_num_df)[0]
        target_unscaled[numeric_cols] = scaler.inverse_transform(target_num_df)[0]

        # 5. Execute Advisory Layer
        adv = AdvisoryLayer()
        prompt, advice = adv.get_advice(
            lime_reasons=lime_reasons, 
            predicted_class=predicted_class_idx, 
            probabilities=probabilities,
            actual_score=actual_score, 
            student_data=student_unscaled,
            target_profile=target_unscaled,
            use_mock=True # Doi thanh False khi muon goi API that
        )

        # 6. Export Report
        report_filename = f"outputs/reports/student_{sample_idx}_classification_diagnostic.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(advice)
            
        print(f"\nSUCCESS: Detailed diagnostic report saved to '{report_filename}'!")
    else:
        print("No students in the 'Needs Improvement' class found in the test set.")

    print("PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    run_automated_pipeline()