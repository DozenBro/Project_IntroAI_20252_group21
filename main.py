import os
from src.utils import ensure_directories
from src.data_layer import DataProcessor
from src.prediction_layer import PredictionModel
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

def run_automated_pipeline():
    """Executes the end-to-end automated machine learning pipeline for Score Prediction."""
    
    # 1. Initialize storage directories
    ensure_directories()
    print("🚀 INITIATING AUTOMATED BACKEND PIPELINE (REGRESSION)...")

    # 2. Execute Data Layer
    dp = DataProcessor(config_path='config.yaml')
    X_train, X_test, y_train, y_test, feature_names = dp.prepare_data()

    # 3. Execute Prediction Layer
    pm = PredictionModel(config_path='config.yaml')
    model = pm.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Execute XAI Layer: Extract a student with a low predicted score for case study
    print("\n🔍 Extracting a student profile for case study analysis...")
    y_pred = model.predict(X_test)
    
    # Tìm một sinh viên dự đoán được dưới 65 điểm để phân tích lý do học kém
    low_score_indices = [i for i, pred in enumerate(y_pred) if pred < 65]
    
    if len(low_score_indices) > 0:
        sample_idx = low_score_indices[0]
        student_data = X_test.iloc[sample_idx]
        predicted_score = float(y_pred[sample_idx])
        
        # Trích xuất điểm thực tế (Ground Truth)
        actual_score = float(y_test.iloc[sample_idx]) if hasattr(y_test, 'iloc') else float(y_test[sample_idx])
        
        print(f"Analyzing student at index {sample_idx}. Predicted Score: {predicted_score:.1f}, Actual: {actual_score:.1f}")
        
        # Run LIME to extract score boosters and reducers
        xai = XAILayer(model_path='outputs/models/ensemble_model.pkl')
        lime_reasons = xai.explain_student(X_train, student_data, feature_names)

        # 5. Execute Advisory Layer
        adv = AdvisoryLayer()
        # Lưu ý: Chỉnh use_mock=False nếu bạn đã cắm API Key thật vào code
        prompt, advice = adv.get_advice(lime_reasons, predicted_score, actual_score, use_mock=True)

        # 6. Export report to a Markdown file
        report_filename = f"outputs/reports/student_{sample_idx}_regression_diagnostic.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(advice)
            
        print(f"\n✅ SUCCESS: Detailed diagnostic report saved to '{report_filename}'!")
    else:
        print("No students with low predicted scores found in the test set.")

    print("🎉 PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    run_automated_pipeline()