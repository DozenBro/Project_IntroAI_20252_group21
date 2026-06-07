import os
import joblib 
from src.utils import ensure_directories
from src.data_layer import DataProcessor
from src.prediction_layer import PredictionModel
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

def run_automated_pipeline():
    """Executes the end-to-end automated machine learning pipeline."""
    
    # 1. Initialize storage directories
    ensure_directories()
    print("🚀 INITIATING AUTOMATED BACKEND PIPELINE...")

    # 2. Execute Data Layer: Data processing and resampling
    dp = DataProcessor(config_path='config.yaml')
    X_train, X_test, y_train, y_test, feature_names = dp.prepare_data()

    # 3. Execute Prediction Layer: Model training
    pm = PredictionModel(config_path='config.yaml')
    model = pm.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Execute XAI Layer: Extract a high-risk student for case study
    print("\n🔍 Extracting a high-risk student profile for analysis...")
    y_pred = model.predict(X_test)
    dropout_indices = [i for i, pred in enumerate(y_pred) if pred == 0]
    
    if len(dropout_indices) > 0:
        sample_idx = dropout_indices[0]
        student_data = X_test.iloc[sample_idx]
        
        # Run LIME to extract root causes of the predicted risk
        xai = XAILayer(model_path='outputs/models/xgboost_model.pkl')
        lime_reasons = xai.explain_student(X_train, student_data, feature_names)

        # 5. Execute Advisory Layer: Generative AI reporting
        probs = model.predict_proba([student_data.values])
        dropout_prob = float(probs[0][0]) * 100 
        
        # Classify AI predicted risk level
        if dropout_prob >= 80:
            status_text = f"🚨 CRITICAL RISK - High chance of dropout ({dropout_prob:.1f}%)"
        elif dropout_prob >= 60:
            status_text = f"⚠️ ACADEMIC WARNING - High risk ({dropout_prob:.1f}%)"
        elif dropout_prob >= 40:
            status_text = f"🟡 NEEDS IMPROVEMENT - Slipping performance ({dropout_prob:.1f}%)"
        elif dropout_prob >= 20:
            status_text = f"✅ STABLE PERFORMANCE - Low risk ({dropout_prob:.1f}%)"
        else:
            status_text = f"🌟 EXCELLENT PERFORMANCE - Safe ({dropout_prob:.1f}%)"

        # Retrieve Ground Truth to provide accurate context for LLM role-prompting
        le = joblib.load('outputs/models/label_encoder.pkl')
        actual_label_code = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
        actual_status = le.inverse_transform([actual_label_code])[0]

        adv = AdvisoryLayer()
        # Generate personalized advice using Groq/Gemini APIs
        prompt, advice = adv.get_advice(lime_reasons, status_text, actual_status, use_mock=False)

        # 6. Export report to a Markdown file
        report_filename = f"outputs/reports/student_{sample_idx}_diagnostic.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(advice)
            
        print(f"\n✅ SUCCESS: Detailed diagnostic report saved to '{report_filename}'!")
    else:
        print("No students predicted as Dropout in the current test set.")

    print("🎉 PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    run_automated_pipeline()