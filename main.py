import os
import joblib # Thêm thư viện này để load Label Encoder
from src.utils import ensure_directories
from src.data_layer import DataProcessor
from src.prediction_layer import PredictionModel
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

def run_automated_pipeline():
    # 1. Tạo sẵn các thư mục lưu trữ
    ensure_directories()
    print("🚀 BẮT ĐẦU CHẠY LUỒNG DỮ LIỆU TỰ ĐỘNG (BACKEND PIPELINE)...")

    # 2. Chạy Tầng 1: Xử lý dữ liệu
    dp = DataProcessor(config_path='config.yaml')
    X_train, X_test, y_train, y_test, feature_names = dp.prepare_data()

    # 3. Chạy Tầng 2: Huấn luyện mô hình
    pm = PredictionModel(config_path='config.yaml')
    model = pm.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Chạy Tầng 3: Giải thích XAI
    print("\n🔍 Đang trích xuất một sinh viên có nguy cơ rớt môn...")
    y_pred = model.predict(X_test)
    dropout_indices = [i for i, pred in enumerate(y_pred) if pred == 0]
    
    if len(dropout_indices) > 0:
        sample_idx = dropout_indices[0]
        student_data = X_test.iloc[sample_idx]
        
        # Chạy XAI để lấy nguyên nhân
        xai = XAILayer(model_path='outputs/models/xgboost_model.pkl')
        lime_reasons = xai.explain_student(X_train, student_data, feature_names)

        # 5. Chạy Tầng 4: Sinh lời khuyên bằng AI
        probs = model.predict_proba([student_data.values])
        dropout_prob = float(probs[0][0]) * 100 
        
        # Phân loại trạng thái AI dự đoán
        if dropout_prob >= 80:
            status_text = f"🚨 NGUY HIỂM - Nguy cơ bị buộc thôi học ({dropout_prob:.1f}%)"
        elif dropout_prob >= 60:
            status_text = f"⚠️ CẢNH CÁO HỌC TẬP - Rủi ro cao ({dropout_prob:.1f}%)"
        elif dropout_prob >= 40:
            status_text = f"🟡 CẦN CẢI THIỆN THÊM - Đang chểnh mảng ({dropout_prob:.1f}%)"
        elif dropout_prob >= 20:
            status_text = f"✅ PHONG ĐỘ ỔN ĐỊNH - Nguy cơ thấp ({dropout_prob:.1f}%)"
        else:
            status_text = f"🌟 PHONG ĐỘ XUẤT SẮC - An toàn ({dropout_prob:.1f}%)"

        # --- ĐOẠN MỚI THÊM ĐỂ SỬA LỖI ---
        # Lấy trạng thái thực tế (Ground Truth) từ tập y_test
        le = joblib.load('outputs/models/label_encoder.pkl')
        actual_label_code = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
        actual_status = le.inverse_transform([actual_label_code])[0]

        adv = AdvisoryLayer()
        # Gọi hàm get_advice với đầy đủ 3 tham số (thêm actual_status)
        prompt, advice = adv.get_advice(lime_reasons, status_text, actual_status, use_mock=False)

        # 6. Lưu báo cáo ra file văn bản (Markdown)
        report_filename = f"outputs/reports/student_{sample_idx}_diagnostic.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(advice)
            
        print(f"\n✅ THÀNH CÔNG: Đã lưu báo cáo tư vấn chi tiết vào file '{report_filename}'!")
    else:
        print("Không tìm thấy sinh viên nào bị dự đoán là Dropout trong tập test này.")

    print("🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH!")

if __name__ == "__main__":
    run_automated_pipeline()