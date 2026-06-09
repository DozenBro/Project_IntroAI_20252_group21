import lime
import lime.lime_tabular
import joblib
import pandas as pd

class XAILayer:
    """Manages the Explainable AI (XAI) operations using LIME for Regression."""
    
    def __init__(self, model_path='outputs/models/ensemble_model.pkl'):
        # Tải siêu mô hình Ensemble Voting Regressor
        self.model = joblib.load(model_path)

    def explain_student(self, X_train, student_data, feature_names):
        """Extracts the top contributing features for a specific score prediction."""
        
        # 1. Initialize LIME Explainer for REGRESSION
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            mode='regression', # [QUAN TRỌNG NHẤT] Chuyển từ classification sang regression
            random_state=42
        )
        
        # 2. Hàm Wrapper: Đảm bảo dữ liệu từ LIME truyền vào mô hình có đúng tên cột
        def ensemble_predict_wrapper(data_array):
            df = pd.DataFrame(data_array, columns=feature_names)
            # Dùng predict (để lấy điểm số) thay vì predict_proba (lấy xác suất)
            return self.model.predict(df)
        
        # 3. Chạy giải thích cho sinh viên cụ thể
        exp = explainer.explain_instance(
            data_row=student_data.values,
            predict_fn=ensemble_predict_wrapper, 
            num_features=10
        )
        
        # 4. Trả về list các nguyên nhân (Bài toán Hồi quy KHÔNG CẦN label=0 nữa)
        return exp.as_list()