import joblib
import lime
import lime.lime_tabular

class XAILayer:
    def __init__(self, model_path='outputs/models/xgboost_model.pkl'):
        # Load mô hình XGBoost đã được train từ Tầng 2
        self.model = joblib.load(model_path)

    def explain_student(self, X_train, student_features, feature_names, target_names=['Dropout', 'Enrolled', 'Graduate']):
        print("3. Đang chạy XAI (LIME) để tìm nguyên nhân...")
        
        # Khởi tạo bộ giải thích LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=target_names,
            mode='classification',
            random_state=42
        )
        
        # Giải thích dự đoán cho đúng 1 sinh viên được truyền vào
        exp = explainer.explain_instance(
            data_row=student_features.values,
            predict_fn=self.model.predict_proba,
            num_features=5
        )
        
        print("-> Đã trích xuất xong nguyên nhân từ LIME!")
        return exp.as_list()