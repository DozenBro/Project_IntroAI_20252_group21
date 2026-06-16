import lime
import lime.lime_tabular
import joblib
import pandas as pd

class XAILayer:
    def __init__(self, model_path='outputs/models/ensemble_model.pkl'):
        self.model = joblib.load(model_path)
        self.class_names = ['Needs Improvement', 'Average', 'Good', 'Excellent']

    def explain_student(self, X_train, student_data, feature_names):
        # 1. Khoi tao LIME cho Classification
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=42
        )
        
        # 2. Wrapper bat buoc phai dung predict_proba
        def predict_proba_wrapper(data_array):
            df = pd.DataFrame(data_array, columns=feature_names)
            return self.model.predict_proba(df)
        
        # Tinh xac suat hien tai cua hoc sinh
        student_df = pd.DataFrame([student_data.values], columns=feature_names)
        probabilities = self.model.predict_proba(student_df)[0]
        predicted_class_idx = int(probabilities.argmax())
        
        # 3. Giai thich nguyen nhan dan den viec sinh vien bi xep vao nhan hien tai
        exp = explainer.explain_instance(
            data_row=student_data.values,
            predict_fn=predict_proba_wrapper,
            num_features=10,
            top_labels=1 # Chi giai thich nhan co xac suat cao nhat
        )
        
        # Lay cac yeu to dong gop (+ hoac -) vao xac suat cua nhan duoc du doan
        lime_reasons = exp.as_list(label=predicted_class_idx)
        
        return lime_reasons, predicted_class_idx, probabilities