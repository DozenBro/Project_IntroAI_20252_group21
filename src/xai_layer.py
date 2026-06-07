import lime
import lime.lime_tabular
import joblib

class XAILayer:
    """Manages the Explainable AI (XAI) operations using LIME."""
    
    def __init__(self, model_path='outputs/models/xgboost_model.pkl'):
        self.model = joblib.load(model_path)

    def explain_student(self, X_train, student_data, feature_names, class_names=None):
        """Extracts the top contributing features for a specific prediction."""
        if class_names is None:
            class_names = ['Dropout', 'Enrolled', 'Graduate']
            
        # Initialize LIME Explainer with training distribution
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        # Generate explanations for the specific student 
        # FIX: Changed 'student_features' to 'student_data' to match the parameter name
        # FIX: Increased num_features to 10 to capture cumulative minor risks
        exp = explainer.explain_instance(
            data_row=student_data.values,
            predict_fn=self.model.predict_proba,
            num_features=10
        )
        
        return exp.as_list()