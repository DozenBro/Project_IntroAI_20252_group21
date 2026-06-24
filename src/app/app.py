import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(src_dir, ".."))
sys.path.append(project_root) 
sys.path.append(src_dir)


from xai_layer import XAILayer
from advisory_layer import AdvisoryLayer
from data_layer import DataProcessor

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="SmartEdu AI Advisor", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .tier-0 { color: #c62828; font-size: 36px !important; font-weight: bold; } /* Needs Improvement */
    .tier-1 { color: #f9a825; font-size: 36px !important; font-weight: bold; } /* Average */
    .tier-2 { color: #2e7d32; font-size: 36px !important; font-weight: bold; } /* Good */
    .tier-3 { color: #1565c0; font-size: 36px !important; font-weight: bold; } /* Excellent */
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_training_data():
    from src.data_layer import DataProcessor
    dp = DataProcessor(config_path='config.yaml')
    X_train_scaled, _, _, _, _, _ = dp.prepare_data()
    return X_train_scaled

@st.cache_data
def load_target_profile():
    df = pd.read_csv('data/raw/StudentPerformanceFactors.csv')
    target_col = 'Exam_Score'
    if target_col in df.columns:
        df['Target_Class'] = pd.qcut(df[target_col], q=4, labels=[0, 1, 2, 3])
        target_group = df[(df['Target_Class'] == 2) | (df['Target_Class'] == 3)]
        return target_group.mean(numeric_only=True)
    return None

# 2. CACHE & LOAD SYSTEM COMPONENTS
@st.cache_resource
def load_backend_components():
    scaler = joblib.load('outputs/models/scaler.pkl')
    mapping_dict = joblib.load('outputs/models/mapping_dict.pkl')
    imputation_values = joblib.load('outputs/models/imputation_values.pkl')
    bin_edges = joblib.load('outputs/models/bin_edges.pkl') 
    xai_layer = XAILayer('outputs/models/ensemble_model.pkl')
    adv_layer = AdvisoryLayer()
    return scaler, mapping_dict, imputation_values, xai_layer, adv_layer, bin_edges

try:
    # Nhận đúng và đủ 6 biến
    scaler, mapping_dict, imputation_values, xai_layer, adv_layer, bin_edges = load_backend_components()
    target_profile_unscaled = load_target_profile()
    X_train_scaled = load_training_data()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}. Vui lòng chạy main.py trước để sinh file mô hình.")
    st.stop()

class_names = ['Needs Improvement', 'Average', 'Good', 'Excellent']
css_classes = ['tier-0', 'tier-1', 'tier-2', 'tier-3']

# 3. UI LAYOUT
st.title("🎓 SmartEdu: AI-Driven Academic Navigation")
st.markdown("AI-powered academic tier prediction and personalized roadmap generation system based on XAI.")

with st.sidebar:
    st.header("Student Parameters")
    # Lấy các biến quan trọng cho demo. (Thực tế bạn cần đủ các trường trong dataset)
    hours_studied = st.slider("Hours Studied/Week", 0, 50, 15)
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    sleep_hours = st.slider("Sleep Hours/Night", 4, 10, 6)
    tutoring = st.slider("Tutoring Sessions/Month", 0, 10, 0)
    prev_scores = st.slider("Previous Scores", 0, 100, 60)
    phys_active = st.slider("Physical Activity (Days/Week)", 0, 7, 2)
    
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    # Bạn có thể thêm các input khác tùy thuộc vào mapping_dict...
    
    st.markdown("---")
    selected_llm = st.selectbox("Select AI Advisor", ["Groq (Llama-3.3)", "Gemini (1.5 Flash)"])
    predict_btn = st.button("Analyze & Generate Roadmap", type="primary", use_container_width=True)

if predict_btn:
    # 4. DATA PIPELINE CHUẨN BỊ INPUT
    numeric_features = {
        'Hours_Studied': hours_studied, 'Attendance': attendance, 
        'Sleep_Hours': sleep_hours, 'Previous_Scores': prev_scores, 
        'Tutoring_Sessions': tutoring, 'Physical_Activity': phys_active
    }
    
    expected_columns = X_train_scaled.columns.tolist()
    
    # Khởi tạo dictionary bám sát theo đúng expected_columns
    student_dict = {col: "Medium" if col in mapping_dict else 0 for col in expected_columns}
    
    # Cập nhật giá trị người dùng nhập vào
    student_dict.update(numeric_features)
    student_dict['Family_Income'] = family_income
    
    #cột phải giống hệt lúc train
    df_student = pd.DataFrame([student_dict])
    df_student = df_student[expected_columns] 
    
    # Apply Mapping
    for col, mapping in mapping_dict.items():
        if col in df_student.columns:
            # Fallback về 1 (Medium/Neutral) nếu gặp lỗi
            df_student[col] = df_student[col].map(mapping).fillna(1) 
            
    # Giữ lại bản unscaled để đưa vào LLM
    student_unscaled = df_student.iloc[0].copy()
    
    # Apply Scaling
    # Định nghĩa chính xác thứ tự cột numeric đã dùng lúc train
    numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
    df_student_scaled = df_student.copy()
    df_student_scaled[numeric_cols] = scaler.transform(df_student[numeric_cols])
    
    student_data_scaled = df_student_scaled.iloc[0]

    # 5. EXECUTION LAYER
    with st.spinner("Running ML Classification & LIME XAI..."):
        lime_reasons, predicted_class, probabilities = xai_layer.explain_student(
            X_train=X_train_scaled, 
            student_data=student_data_scaled, 
            feature_names=expected_columns # Truyền danh sách cột đã chuẩn hóa vào LIME
        )
    
    # 6. DISPLAY RESULTS
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Step 1: Machine Learning Diagnosis")
        st.markdown("### Predicted Academic Tier:")
        
        # Lấy mốc điểm dưới và trên dựa vào predicted_class
        lower_bound = bin_edges[predicted_class]
        upper_bound = bin_edges[predicted_class + 1]
        range_text = f"({lower_bound:.1f} - {upper_bound:.1f} points)"
        
        # In ra màn hình kết hợp cả Chữ và Số
        st.markdown(f"<span class='{css_classes[predicted_class]}'>{class_names[predicted_class]} {range_text}</span>", unsafe_allow_html=True)
        
        st.write("**Tier Probabilities:**")
        for i, prob in enumerate(probabilities):
            st.progress(float(prob), text=f"{class_names[i]}: {prob*100:.1f}%")
        
        st.markdown("---")
        boosters = [r for r in lime_reasons if r[1] > 0]
        reducers = [r for r in lime_reasons if r[1] < 0]
        
        if reducers:
            st.error("**🔴 Risk Factors (Pushing Tier Down):**")
            for c, w in reducers: st.write(f"- {c} (Impact: {w:.2f})")
                
        if boosters:
            st.success("**🟢 Retention Factors (Holding Tier Up):**")
            for c, w in boosters: st.write(f"- {c} (Impact: +{w:.2f})")

    with col2:
        st.subheader("Step 2: GenAI Advisory Roadmap")
        with st.spinner("Generating personalized advisory roadmap..."):
            prompt, advice = adv_layer.get_advice(
                lime_reasons, predicted_class, probabilities, 
                 student_unscaled, target_profile_unscaled, 
                use_mock=False, provider=selected_llm
            )
            
            with st.expander("View System Prompt Context (Debug)", expanded=False):
                st.code(prompt, language="markdown")
                
            st.markdown(advice)