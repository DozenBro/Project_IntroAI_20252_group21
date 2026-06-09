import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.data_layer import DataProcessor
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="SmartEdu AI Advisor", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .score-green { color: #2e7d32; font-size: 48px !important; font-weight: bold; }
    .score-red { color: #c62828; font-size: 48px !important; font-weight: bold; }
    .score-yellow { color: #f9a825; font-size: 48px !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# 2. LOAD BACKEND SYSTEM (Cached for speed)

@st.cache_resource
def load_system_backend():
    print("Loading Backend System for Web App...")
    dp = DataProcessor(config_path='config.yaml')
    X_train, X_test, y_train, y_test, feature_names = dp.prepare_data()
    
    # Check if model exists
    if not os.path.exists('outputs/models/ensemble_model.pkl'):
        st.error("🚨 Model not found! Please run 'python main.py' in your terminal first to train the model.")
        st.stop()
        
    model = joblib.load('outputs/models/ensemble_model.pkl')
    xai_layer = XAILayer(model_path='outputs/models/ensemble_model.pkl')
    adv_layer = AdvisoryLayer()
    
    return X_train, X_test, y_train, y_test, feature_names, model, xai_layer, adv_layer

try:
    X_train, X_test, y_train, y_test, feature_names, model, xai_layer, adv_layer = load_system_backend()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# 3. SIDEBAR & STATE MANAGEMENT

st.sidebar.title("🎓 SmartEdu System")
st.sidebar.write("Student Performance Prediction & Diagnostics")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ AI Settings")

# Thêm Dropdown chọn Model AI
selected_llm = st.sidebar.selectbox("Select GenAI Model:", ["Groq (Llama-3.3)", "Google (Gemini 1.5)"])

if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = None

st.sidebar.markdown("---")
if st.sidebar.button("🎲 Extract Random Student", use_container_width=True):
    st.session_state.selected_idx = random.randint(0, len(X_test) - 1)

# 4. MAIN DASHBOARD
st.title("📊 AI Academic Diagnostic Dashboard")

if st.session_state.selected_idx is not None:
    idx = st.session_state.selected_idx
    student_data = X_test.iloc[idx]
    
    # Thực hiện dự đoán bằng Ensemble Model
    # model.predict nhận mảng 2D, trả về mảng 1D. Lấy phần tử [0]
    predicted_score = float(model.predict(student_data.values.reshape(1, -1))[0])
    
    # Lấy điểm thực tế (Ground Truth)
    actual_score = float(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else float(y_test[idx])
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("🤖 **AI Predicted Exam Score**")
        # Đổi màu điểm số tùy theo mức độ
        if predicted_score >= 80:
            score_class = "score-green"
            status_text = "🌟 EXCELLENT"
        elif predicted_score >= 65:
            score_class = "score-yellow"
            status_text = "✅ STABLE / GOOD"
        else:
            score_class = "score-red"
            status_text = "🚨 NEEDS IMPROVEMENT"
            
        st.markdown(f"<p class='{score_class}'>{predicted_score:.1f} / 100</p>", unsafe_allow_html=True)
        st.info(f"**Status:** {status_text}")
        
    with col2:
        st.write("👨‍🏫 **Actual Score (Ground Truth)**")
        st.markdown(f"<p class='big-font' style='color: #555;'>{actual_score:.1f} / 100</p>", unsafe_allow_html=True)
        st.write(f"*AI prediction error margin: {abs(predicted_score - actual_score):.1f} points*")

    st.markdown("---")
    
    #PHẦN 2: LIME XAI & LLM ADVISOR
    st.markdown("### 🔍 Root Cause Analysis & AI Advice")
    
    col_xai, col_llm = st.columns([1, 1.2], gap="large")
    
    with col_xai:
        st.subheader("Step 1: Point Extraction (LIME)")
        with st.spinner("Extracting score boosters and reducers..."):
            # Chạy LIME (Phiên bản Regression)
            lime_reasons = xai_layer.explain_student(X_train, student_data, feature_names)
            
            boosters = [] # Các yếu tố làm TĂNG điểm (Weight > 0)
            reducers = [] # Các yếu tố làm GIẢM điểm (Weight < 0)
            minor = []
            
            for condition, weight in lime_reasons:
                if abs(weight) < 0.5: # Lọc các yếu tố tác động dưới 0.5 điểm
                    minor.append((condition, weight))
                elif weight > 0:
                    boosters.append((condition, weight))
                else:
                    reducers.append((condition, weight))
                    
            # Hiển thị Điểm bị trừ (Reducers)
            if reducers:
                st.error("**🔴 Score Reducers (Lost Points):**")
                for c, w in reducers:
                    st.write(f"- {c} (Impact: **{w:.2f} pts**)")
            
            # Hiển thị Điểm được cộng (Boosters)
            if boosters:
                st.success("**🟢 Score Boosters (Added Points):**")
                for c, w in boosters:
                    st.write(f"- {c} (Impact: **+{w:.2f} pts**)")
                    
            if minor:
                st.caption("*Minor factors (< 0.5 points) were filtered out.*")
                
    with col_llm:
        st.subheader("Step 2: GenAI Academic Advisor")
        with st.spinner("Generating personalized advisory report..."):
            # Gọi LLM sinh lời khuyên dựa trên Hồi quy
            # (Thay use_mock=False nếu đã có API Key)
            # Truyền selected_llm từ Sidebar vào hàm
            prompt, advice = adv_layer.get_advice(lime_reasons, predicted_score, actual_score, use_mock=False, provider=selected_llm)
            
            with st.expander("Show LLM Prompt Context", expanded=False):
                st.code(prompt, language="markdown")
                
            st.markdown(advice)

else:
    st.info(" Please click **'Extract Random Student'** from the sidebar to start the diagnostic.")