import streamlit as st
import pandas as pd
import random
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_layer import DataProcessor
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

st.set_page_config(page_title="SmartEdu Advisor", page_icon="🎓", layout="wide")

st.title("🎓 SmartEdu Advisor")
st.markdown("**Early Warning and Personalized Academic Advising System** *(Powered by XGBoost, LIME, and GenAI)*")

@st.cache_resource
def load_system_backend():
    dp = DataProcessor(config_path='config.yaml')
    X_train, X_test, y_train, y_test, feature_names = dp.prepare_data()
    
    xai = XAILayer(model_path='outputs/models/xgboost_model.pkl')
    adv = AdvisoryLayer()
    
    le = joblib.load('outputs/models/label_encoder.pkl')
    target_names = le.classes_.tolist()
    
    return X_train, X_test, y_test, feature_names, xai, adv, target_names

X_train, X_test, y_test, feature_names, xai_layer, adv_layer, target_names = load_system_backend()

st.sidebar.header("👨‍🎓 Student Profile")
st.sidebar.write("Simulate extracting a student record from the database.")

if st.sidebar.button("🎲 Extract Random Student", width="stretch"):
    st.session_state['student_idx'] = random.randint(0, len(X_test) - 1)

if 'student_idx' in st.session_state:
    idx = st.session_state['student_idx']
    student_data = X_test.iloc[idx]
    actual_status = target_names[y_test[idx]]
    
    st.sidebar.success(f"Analyzing Student Index: #{idx}")
    st.sidebar.info(f"Ground Truth: **{actual_status}**")
    
    st.subheader("📋 Student Data Overview (Z-score)")
    st.dataframe(student_data.to_frame().T, width="stretch")
    st.divider()
    
    if st.button("🚀 Run AI Diagnosis & Get Advice", type="primary", width="stretch"):
        probs = xai_layer.model.predict_proba([student_data.values])
        dropout_idx = target_names.index("Dropout") if "Dropout" in target_names else 0
        dropout_prob = float(probs[0][dropout_idx]) * 100 
        
        if dropout_prob >= 80:
            status_text = "🚨 Critical Risk - High chance of dropout"
            st.error(f"Status: **{status_text}**")
        elif dropout_prob >= 60:
            status_text = "⚠️ Academic Warning"
            st.warning(f"Status: **{status_text}**")
        elif dropout_prob >= 40:
            status_text = "🟡 Needs Improvement"
            st.warning(f"Status: **{status_text}**")
        elif dropout_prob >= 20:
            status_text = "✅ Stable Performance"
            st.success(f"Status: **{status_text}**")
        else:
            status_text = "🌟 Excellent Performance"
            st.success(f"Status: **{status_text}**")

        st.metric(label="📊 AI Predicted Risk Level (Dropout)", value=f"{dropout_prob:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Step 1: Risk Extraction (LIME XAI)")
            with st.spinner("Analyzing XGBoost model to find root causes..."):
                lime_reasons = xai_layer.explain_student(X_train, student_data, feature_names, target_names)
                
                negative, positive, minor = [], [], []
                for condition, weight in lime_reasons:
                    if abs(weight) < 0.02: 
                        minor.append((condition, weight))
                    elif weight > 0:
                        negative.append((condition, weight))
                    else:
                        positive.append((condition, weight))
                
                if negative:
                    st.error("**Major Barriers (Pushing towards Dropout):**")
                    for cond, weight in negative:
                        st.write(f"- {cond} (Impact: {weight:.2f})")
                
                if positive:
                    st.success("**Supportive Factors (Preventing Dropout):**")
                    for cond, weight in positive:
                        st.write(f"- {cond} (Impact: {weight:.2f})")
                        
                if minor:
                    st.caption("*Minor factors were filtered out to reduce noise.*")
        
        with col2:
            st.subheader("🤖 Step 2: GenAI Academic Advisor")
            with st.spinner("Generating personalized advisory report..."):
                raw_prompt, advice = adv_layer.get_advice(lime_reasons, status_text, actual_status)
                st.markdown(advice)