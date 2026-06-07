import streamlit as st
import pandas as pd
import random
import sys
import os
import joblib

# Chỉ định đường dẫn để hệ thống nhận diện được thư mục src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_layer import DataProcessor
from src.xai_layer import XAILayer
from src.advisory_layer import AdvisoryLayer

# ==========================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="SmartEdu Advisor", page_icon="🎓", layout="wide")

st.title("🎓 SmartEdu Advisor")
st.markdown("""
**Hệ thống Cảnh báo sớm và Tư vấn học tập cá nhân hóa** *(Tích hợp XGBoost, Explainable AI (LIME) và Generative AI)*
""")

# ==========================================
# KHỞI TẠO HỆ THỐNG (CACHE ĐỂ CHẠY NHANH)
# ==========================================
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

# ==========================================
# THANH BÊN (SIDEBAR)
# ==========================================
st.sidebar.header("👨‍🎓 Hồ sơ Sinh viên")
st.sidebar.write("Mô phỏng lấy dữ liệu của một sinh viên từ hệ thống quản lý học tập (LMS).")

if st.sidebar.button("🎲 Trích xuất ngẫu nhiên 1 Sinh viên", width="stretch"):
    st.session_state['student_idx'] = random.randint(0, len(X_test) - 1)

# ==========================================
# KHU VỰC HIỂN THỊ CHÍNH (MAIN AREA)
# ==========================================
if 'student_idx' in st.session_state:
    idx = st.session_state['student_idx']
    student_data = X_test.iloc[idx]
    actual_status = target_names[y_test[idx]]
    
    st.sidebar.success(f"Đang phân tích Sinh viên Index: #{idx}")
    st.sidebar.info(f"Trạng thái thực tế: **{actual_status}**")
    
    # 🌟 NÂNG CẤP 2: HIỂN THỊ TỔNG QUAN DỮ LIỆU SINH VIÊN
    st.subheader("📋 Tổng quan Dữ liệu Sinh viên (Z-score)")
    st.dataframe(student_data.to_frame().T, width="stretch")
    st.divider()
    
    if st.button("🚀 Chẩn đoán AI & Sinh Lời khuyên", type="primary", width="stretch"):
        
        # --- SỬA LỖI TRÍCH XUẤT XÁC SUẤT TẠI ĐÂY ---
        probs = xai_layer.model.predict_proba([student_data.values])
        
        # Tự động tìm cột chứa nhãn "Dropout" (Thường là index 0)
        dropout_idx = target_names.index("Dropout") if "Dropout" in target_names else 0
        
        # Ép kiểu rõ ràng về số thực (float)
        dropout_prob = float(probs[0][dropout_idx]) * 100 
        
        # 🌟 NÂNG CẤP 1: PHÂN LOẠI MỨC ĐỘ RỦI RO
        if dropout_prob >= 80:
            status_text = "🚨 Nguy hiểm - Nguy cơ bị buộc thôi học"
            st.error(f"Trạng thái: **{status_text}**")
        elif dropout_prob >= 60:
            status_text = "⚠️ Cảnh cáo học tập"
            st.warning(f"Trạng thái: **{status_text}**")
        elif dropout_prob >= 40:
            status_text = "🟡 Cần cải thiện thêm"
            st.warning(f"Trạng thái: **{status_text}**")
        elif dropout_prob >= 20:
            status_text = "✅ Phong độ ổn định"
            st.success(f"Trạng thái: **{status_text}**")
        else:
            status_text = "🌟 Phong độ xuất sắc"
            st.success(f"Trạng thái: **{status_text}**")

        st.metric(label="📊 Đo lường mức độ rủi ro (AI Prediction)", value=f"{dropout_prob:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Bước 1: Trích xuất rủi ro (LIME XAI)")
            with st.spinner("Đang mở hộp đen XGBoost để tìm nguyên nhân..."):
                lime_reasons = xai_layer.explain_student(X_train, student_data, feature_names, target_names)
                
                # 🌟 NÂNG CẤP 3 & 4: TÁCH BẠCH YẾU TỐ TÍCH CỰC / TIÊU CỰC VÀ LỌC NHIỄU
                tieu_cuc, tich_cuc, yeu_to_nho = [], [], []
                for condition, weight in lime_reasons:
                    if abs(weight) < 0.05:
                        yeu_to_nho.append((condition, weight))
                    elif weight > 0:
                        tieu_cuc.append((condition, weight))
                    else:
                        tich_cuc.append((condition, weight))

                st.markdown("🔴 **Yếu tố Rào cản:**")
                if not tieu_cuc: st.write("*Không có rào cản lớn.*")
                for cond, w in tieu_cuc:
                    st.error(f"{cond} (Tác động: {w:.4f})")

                st.markdown("🟢 **Yếu tố Thuận lợi:**")
                if not tich_cuc: st.write("*Không có yếu tố nổi bật.*")
                for cond, w in tich_cuc:
                    st.success(f"{cond} (Tác động: {w:.4f})")

                with st.expander("🔍 Xem các yếu tố ảnh hưởng nhỏ khác"):
                    for cond, w in yeu_to_nho:
                        st.caption(f"- {cond} (Tác động: {w:.4f})")
                        
        with col2:
            st.subheader("💡 Bước 2: Báo cáo Tư vấn (GenAI)")
            with st.spinner("LLM đang soạn thảo lời khuyên cá nhân hóa..."):
                
                # 🌟 TRUYỀN status_text VÀO THAY VÌ dropout_prob ĐỂ AI CÓ NGỮ CẢNH
                # Lưu ý: Nhớ vào file advisory_layer.py sửa hàm get_advice để nhận biến status_text nhé!
                with st.spinner("LLM đang soạn thảo lời khuyên..."):
                    # Truyền thêm biến actual_status vào hàm get_advice
                    prompt, advice = adv_layer.get_advice(
                        lime_reasons, 
                        status_text=status_text, 
                        actual_status=actual_status, # <--- Dán chỗ này
                        use_mock=False
                    )
                    st.info(advice)
                
        with st.expander("🛠️ Xem cấu trúc Prompt đã gửi cho LLM (Dành cho Giảng viên)"):
            st.code(prompt, language='markdown')

else:
    st.warning("👈 Hãy bấm nút 'Trích xuất ngẫu nhiên' ở thanh bên trái để bắt đầu demo!")