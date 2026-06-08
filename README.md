# Project_IntroAI_20252_group21
SmartEdu Advisor
SmartEdu Advisor là hệ thống cảnh báo sớm và tư vấn học tập cá nhân hóa dành cho sinh viên đại học. Dự án kết hợp các mô hình Machine Learning mạnh mẽ với khả năng giải thích của XAI và trí tuệ từ các mô hình ngôn ngữ lớn (LLM).

Tính năng chính
Cảnh báo sớm nguy cơ bỏ học (Dropout) của sinh viên dựa trên phân tích dữ liệu học thuật và phi học thuật.

Giải thích lý do đằng sau các dự đoán của AI bằng kỹ thuật LIME, giúp giảng viên và sinh viên hiểu rõ rào cản học tập.

Sử dụng LLM (Llama-3.3/Gemini) để tự động sinh báo cáo phân tích và tư vấn lộ trình học tập cá nhân hóa.

Giao diện Web trực quan giúp trích xuất dữ liệu sinh viên và chạy chẩn đoán theo thời gian thực.

Công nghệ sử dụng
XGBoost Classifier

LIME (Explainable AI)

Streamlit (Giao diện Web)

Groq API (Llama-3.3) & Google Gemini API

Scikit-learn, Imbalanced-learn (SMOTE)

Hướng dẫn cài đặt
Clone repository về máy:
git clone https://github.com/DozenBro/Project_IntroAI_20252_group21.git

Tạo môi trường ảo và cài đặt thư viện:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

Cấu hình file .env:
Tạo file .env trong thư mục gốc và điền các API Key cần thiết (GROQ_API_KEY, GEMINI_API_KEYS).

Chạy hệ thống:

Chạy pipeline xử lý dữ liệu: python main.py

Chạy giao diện Web: streamlit run app/app.py

Cấu trúc dự án
app/: Chứa giao diện Streamlit.

data/: Chứa dữ liệu đầu vào.

src/: Chứa các module xử lý dữ liệu, mô hình, XAI và advisory.

notebooks/: Chứa các file thử nghiệm và phân tích.

outputs/: Chứa mô hình đã train và các báo cáo tư vấn.

prompts/: Chứa các cấu hình Prompt cho LLM.
